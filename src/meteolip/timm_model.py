""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
import types
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torch.nn import functional as F

try:
    import timm
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d
    from timm.layers import Mlp, to_2tuple
    from . import custom_timm_models  # registers custom timm models
except ImportError:
    timm = None

from .rope_2d import VisionRotaryEmbedding2D, apply_rotary_emb_2d
from .meteorology_tokenizer import MeteorologyTokenizer
from .utils import freeze_batch_norm_2d
from .utils_climate import (
    CLIMATE_INPUT_NUM_VARIABLES,
    CLIMATE_METADATA_TIME_HOUR_NUM_VARIABLES,
    CLIMATE_METADATA_TIME_DOY_NUM_VARIABLES,
    CLIMATE_METADATA_TIME_YEAR_NUM_VARIABLES,
    CLIMATE_METADATA_POS_NUM_VARIABLES,
    CLIMATE_INJECT_TIME_HOUR,
    CLIMATE_INJECT_TIME_DOY,
    CLIMATE_INJECT_TIME_YEAR,
    CLIMATE_INJECT_POSITION,
    USE_METEOROLOGY_TOKENIZER,
    USE_ROPE_2D,
)

class MetTokPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        # MetTok expects int, while timm gives tuple like (4, 4)
        mettok_patch_size = patch_size[0]
        assert patch_size[0] == patch_size[1], f"MetTok only supports square patches, got {patch_size}"

        self.mettok = MeteorologyTokenizer(
            patch_size=mettok_patch_size,
            embed_dim=embed_dim,
            mettok_dim=embed_dim // 8,
            num_heads=1,
            num_latent_queries=1,
            mettok_output_norm_type="batch_norm",
        )

    def forward(self, x):
        return self.mettok(x)

def _resolve_timm_attn_mask(
    seq_len: int,
    attn: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
):
    # Minimal local version of timm's resolve_self_attn_mask / maybe_add_mask.
    # For ViT/SigLIP image towers this is usually unused, but this keeps the
    # patched attention compatible with timm's Attention.forward signature.
    if is_causal:
        return attn.new_full((seq_len, seq_len), float("-inf")).triu_(1)

    if attn_mask is None:
        return None

    if attn_mask.dtype == torch.bool:
        # timm bool mask convention follows SDPA: True = keep.
        attn_bias = torch.zeros_like(attn_mask, dtype=attn.dtype)
        attn_bias.masked_fill_(~attn_mask, float("-inf"))
        return attn_bias

    return attn_mask.to(dtype=attn.dtype)


def _apply_timm_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: Optional[torch.Tensor],
    num_prefix_tokens: int,
):
    """
    q/k: [B, heads, N, head_dim]
    rope:
        mixed per-layer: [heads, num_patch_tokens, head_dim // 2]
        axial/shared:    [num_patch_tokens, head_dim // 2]
    """
    if rope is None:
        return q, k

    if num_prefix_tokens < 0:
        raise ValueError(f"num_prefix_tokens must be >= 0, got {num_prefix_tokens}")

    q_prefix = q[:, :, :num_prefix_tokens]
    k_prefix = k[:, :, :num_prefix_tokens]

    q_patch = q[:, :, num_prefix_tokens:]
    k_patch = k[:, :, num_prefix_tokens:]

    if q_patch.shape[2] != rope.shape[-2]:
        raise RuntimeError(
            f"RoPE token mismatch in timm attention: q/k have "
            f"{q_patch.shape[2]} patch tokens, but rope has {rope.shape[-2]} "
            f"positions. Check num_prefix_tokens, patch_drop, and grid size."
        )

    q_patch, k_patch = apply_rotary_emb_2d(q_patch, k_patch, rope)

    q = torch.cat([q_prefix, q_patch], dim=2)
    k = torch.cat([k_prefix, k_patch], dim=2)

    return q, k


def _timm_attention_forward_with_2d_rope(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Replacement forward for timm.layers.Attention-like modules.

    Supports the common timm ViT/SigLIP Attention fields:
      qkv, q_norm, k_norm, fused_attn, attn_drop, norm, gate, proj, proj_drop
    """
    B, N, C = x.shape

    gate = self.gate(x).sigmoid() if getattr(self, "gate", None) is not None else None

    if getattr(self, "qkv", None) is not None:
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
        # This is mostly for AttentionRope-like variants with split q/k/v.
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    q = self.q_norm(q)
    k = self.k_norm(k)

    rope = getattr(self, "_openclip_rope", None)
    num_prefix_tokens = getattr(self, "_openclip_rope_num_prefix_tokens", 1)

    q, k = _apply_timm_2d_rope(
        q,
        k,
        rope=rope,
        num_prefix_tokens=num_prefix_tokens,
    )

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=is_causal,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn_bias = _resolve_timm_attn_mask(
            seq_len=N,
            attn=attn,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)

    x = self.norm(x)

    if gate is not None:
        x = x * gate

    x = self.proj(x)
    x = self.proj_drop(x)
    return x

class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name: str,
            embed_dim: int,
            image_size: Union[int, Tuple[int, int]] = 224,
            pool: str = 'avg',
            proj: str = 'linear',
            proj_bias: bool = False,
            drop: float = 0.,
            drop_path: Optional[float] = None,
            patch_drop: Optional[float] = None,
            pretrained: bool = False,
            model_kwargs=None,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please install the latest timm (`pip install timm`) to use timm based models.")
        self.image_size = to_2tuple(image_size)

        if USE_ROPE_2D:
            rope_type = "mixed"
            rope_theta = 10.0
        else:
            rope_type = None
            rope_theta = None

        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.rope = None
        self.rope_num_prefix_tokens = 1

        if self.rope_type is not None:
            assert self.rope_type in ("mixed", "axial")
            assert patch_drop in (None, 0, 0.), (
                "patch_drop must be 0 when using RoPE unless timm PatchDropout returns "
                "kept patch indices and RoPE frequencies are indexed accordingly."
            )

        # setup kwargs that may not be common across all models
        timm_kwargs = dict(model_kwargs or {})
        if self.rope_type is not None:
            assert timm_kwargs.get("patch_drop_rate", 0.) == 0., (
                "timm patch_drop_rate must be 0 when using RoPE."
            )
            timm_kwargs["pos_embed"] = "none"
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            if self.rope_type is not None:
                assert patch_drop == 0., (
                    "patch_drop must be 0 when using RoPE unless kept patch indices "
                    "are used to index RoPE positions."
                )
            timm_kwargs['patch_drop_rate'] = patch_drop
        
        timm_kwargs['in_chans'] = CLIMATE_INPUT_NUM_VARIABLES
        if CLIMATE_INJECT_TIME_HOUR == True:
            timm_kwargs['in_chans'] += CLIMATE_METADATA_TIME_HOUR_NUM_VARIABLES
        if CLIMATE_INJECT_TIME_DOY == True:
            timm_kwargs['in_chans'] += CLIMATE_METADATA_TIME_DOY_NUM_VARIABLES
        if CLIMATE_INJECT_TIME_YEAR == True:
            timm_kwargs['in_chans'] += CLIMATE_METADATA_TIME_YEAR_NUM_VARIABLES
        if CLIMATE_INJECT_POSITION == True:
            timm_kwargs['in_chans'] += CLIMATE_METADATA_POS_NUM_VARIABLES

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            # if projection is explicitly set to "none" will be pass through from network trunk
            proj_dim = 0 if proj == 'none' else embed_dim
            self.trunk = timm.create_model(
                model_name,
                num_classes=proj_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

        self.use_mettok = USE_METEOROLOGY_TOKENIZER  # manually switch this to False when needed
        if self.use_mettok:
            self._replace_patch_embed_with_mettok()
        
        self._install_rope_on_timm_vit()

    def _replace_patch_embed_with_mettok(self):
        patch_embed = self.trunk.patch_embed

        img_size = patch_embed.img_size
        patch_size = patch_embed.patch_size
        embed_dim = self.trunk.embed_dim

        self.trunk.patch_embed = MetTokPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NCHW',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply norm layer to all intermediates
            intermediates_only: Only return intermediate features
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both prefix and spatial intermediate tokens
        Returns:
        """
        extra_args = {}
        if output_extra_tokens:
            extra_args['return_prefix_tokens'] = True
        self._set_rope_for_timm_forward(x)
        trunk_output = self.trunk.forward_intermediates(
                x,
                indices=indices,
                intermediates_only=intermediates_only,
                norm=normalize_intermediates,
                stop_early=stop_early,
                output_fmt=output_fmt,
                **extra_args,
            )

        return_dict = {}
        intermediates = trunk_output if intermediates_only else trunk_output[1]
        if output_extra_tokens and intermediates and isinstance(intermediates[0], tuple):
            intermediates_prefix = [xi[1] for xi in intermediates]
            intermediates = [xi[0] for xi in intermediates]
            return_dict['image_intermediates_prefix'] = intermediates_prefix

        return_dict['image_intermediates'] = intermediates
        if intermediates_only:
            return return_dict

        image_features = self.trunk.forward_head(trunk_output[0])  # run through timm pooling / projection
        image_features = self.head(image_features) # run through adapter pooling / projection
        return_dict['image_features'] = image_features
        return return_dict

    def set_input_size(self, image_size: Union[int, Tuple[int, int]]):
        """Set the input image size for the model after initialization.

        This method attempts to call set_input_size on the underlying timm model
        if it supports dynamic input size adjustment.

        Args:
            image_size: New image size as int (square) or tuple (h, w)
        """
        self.image_size = to_2tuple(image_size)

        # Check if the underlying timm model has set_input_size method
        if hasattr(self.trunk, 'set_input_size'):
            self.trunk.set_input_size(image_size)
        else:
            logging.info(f"timm model {self.trunk.__class__.__name__} does not have set_input_size method. Skipping.")

    def forward(self, x):
        self._set_rope_for_timm_forward(x)
        x = self.trunk(x)
        x = self.head(x)
        return x

    def _infer_num_prefix_tokens(self) -> int:
        if hasattr(self.trunk, "num_prefix_tokens"):
            return int(self.trunk.num_prefix_tokens)

        num_prefix_tokens = 0

        cls_token = getattr(self.trunk, "cls_token", None)
        if cls_token is not None:
            num_prefix_tokens += cls_token.shape[1]

        reg_token = getattr(self.trunk, "reg_token", None)
        if reg_token is not None:
            num_prefix_tokens += reg_token.shape[1]

        dist_token = getattr(self.trunk, "dist_token", None)
        if dist_token is not None:
            num_prefix_tokens += dist_token.shape[1]

        return num_prefix_tokens


    def _infer_patch_grid_size(self):
        patch_embed = getattr(self.trunk, "patch_embed", None)
        if patch_embed is None:
            raise RuntimeError("RoPE for TimmModel currently requires trunk.patch_embed.")

        if hasattr(patch_embed, "grid_size"):
            grid_size = patch_embed.grid_size
            return int(grid_size[0]), int(grid_size[1])

        img_size = to_2tuple(getattr(patch_embed, "img_size", self.image_size))
        patch_size = to_2tuple(getattr(patch_embed, "patch_size", 1))
        return img_size[0] // patch_size[0], img_size[1] // patch_size[1]


    def _infer_runtime_patch_grid_size(self, x: torch.Tensor):
        patch_embed = getattr(self.trunk, "patch_embed", None)
        if patch_embed is None:
            raise RuntimeError("RoPE for TimmModel currently requires trunk.patch_embed.")

        patch_size = to_2tuple(getattr(patch_embed, "patch_size", 1))

        grid_h = x.shape[-2] // patch_size[0]
        grid_w = x.shape[-1] // patch_size[1]

        return int(grid_h), int(grid_w)


    def _install_rope_on_timm_vit(self):
        if self.rope_type is None:
            self.rope = None
            return

        if not hasattr(self.trunk, "blocks"):
            raise RuntimeError(
                f"RoPE currently supports timm ViT-style models with trunk.blocks. "
                f"Got {self.trunk.__class__.__name__}."
            )

        if len(self.trunk.blocks) == 0:
            raise RuntimeError("Cannot install RoPE: trunk.blocks is empty.")

        first_block = self.trunk.blocks[0]
        if not hasattr(first_block, "attn"):
            raise RuntimeError("Cannot install RoPE: trunk.blocks[0] has no .attn module.")

        first_attn = first_block.attn

        if not hasattr(first_attn, "num_heads") or not hasattr(first_attn, "head_dim"):
            raise RuntimeError(
                "Cannot install RoPE: timm attention module does not expose "
                ".num_heads and .head_dim."
            )

        grid_size = self._infer_patch_grid_size()
        depth = len(self.trunk.blocks)
        num_heads = int(first_attn.num_heads)
        head_dim = int(first_attn.head_dim)

        self.rope_num_prefix_tokens = self._infer_num_prefix_tokens()

        self.rope = VisionRotaryEmbedding2D(
            head_dim=head_dim,
            num_heads=num_heads,
            depth=depth,
            grid_size=grid_size,
            rope_type=self.rope_type,
            theta=self.rope_theta,
        )

        for block in self.trunk.blocks:
            attn = getattr(block, "attn", None)

            if attn is None:
                raise RuntimeError("Cannot install RoPE: one timm block has no .attn module.")

            required = ["num_heads", "head_dim", "q_norm", "k_norm", "attn_drop", "proj", "proj_drop"]
            missing = [name for name in required if not hasattr(attn, name)]
            if missing:
                raise RuntimeError(
                    f"Cannot install RoPE on attention module {attn.__class__.__name__}; "
                    f"missing fields: {missing}"
                )

            if not hasattr(attn, "qkv") and not all(hasattr(attn, n) for n in ["q_proj", "k_proj", "v_proj"]):
                raise RuntimeError(
                    f"Cannot install RoPE on attention module {attn.__class__.__name__}; "
                    f"expected either .qkv or .q_proj/.k_proj/.v_proj."
                )

            attn._openclip_rope = None
            attn._openclip_rope_num_prefix_tokens = self.rope_num_prefix_tokens
            attn.forward = types.MethodType(_timm_attention_forward_with_2d_rope, attn)


    def _set_rope_for_timm_forward(self, x: torch.Tensor):
        if self.rope is None:
            return

        grid_h, grid_w = self._infer_runtime_patch_grid_size(x)

        rope = self.rope(
            grid_h=grid_h,
            grid_w=grid_w,
            device=x.device,
        )

        for i, block in enumerate(self.trunk.blocks):
            if rope.ndim == 4:
                block.attn._openclip_rope = rope[i]
            else:
                block.attn._openclip_rope = rope