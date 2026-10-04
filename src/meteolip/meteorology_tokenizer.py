from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.nn.attention import SDPBackend, sdpa_kernel

from .utils_climate import (
    CLIMATE_USE_METEOROLOGY_DATA,
    CLIMATE_INJECT_TIME_HOUR,
    CLIMATE_INJECT_TIME_DOY,
    CLIMATE_INJECT_TIME_YEAR,
    CLIMATE_INJECT_POSITION,
    CLIMATE_INPUT_NUM_VARIABLES,
    CLIMATE_METADATA_TIME_HOUR_NUM_VARIABLES,
    CLIMATE_METADATA_TIME_DOY_NUM_VARIABLES,
    CLIMATE_METADATA_TIME_YEAR_NUM_VARIABLES,
    CLIMATE_METADATA_POS_NUM_VARIABLES,
)


class MeteorologyTokenizer(nn.Module):
    """
    MetTok (Meteorology Tokenizer)

    Expected channel order:
        [physical variables, optional position metadata, optional time metadata]

    Input:
        x: (B, C, H, W)

    Output:
        if num_latent_queries == 1:
            z: (B, N, D)

        if num_latent_queries > 1:
            z: (B, N * K, D)

    where:
        B = batch size
        C = total input channels
        H, W = spatial resolution
        N = number of spatial patches
        D = embedding dimension
        K = number of latent queries per patch
    """

    def __init__(
        self,
        num_variables: int = CLIMATE_INPUT_NUM_VARIABLES,
        inject_position: bool = CLIMATE_INJECT_POSITION,
        inject_time_hour: bool = CLIMATE_INJECT_TIME_HOUR,
        inject_time_doy: bool = CLIMATE_INJECT_TIME_DOY,
        inject_time_year: bool = CLIMATE_INJECT_TIME_YEAR,
        patch_size: int = 4,
        embed_dim: int = 768,
        mettok_dim: int = 256,
        num_heads: int = 12,
        num_latent_queries: int = 1,
        attention_dropout: float = 0.0,
        metadata_dropout: float = 0.0,
        output_dropout: float = 0.0,
        use_query_residual: bool = True,
        mettok_output_norm_type: str = "identity",
        add_variable_embedding: bool = True,
        use_meteorology_data: bool = CLIMATE_USE_METEOROLOGY_DATA,
    ):
        super().__init__()

        if mettok_dim % num_heads != 0:
            raise ValueError(
                f"mettok_dim={mettok_dim} must be divisible by num_heads={num_heads}."
            )

        if num_latent_queries < 1:
            raise ValueError(
                f"num_latent_queries must be >= 1, got {num_latent_queries}."
            )

        self.num_variables = num_variables
        self.use_meteorology_data = use_meteorology_data
        self.inject_position = inject_position
        self.inject_time_hour = inject_time_hour
        self.inject_time_doy = inject_time_doy
        self.inject_time_year = inject_time_year

        self.position_metadata_dim = CLIMATE_METADATA_POS_NUM_VARIABLES if inject_position else 0
        self.time_metadata_dim = 0
        if inject_time_hour: self.time_metadata_dim += CLIMATE_METADATA_TIME_HOUR_NUM_VARIABLES
        if inject_time_doy:  self.time_metadata_dim += CLIMATE_METADATA_TIME_DOY_NUM_VARIABLES
        if inject_time_year: self.time_metadata_dim += CLIMATE_METADATA_TIME_YEAR_NUM_VARIABLES
        
        self.metadata_dim = self.position_metadata_dim + self.time_metadata_dim

        # At least one source of information must be enabled.
        if not self.use_meteorology_data and self.metadata_dim == 0:
            raise ValueError(
                "Invalid MeteorologyTokenizer configuration: "
                "use_meteorology_data=False while both position and temporal "
                "metadata are disabled. At least meteorology data or metadata "
                "must be enabled."
            )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mettok_dim = mettok_dim
        self.num_heads = num_heads
        self.num_latent_queries = num_latent_queries
        self.use_query_residual = use_query_residual

        self.expected_input_channels = self.num_variables + self.metadata_dim

        scale = mettok_dim ** -0.5

        # ------------------------------------------------------------------
        # Variable-wise depthwise patch embedding.
        # This embeds each physical variable independently for MetTok cross-attention.
        #
        # (B, V, H, W)
        # -> (B, V * D, H / P, W / P)
        # -> (B, N, V, D)
        # ------------------------------------------------------------------
        if self.use_meteorology_data:
            self.variable_patch_embed = nn.Conv2d(
                in_channels=self.num_variables,
                out_channels=self.num_variables * mettok_dim,
                kernel_size=patch_size,
                stride=patch_size,
                groups=self.num_variables,
                bias=True,
            )

            if add_variable_embedding:
                self.variable_embedding = nn.Parameter(
                    scale * torch.randn(self.num_variables, mettok_dim)
                )
            else:
                self.variable_embedding = None

        else:
            self.variable_patch_embed = None
            self.variable_embedding = None

        # ------------------------------------------------------------------
        # Metadata patch embedding.
        #
        # This is a standard, non-grouped Conv2d.
        # It learns both spatial patch aggregation and metadata-channel mixing.
        #
        # (B, M, H, W)
        # -> (B, D, H / P, W / P)
        # -> (B, N, D)
        # ------------------------------------------------------------------
        if self.metadata_dim > 0:
            self.metadata_patch_embed = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.metadata_dim,
                    out_channels=mettok_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=True,
                ),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=mettok_dim,
                    out_channels=mettok_dim,
                    kernel_size=1,
                    bias=True,
                ),
            )
            self.metadata_scale = nn.Parameter(scale * torch.ones(mettok_dim))
            self.metadata_dropout = nn.Dropout(metadata_dropout)
        else:
            self.metadata_patch_embed = None
            self.metadata_scale = None
            self.metadata_dropout = None

        self.query_embedding = nn.Parameter(scale * torch.randn(num_latent_queries, mettok_dim))

        if self.use_meteorology_data:
            self.query_norm = nn.LayerNorm(mettok_dim)
            self.variable_norm = nn.LayerNorm(mettok_dim)

            self.cross_attention = nn.MultiheadAttention(
                embed_dim=mettok_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
        else:
            self.query_norm = None
            self.variable_norm = None
            self.cross_attention = None

        self.output_dropout = nn.Dropout(output_dropout)

        if mettok_output_norm_type == "batch_norm":
            self.mettok_output_norm = nn.BatchNorm1d(mettok_dim)
        elif mettok_output_norm_type == "layer_norm":
            self.mettok_output_norm = nn.LayerNorm(mettok_dim)
        elif mettok_output_norm_type == "identity":
            self.mettok_output_norm = nn.Identity()
        else:
            raise ValueError(
                f"Unknown mettok_output_norm_type: {mettok_output_norm_type}. "
                "Expected 'batch_norm', 'layer_norm', or 'identity'."
            )
        
        self.output_projection = nn.Linear(mettok_dim, embed_dim)

    def _split_input(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Splits input into physical variables and metadata channels.

        Args:
            x:
                Tensor of shape (B, C, H, W)

        Returns:
            climate_x:
                Tensor of shape (B, V, H, W)

            metadata_x:
                Tensor of shape (B, M, H, W), or None
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}."
            )

        _, C, _, _ = x.shape

        if C != self.expected_input_channels:
            raise ValueError(
                "Unexpected number of input channels. "
                f"Got C={C}, expected C={self.expected_input_channels}. "
                f"num_variables={self.num_variables}, "
                f"position_metadata_dim={self.position_metadata_dim}, "
                f"time_metadata_dim={self.time_metadata_dim}."
            )

        if self.use_meteorology_data:
            climate_x = x[:, :self.num_variables]
        else:
            climate_x = None

        if self.metadata_dim > 0:
            metadata_x = x[:, self.num_variables:]
        else:
            metadata_x = None

        return climate_x, metadata_x

    def _embed_variables(self, climate_x: torch.Tensor) -> torch.Tensor:
        """
        Applies variable-wise depthwise patch embedding.

        Args:
            climate_x:
                Tensor of shape (B, V, H, W)

        Returns:
            variable_tokens:
                Tensor of shape (B, N, V, D)
        """

        if not self.use_meteorology_data:
            raise RuntimeError(
                "_embed_variables() was called while "
                "use_meteorology_data=False."
            )

        if climate_x is None:
            raise ValueError(
                "Meteorology data is enabled but climate_x is None."
            )

        B, V, H, W = climate_x.shape

        if V != self.num_variables:
            raise ValueError(
                f"Expected V={self.num_variables}, got V={V}."
            )

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size={self.patch_size}. "
                f"Got H={H}, W={W}."
            )

        Hp = H // self.patch_size
        Wp = W // self.patch_size
        N = Hp * Wp

        x = self.variable_patch_embed(climate_x) # (B, V * D, Hp, Wp)

        x = x.reshape(B, self.num_variables, self.mettok_dim, Hp, Wp) # (B, V, D, Hp, Wp)

        x = x.permute(0, 3, 4, 1, 2).contiguous() # (B, Hp, Wp, V, D)

        variable_tokens = x.reshape(B, N, self.num_variables, self.mettok_dim) # (B, N, V, D)

        if self.variable_embedding is not None:
            variable_tokens = variable_tokens + self.variable_embedding[None, None, :, :]
        # (B, N, V, D)

        return variable_tokens

    def _embed_metadata(
        self,
        metadata_x: Optional[torch.Tensor],
        expected_num_patches: int,
    ) -> Optional[torch.Tensor]:
        """
        Applies learnable metadata patch embedding.

        Args:
            metadata_x:
                Tensor of shape (B, M, H, W), or None.

            expected_num_patches:
                Expected number of spatial patches N.

        Returns:
            metadata_tokens:
                Tensor of shape (B, N, D), or None.
        """
        if self.metadata_dim == 0:
            return None

        if metadata_x is None:
            raise ValueError(
                "Metadata channels are enabled but metadata_x is None."
            )

        if metadata_x.ndim != 4:
            raise ValueError(
                f"Expected metadata with shape (B, M, H, W), got {tuple(metadata_x.shape)}."
            )

        B, M, H, W = metadata_x.shape

        if M != self.metadata_dim:
            raise ValueError(
                f"Expected metadata_dim={self.metadata_dim}, got M={M}."
            )

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size={self.patch_size}. "
                f"Got H={H}, W={W}."
            )

        metadata_tokens = self.metadata_patch_embed(metadata_x) # (B, D, Hp, Wp)

        metadata_tokens = metadata_tokens.flatten(2).transpose(1, 2).contiguous() # (B, N, D)

        if metadata_tokens.shape[1] != expected_num_patches:
            raise ValueError(
                "Metadata patch count does not match variable patch count. "
                f"Got metadata N={metadata_tokens.shape[1]}, "
                f"expected N={expected_num_patches}."
            )

        # metadata_tokens = self.metadata_norm(metadata_tokens)
        metadata_tokens = self.metadata_dropout(metadata_tokens)

        return metadata_tokens

    def _build_queries(
        self,
        batch_size: int,
        num_patches: int,
        metadata_tokens: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Builds latent queries, optionally conditioned on metadata.

        Args:
            batch_size:
                Batch size B.

            num_patches:
                Number of spatial patches N.

            metadata_tokens:
                Tensor of shape (B, N, D), or None.

            device:
                Target device.

        Returns:
            query:
                Tensor of shape (B, N, K, D)
        """

        query = self.query_embedding[None, None, :, :].expand(
            batch_size,
            num_patches,
            self.num_latent_queries,
            self.mettok_dim,
        )
        # (B, N, K, D)

        if metadata_tokens is not None:
            query = query + self.metadata_scale[None, None, None, :] * metadata_tokens[:, :, None, :] # (B, N, K, D)

        return query

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x:
                Tensor of shape (B, C, H, W)

            return_attention:
                If True, returns cross-attention weights over variables.

        Returns:
            If return_attention is False:
                z

            If return_attention is True:
                z, attention_weights

            z:
                Shape (B, N, D) if num_latent_queries == 1.
                Shape (B, N * K, D) if num_latent_queries > 1.

            attention_weights:
                Shape (B, N, num_heads, K, V).
        """

        if return_attention and not self.use_meteorology_data:
            raise ValueError(
                "return_attention=True is only supported when "
                "use_meteorology_data=True, because there are no "
                "meteorological variables to attend over otherwise."
            )

        climate_x, metadata_x = self._split_input(x)

        B, _, H, W = x.shape

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size={self.patch_size}. "
                f"Got H={H}, W={W}."
            )

        Hp = H // self.patch_size
        Wp = W // self.patch_size
        N = Hp * Wp
        K = self.num_latent_queries
        D = self.mettok_dim

        
        if self.use_meteorology_data: # Meteorological variable tokens
            variable_tokens = self._embed_variables(climate_x) # (B, N, V, D)

            _, variable_N, V, variable_D = variable_tokens.shape

            if variable_N != N:
                raise RuntimeError(
                    f"Variable patch count mismatch: {variable_N} vs {N}."
                )

            if variable_D != D:
                raise RuntimeError(
                    f"Variable embedding dimension mismatch: "
                    f"{variable_D} vs {D}."
                )

        else:
            variable_tokens = None
            V = 0

        # Metadata
        metadata_tokens = self._embed_metadata(metadata_x=metadata_x, expected_num_patches=N)
        # (B, N, D) or None

        query = self._build_queries(
            batch_size=B,
            num_patches=N,
            metadata_tokens=metadata_tokens,
            device=x.device,
        )
        # (B, N, K, D)

        query_flat = query.reshape(B * N, K, D) # (B * N, K, D)
        
        if self.use_meteorology_data: # Meteorology-enabled path
            variable_tokens_flat = variable_tokens.reshape(B * N, V, D) # (B * N, V, D)

            query_flat_norm = self.query_norm(query_flat)
            key_flat_norm = self.variable_norm(variable_tokens_flat)

            with sdpa_kernel(backends=[SDPBackend.MATH]):
                attention_output, attention_weights = self.cross_attention(
                    query=query_flat_norm,
                    key=key_flat_norm,
                    value=variable_tokens_flat,
                    need_weights=return_attention,
                    average_attn_weights=False,
                )

            # attention_output: (B * N, K, D)
            # attention_weights: (B * N, num_heads, K, V), if returned

            attention_output = self.output_dropout(attention_output)

            if self.use_query_residual:
                z = query_flat + attention_output
            else:
                z = attention_output
        else: # Metadata-only path
            z = query_flat
            attention_weights = None

        # --------------------------------------------------------------
        # Output normalization
        # --------------------------------------------------------------
        # z: (B * N, K, mettok_dim)
        if isinstance(self.mettok_output_norm, nn.BatchNorm1d):
            z = self.mettok_output_norm(z.reshape(-1, z.size(-1))).reshape_as(z)
        elif isinstance(self.mettok_output_norm, nn.LayerNorm):
            z = self.mettok_output_norm(z)
        elif isinstance(self.mettok_output_norm, nn.Identity):
            pass
        else:
            raise TypeError(f"Unsupported mettok_output_norm type: {type(self.mettok_output_norm)}")
        
        # --------------------------------------------------------------
        # Restore patch/query dimensions
        # --------------------------------------------------------------
        z = z.reshape(B, N, K, D) # (B, N, K, mettok_dim)

        if K == 1:
            z = z[:, :, 0, :] # (B, N, mettok_dim)
        else:
            z = z.reshape(B, N * K, D) # (B, N * K, mettok_dim)

        z = self.output_projection(z) # (B, N, embed_dim) or (B, N * K, embed_dim)

        if not return_attention:
            return z

        attention_weights = attention_weights.reshape(
            B,
            N,
            self.num_heads,
            K,
            V,
        )
        # (B, N, num_heads, K, V)

        return z, attention_weights