import torch.nn as nn

from timm.models import register_model
from timm.models.vision_transformer_hybrid import (
    ConvStem,
    _create_vision_transformer_hybrid,
    _dd_from_kwargs,
)


@register_model
def vit_base_mci_56_s122(pretrained=False, **kwargs):
    backbone = ConvStem(
        channels=(768 // 4, 768 // 4, 768),
        stride=(1, 2, 2),
        kernel_size=(1, 2, 2),
        padding=0,
        in_chans=kwargs.get("in_chans", 3),
        act_layer=nn.GELU,
        **_dd_from_kwargs(**kwargs),
    )

    model_args = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        no_embed_class=True,
    )

    model = _create_vision_transformer_hybrid(
        "vit_base_mci_56_s122",
        backbone=backbone,
        embed_args=dict(proj=False),
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )
    return model