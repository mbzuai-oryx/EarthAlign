from typing import Optional, Tuple

import torch
from torch import nn


def init_t_xy(grid_w: int, grid_h: int, device: Optional[torch.device] = None):
    """
    Row-major spatial coordinates for flattened patch tokens.

    Token index i corresponds to:
        x = i % grid_w
        y = i // grid_w
    """
    t = torch.arange(grid_w * grid_h, dtype=torch.float32, device=device)
    t_x = (t % grid_w).float()
    t_y = torch.div(t, grid_w, rounding_mode="floor").float()
    return t_x, t_y


def init_random_2d_freqs(
    head_dim: int,
    num_heads: int,
    theta: float = 10.0,
    rotate: bool = True,
):
    """
    RoPE-Mixed learnable frequency initialization.

    Returns:
        freqs: [2, num_heads, head_dim // 2]
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"2D RoPE requires head_dim divisible by 4, got {head_dim}."
        )

    mag = 1.0 / (
        theta ** (torch.arange(0, head_dim, 4).float() / head_dim)
    )  # [head_dim // 4]

    freqs_x = []
    freqs_y = []

    for _ in range(num_heads):
        angle = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)

        fx = torch.cat(
            [
                mag * torch.cos(angle),
                mag * torch.cos(torch.pi / 2 + angle),
            ],
            dim=-1,
        )  # [head_dim // 2]

        fy = torch.cat(
            [
                mag * torch.sin(angle),
                mag * torch.sin(torch.pi / 2 + angle),
            ],
            dim=-1,
        )  # [head_dim // 2]

        freqs_x.append(fx)
        freqs_y.append(fy)

    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)

    return torch.stack([freqs_x, freqs_y], dim=0)


def compute_mixed_cis(
    freqs: torch.Tensor,
    t_x: torch.Tensor,
    t_y: torch.Tensor,
):
    """
    Args:
        freqs: [layers, 2, heads, head_dim // 2]
        t_x: [num_patches]
        t_y: [num_patches]

    Returns:
        freqs_cis: [layers, heads, num_patches, head_dim // 2]
    """
    # [layers, heads, 1, head_dim // 2]
    freqs_x = freqs[:, 0].unsqueeze(2)
    freqs_y = freqs[:, 1].unsqueeze(2)

    # [1, 1, num_patches, 1]
    t_x = t_x.view(1, 1, -1, 1)
    t_y = t_y.view(1, 1, -1, 1)

    # Keep this in fp32 for numerical stability.
    with torch.cuda.amp.autocast(enabled=False):
        phase = t_x * freqs_x.float() + t_y * freqs_y.float()
        freqs_cis = torch.polar(torch.ones_like(phase), phase)

    return freqs_cis


def compute_axial_cis(
    head_dim: int,
    grid_w: int,
    grid_h: int,
    theta: float = 100.0,
    device: Optional[torch.device] = None,
):
    """
    RoPE-Axial frequencies.

    Returns:
        freqs_cis: [num_patches, head_dim // 2]
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"2D RoPE requires head_dim divisible by 4, got {head_dim}."
        )

    freqs_x = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 4, dtype=torch.float32, device=device)
            / head_dim
        )
    )
    freqs_y = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 4, dtype=torch.float32, device=device)
            / head_dim
        )
    )

    t_x, t_y = init_t_xy(grid_w, grid_h, device=device)

    phase_x = torch.outer(t_x, freqs_x)
    phase_y = torch.outer(t_y, freqs_y)

    cis_x = torch.polar(torch.ones_like(phase_x), phase_x)
    cis_y = torch.polar(torch.ones_like(phase_y), phase_y)

    return torch.cat([cis_x, cis_y], dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Args:
        freqs_cis:
            axial: [N, head_dim // 2]
            mixed: [heads, N, head_dim // 2]
        x:
            complex q/k: [B, heads, N, head_dim // 2]
    """
    if freqs_cis.ndim == 2:
        return freqs_cis[None, None, :, :]

    if freqs_cis.ndim == 3:
        return freqs_cis[None, :, :, :]

    raise ValueError(f"Unexpected freqs_cis shape: {freqs_cis.shape}")


def apply_rotary_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    """
    Args:
        q: [B, heads, N, head_dim]
        k: [B, heads, N, head_dim]
        freqs_cis:
            axial: [N, head_dim // 2]
            mixed: [heads, N, head_dim // 2]

    Returns:
        rotated q, rotated k with original dtype.
    """
    q_complex = torch.view_as_complex(
        q.float().reshape(*q.shape[:-1], -1, 2)
    )
    k_complex = torch.view_as_complex(
        k.float().reshape(*k.shape[:-1], -1, 2)
    )

    freqs_cis = _reshape_for_broadcast(freqs_cis, q_complex)
    freqs_cis = freqs_cis.to(device=q_complex.device)

    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

    return q_out.type_as(q), k_out.type_as(k)


class VisionRotaryEmbedding2D(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        depth: int,
        grid_size: Tuple[int, int],
        rope_type: str = "mixed",
        theta: Optional[float] = None,
    ):
        super().__init__()

        if rope_type not in ("mixed", "axial"):
            raise ValueError(f"Unsupported rope_type: {rope_type}")

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.depth = depth
        self.grid_size = grid_size  # [grid_h, grid_w]
        self.rope_type = rope_type

        grid_h, grid_w = grid_size

        if rope_type == "mixed":
            self.theta = 10.0 if theta is None else theta

            freqs = [
                init_random_2d_freqs(
                    head_dim=head_dim,
                    num_heads=num_heads,
                    theta=self.theta,
                )
                for _ in range(depth)
            ]
            freqs = torch.stack(freqs, dim=0)
            # [layers, 2, heads, head_dim // 2]
            self.freqs = nn.Parameter(freqs)

            t_x, t_y = init_t_xy(grid_w, grid_h)
            self.register_buffer("base_t_x", t_x, persistent=False)
            self.register_buffer("base_t_y", t_y, persistent=False)

        else:
            self.theta = 100.0 if theta is None else theta

            freqs_cis = compute_axial_cis(
                head_dim=head_dim,
                grid_w=grid_w,
                grid_h=grid_h,
                theta=self.theta,
            )
            self.register_buffer("base_freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ):
        if self.rope_type == "mixed":
            if (grid_h, grid_w) == self.grid_size:
                t_x = self.base_t_x.to(device=device)
                t_y = self.base_t_y.to(device=device)
            else:
                t_x, t_y = init_t_xy(grid_w, grid_h, device=device)

            return compute_mixed_cis(
                self.freqs.to(device=device),
                t_x,
                t_y,
            )

        if (grid_h, grid_w) == self.grid_size:
            return self.base_freqs_cis.to(device=device)

        return compute_axial_cis(
            head_dim=self.head_dim,
            grid_w=grid_w,
            grid_h=grid_h,
            theta=self.theta,
            device=device,
        )