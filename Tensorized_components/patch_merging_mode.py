import torch
import torch.nn as nn
from Tensorized_Layers.TCL import TCL_extended as TCL_CHANGED


class TensorizedPatchMerging(nn.Module):
    """
    Tensorized patch merging for Swin Transformer with a configurable `mode`
    deciding which embedding axis (r1, r2, or C) receives the x4 factor that
    comes from concatenating the 2x2 spatial neighbors.

    mode = 0 -> shape after merge: (B, H/2, W/2, 4*r1,  r2,   C)
    mode = 1 -> shape after merge: (B, H/2, W/2,  r1,  4*r2,  C)
    mode = 2 -> shape after merge: (B, H/2, W/2,  r1,   r2,  4*C)
    """

    def __init__(
        self,
        input_size=(16, 56, 56, 4, 4, 3),
        in_embed_shape=(4, 4, 3),
        out_embed_shape=(4, 4, 6),
        mode: int = 2,
        bias=True,
        ignore_modes=(0, 1, 2),
        device="cuda",
    ):
        super().__init__()

        assert mode in (0, 1, 2), "mode must be in {0, 1, 2}"
        self.mode = mode

        self.in_r1, self.in_r2, self.in_C = in_embed_shape
        self.out_r1, self.out_r2, self.out_C = out_embed_shape
        self.in_dim = self.in_r1 * self.in_r2 * self.in_C
        self.out_dim = self.out_r1 * self.out_r2 * self.out_C

        # Keep your original constraint if you still need it.
        if 4 * self.in_dim != 2 * self.out_dim:
            raise ValueError(
                f"Dimension mismatch: expected out_dim = 2 * in_dim, "
                f"got {self.out_dim} != {2 * self.in_dim}"
            )

        self.ignore_modes = ignore_modes
        self.bias = bias
        self.device = device
        self.input_size = input_size

        # Compute the TCL input size & LN normalized shape based on mode
        self.tcl_input_size, norm_shape = self._compute_tcl_input_and_norm_shapes()

        self.norm = nn.LayerNorm(norm_shape)

        self.tcl = TCL_CHANGED(
            input_size=self.tcl_input_size,
            rank=out_embed_shape,
            ignore_modes=self.ignore_modes,
            bias=self.bias,
            device=self.device,
        )

    def _compute_tcl_input_and_norm_shapes(self):
        """
        Returns:
            tcl_input_size: tuple
            norm_shape: tuple (to pass into nn.LayerNorm)
        """
        B, H, W, r1, r2, C = self.input_size

        if self.mode == 0:
            # (B, H/2, W/2, 4*r1, r2, C)
            tcl_input_size = (
                B,
                H // 2,
                W // 2,
                4 * r1,
                r2,
                C,
            )
            norm_shape = (4 * self.in_r1, self.in_r2, self.in_C)

        elif self.mode == 1:
            # (B, H/2, W/2, r1, 4*r2, C)
            tcl_input_size = (
                B,
                H // 2,
                W // 2,
                r1,
                4 * r2,
                C,
            )
            norm_shape = (self.in_r1, 4 * self.in_r2, self.in_C)

        else:  # mode == 2
            # (B, H/2, W/2, r1, r2, 4*C)
            tcl_input_size = (
                B,
                H // 2,
                W // 2,
                r1,
                r2,
                4 * C,
            )
            norm_shape = (self.in_r1, self.in_r2, 4 * self.in_C)

        return tcl_input_size, norm_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, r1, r2, C)

        Returns:
            (B, H/2, W/2, out_r1, out_r2, out_C)
        """
        B, H, W, r1, r2, C = x.shape
        if (r1, r2, C) != (self.in_r1, self.in_r2, self.in_C):
            raise ValueError("Input patch embedding shape mismatch.")

        # 2x2 spatial merge -> concatenate along the embedding axis (last)
        top_left = x[:, 0::2, 0::2, :, :, :]
        bottom_left = x[:, 1::2, 0::2, :, :, :]
        top_right = x[:, 0::2, 1::2, :, :, :]
        bottom_right = x[:, 1::2, 1::2, :, :, :]

        # (B, H/2, W/2, r1, r2, 4*C)
        x_merged = torch.cat([top_left, bottom_left, top_right, bottom_right], dim=-1)

        H2, W2 = H // 2, W // 2

        if self.mode == 0:
            # (B, H/2, W/2, r1, r2, 4*C) -> (B, H/2, W/2, 4, r1, r2, C)
            x_merged = x_merged.view(B, H2, W2, r1, r2, 4, C)
            # -> (B, H/2, W/2, 4, r1, r2, C) -> (B, H/2, W/2, 4*r1, r2, C)
            x_merged = x_merged.permute(0, 1, 2, 5, 3, 4, 6).contiguous()
            x_merged = x_merged.view(B, H2, W2, 4 * r1, r2, C)

        elif self.mode == 1:
            # (B, H/2, W/2, r1, r2, 4*C) -> (B, H/2, W/2, r1, r2, 4, C)
            x_merged = x_merged.view(B, H2, W2, r1, r2, 4, C)
            # -> (B, H/2, W/2, r1, 4, r2, C) -> (B, H/2, W/2, r1, 4*r2, C)
            x_merged = x_merged.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
            x_merged = x_merged.view(B, H2, W2, r1, 4 * r2, C)

        else:  # mode == 2 (already in desired shape)
            # Keep (B, H/2, W/2, r1, r2, 4*C)
            pass

        x_merged = self.norm(x_merged)
        out = self.tcl(x_merged)
        return out
