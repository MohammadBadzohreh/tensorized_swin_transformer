import torch
import torch.nn as nn
from Tensorized_Layers.TCL import TCL as TCL_CHANGED


class TensorizedPatchMerging(nn.Module):
    """
    mode = 0 -> x_merged: (B, H/2, W/2, 4*r1,  r2,   C)
    mode = 1 -> x_merged: (B, H/2, W/2,  r1,  4*r2,  C)
    mode = 2 -> x_merged: (B, H/2, W/2,  r1,   r2,  4*C)
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

        if 4 * self.in_dim != 2 * self.out_dim:
            raise ValueError(
                f"Dimension mismatch: expected out_dim = 2 * in_dim, "
                f"got {self.out_dim} != {2 * self.in_dim}"
            )

        self.ignore_modes = ignore_modes
        self.bias = bias
        self.device = device
        self.input_size = input_size

        # Compute TCL input size + LN shape based on mode
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
        B, H, W, r1, r2, C = self.input_size

        if self.mode == 0:
            # (B, H/2, W/2, 4*r1, r2, C)
            tcl_input_size = (B, H // 2, W // 2, 4 * r1, r2, C)
            norm_shape = (4 * self.in_r1, self.in_r2, self.in_C)
        elif self.mode == 1:
            # (B, H/2, W/2, r1, 4*r2, C)
            tcl_input_size = (B, H // 2, W // 2, r1, 4 * r2, C)
            norm_shape = (self.in_r1, 4 * self.in_r2, self.in_C)
        else:  # mode == 2
            # (B, H/2, W/2, r1, r2, 4*C)
            tcl_input_size = (B, H // 2, W // 2, r1, r2, 4 * C)
            norm_shape = (self.in_r1, self.in_r2, 4 * self.in_C)

        return tcl_input_size, norm_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, r1, r2, C)
        """
        B, H, W, r1, r2, C = x.shape
        if (r1, r2, C) != (self.in_r1, self.in_r2, self.in_C):
            raise ValueError("Input patch embedding shape mismatch.")

        # 2x2 spatial neighbors
        tl = x[:, 0::2, 0::2, :, :, :]  # top-left
        bl = x[:, 1::2, 0::2, :, :, :]  # bottom-left
        tr = x[:, 0::2, 1::2, :, :, :]  # top-right
        br = x[:, 1::2, 1::2, :, :, :]  # bottom-right

        # Decide which dim to concatenate on:
        # dims: B=0, H=1, W=2, r1=3, r2=4, C=5
        if self.mode == 0:   # grow r1
            cat_dim = 3
        elif self.mode == 1: # grow r2
            cat_dim = 4
        else:                # grow C
            cat_dim = 5

        x_merged = torch.cat([tl, bl, tr, br], dim=cat_dim)

        x_merged = self.norm(x_merged)
        out = self.tcl(x_merged)
        return out
