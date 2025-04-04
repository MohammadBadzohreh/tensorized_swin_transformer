import torch
import torch.nn as nn
# Ensure you have your TCL implementation in this module
from Tensorized_Layers.TCL_CHANGED import TCL_CHANGED


class TensorizedPatchMerging(nn.Module):
    def __init__(
        self,
        input_size=(16, 56, 56, 4, 4, 3),
        in_embed_shape=(4, 4, 3),
        out_embed_shape=(4, 4, 6),
        bias=True,
        ignore_modes=(0, 1, 2),
        device='cuda'
    ):
        """
        Tensorized patch merging for Swin Transformer.

        Args:
            input_size (tuple): Overall input size tuple.
                (e.g., (num_patches, H, W, r1, r2, C)).
            in_embed_shape (tuple): Shape of each input patch embedding (r1, r2, C).
            out_embed_shape (tuple): Desired shape of each merged patch embedding (r'_1, r'_2, C').
            bias (bool): Whether to include bias in the linear reduction.
            ignore_modes (tuple): Modes to ignore in the tensorized layer.
            device (str): Device to use.
        """
        super(TensorizedPatchMerging, self).__init__()
        self.in_r1, self.in_r2, self.in_C = in_embed_shape
        self.out_r1, self.out_r2, self.out_C = out_embed_shape
        self.in_dim = self.in_r1 * self.in_r2 * self.in_C   # e.g., 4*4*3 = 48
        self.out_dim = self.out_r1 * self.out_r2 * self.out_C  # e.g., 4*4*6 = 96

        # Validate that standard merging holds: 4 * in_dim should equal 2 * out_dim.
        if 4 * self.in_dim != 2 * self.out_dim:
            raise ValueError(
                f"Dimension mismatch: expected out_dim = 2 * in_dim, got {self.out_dim} != {2 * self.in_dim}"
            )

        self.ignore_modes = ignore_modes
        self.bias = bias
        self.device = device
        self.input_size = input_size

        # Adjust the TCL input size for patch merging:
        # When merging, spatial dimensions (H, W) are halved, while the channel dimension is quadrupled.
        # For example, if input_size=(B, H, W, r1, r2, C), then the merged tensor shape will be:
        # (B, H/2, W/2, r1, r2, 4*C)
        self.tcl_input_size = (
            self.input_size[0],
            self.input_size[1] // 2,
            self.input_size[2] // 2,
            self.input_size[3],
            self.input_size[4],
            4 * self.input_size[5]
        )

        # Instantiate the TCL layer for patch merging.
        self.tcl = TCL_CHANGED(
            input_size=self.tcl_input_size,
            rank=out_embed_shape,
            ignore_modes=self.ignore_modes,
            bias=self.bias,
            device=self.device
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, H, W, in_r1, in_r2, in_C).

        Returns:
            Tensor: Merged patch embeddings of shape (B, H/2, W/2, out_r1, out_r2, out_C).
        """
        B, H, W, r1, r2, C = x.shape
        if (r1, r2, C) != (self.in_r1, self.in_r2, self.in_C):
            raise ValueError("Input patch embedding shape mismatch.")

        # Extract 2x2 neighboring patches along the spatial dimensions.
        top_left = x[:, 0::2, 0::2, :, :, :]
        bottom_left = x[:, 1::2, 0::2, :, :, :]
        top_right = x[:, 0::2, 1::2, :, :, :]
        bottom_right = x[:, 1::2, 1::2, :, :, :]

        # Concatenate along the channel dimension.
        x_merged = torch.cat(
            [top_left, bottom_left, top_right, bottom_right], dim=-1)
        # x_merged now has shape: (B, H/2, W/2, r1, r2, 4*C)

        # Apply the tensorized linear layer to merge patches.
        out = self.tcl(x_merged)
        return out

# -------------------------------
# Example usage:
# -------------------------------
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Example input tensor shape: (B, H, W, r1, r2, C)
#     # For instance, batch=1, H=56, W=56, in_embed_shape=(4, 4, 3)
#     x = torch.randn(1, 56, 56, 4, 4, 3, device=device)
#     # Overall input size for patch merging might be provided as follows:
#     input_size = (1, 56, 56, 4, 4, 3)
#     in_embed_shape = (4, 4, 3)
#     out_embed_shape = (4, 4, 6)  # desired merged embedding shape
#     patch_merging = TensorizedPatchMerging(
#         input_size=input_size,
#         in_embed_shape=in_embed_shape,
#         out_embed_shape=out_embed_shape,
#         bias=True,
#         ignore_modes=(0, 1, 2),
#         device=device
#     ).to(device)

#     out = patch_merging(x)
#     print("Merged patch embeddings shape:", out.shape)
