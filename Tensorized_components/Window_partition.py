import torch
import torch.nn as nn
class WindowPartition(nn.Module):
    """
    Utility module for partitioning and reversing windows in a patch grid.

    Input shape: (B, H, W, *embed_dims)
    After partitioning with a given window_size, the tensor is reshaped into:
        (B, H//window_size, W//window_size, window_size, window_size, *embed_dims)
    """
    def __init__(self, window_size: int):
        super(WindowPartition, self).__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Partition the input tensor into non-overlapping windows.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, *embed_dims).

        Returns:
            torch.Tensor: Partitioned tensor with shape 
                (B, H//window_size, W//window_size, window_size, window_size, *embed_dims).
        """
        B, H, W, *embed_dims = x.shape
        ws = self.window_size
        if H % ws != 0 or W % ws != 0:
            raise ValueError(f"H and W must be divisible by window_size {ws}. Got H={H}, W={W}.")
        # Reshape to split H and W into windows.
        x = x.view(B, H // ws, ws, W // ws, ws, *embed_dims)
        # Permute to group the window blocks together.
        windows = x.permute(0, 1, 3, 2, 4, *range(5, x.dim()))
        return windows

    def reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Reverse the window partition to reconstruct the original tensor.

        Args:
            windows (torch.Tensor): Partitioned windows with shape 
                (B, H//window_size, W//window_size, window_size, window_size, *embed_dims).
            H (int): Original height.
            W (int): Original width.

        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, H, W, *embed_dims).
        """
        ws = self.window_size
        B, num_h, num_w, ws1, ws2, *embed_dims = windows.shape
        # Permute back to interleave the window dimensions.
        x = windows.permute(0, 1, 3, 2, 4, *range(5, windows.dim())).contiguous()
        # Reshape to reconstruct the original feature map.
        x = x.view(B, num_h * ws1, num_w * ws2, *embed_dims)
        return x
