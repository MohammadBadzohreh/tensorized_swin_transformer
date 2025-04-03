import torch
import torch.nn as nn
import sys
sys.path.append("..")
from Tensorized_components.Window_partition import WindowPartition

class ShiftedWindowPartition(nn.Module):
    """
    Utility module for partitioning and reversing windows with a spatial shift applied.
    
    This class performs a spatial shift before partitioning the input tensor into windows.
    After partitioning and reverse operations, the spatial shift is compensated for.
    
    Input shape: (B, H, W, *embed_dims)
    """
    def __init__(self, window_size: int, shift_size: int):
        super(ShiftedWindowPartition, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_partition = WindowPartition(window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift the input tensor and partition it into windows.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, *embed_dims).

        Returns:
            torch.Tensor: Partitioned windows after spatial shift.
        """
        # Apply negative shift to the spatial dimensions.
        x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # Partition the shifted tensor using WindowPartition.
        windows = self.window_partition(x_shifted)
        return windows

    def reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Reverse the partition and then reverse the spatial shift to reconstruct the original tensor.

        Args:
            windows (torch.Tensor): Partitioned windows with shape 
                (B, H//window_size, W//window_size, window_size, window_size, *embed_dims).
            H (int): Original height.
            W (int): Original width.

        Returns:
            torch.Tensor: Reconstructed tensor after compensating for the spatial shift.
        """
        # Reverse partition to get back the shifted tensor.
        x_reconstructed = self.window_partition.reverse(windows, H, W)
        # Roll back the tensor to reverse the spatial shift.
        x_final = torch.roll(x_reconstructed, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x_final
