import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from Window_partition import WindowPartition
from Tensorized_Layers.TCL import TCL   as  TCL_CHANGED


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
            raise ValueError(
                f"H and W must be divisible by window_size {ws}. Got H={H}, W={W}.")
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
        x = windows.permute(
            0, 1, 3, 2, 4, *range(5, windows.dim())).contiguous()
        # Reshape to reconstruct the original feature map.
        x = x.view(B, num_h * ws1, num_w * ws2, *embed_dims)
        return x


class WindowMSA(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA) module.

    This module partitions the input tensor into windows, computes tensorized Q, K, V
    using TCL layers (which operate on each window), applies relative positional bias,
    computes self-attention scores, and reconstructs the full feature map.

    Args:
        window_size (int): Spatial size of the window (e.g., 7).
        embed_dims (tuple): Embedding dimensions for each patch (e.g., (4, 4, 3)).
        rank_window (tuple): Output dimensions from TCL layers for each window 
                             (e.g., (4, 4, 3)). These should be divisible by the
                             corresponding head factors.
        head_factors (tuple): Factors to split the TCL output channels into heads.
                              For example, (2, 2, 1) will yield 2*2*1 = 4 heads.
        device (str): Device identifier (default 'cpu').
    """

    def __init__(self, window_size: int, embed_dims: tuple, rank_window: tuple, head_factors: tuple, device='cpu'):
        super(WindowMSA, self).__init__()
        self.window_size = window_size
        self.embed_dims = embed_dims      # e.g., (4, 4, 3)
        self.rank_window = rank_window    # e.g., (4, 4, 3)
        self.head_factors = head_factors  # e.g., (2, 2, 1)

        self.device = device
        # Number of heads is the product of the head factors.

        self.scale = ((self.embed_dims[0] // self.head_factors[0]) *
                      (self.embed_dims[1] // self.head_factors[1]) *
                      (self.embed_dims[2] // self.head_factors[2])) ** (-0.5)

        self.num_heads = 1
        for h in head_factors:
            self.num_heads *= h

        # Input size for TCL layers for each window: (window_size, window_size, *embed_dims)
        self.input_size_window = (1,window_size, window_size) + embed_dims

        # Instantiate TCL layers for Q, K, and V.
        self.tcl_q = TCL_CHANGED(input_size=self.input_size_window,
                                 rank=rank_window, ignore_modes=(0, 1,2), bias=True, device=self.device)
        self.tcl_k = TCL_CHANGED(input_size=self.input_size_window,
                                 rank=rank_window, ignore_modes=(0, 1,2), bias=True, device=self.device)
        self.tcl_v = TCL_CHANGED(input_size=self.input_size_window,
                                 rank=rank_window, ignore_modes=(0, 1,2), bias=True, device=self.device)

        # Create a learnable relative bias table.
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) *
                        (2 * window_size - 1), self.num_heads, device=self.device)
        )
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # Pre-compute relative position indices for a window.
        coords_h = torch.arange(window_size, device=self.device)
        coords_w = torch.arange(window_size, device=self.device)
        coords = torch.stack(torch.meshgrid(
            coords_h, coords_w, indexing='ij'))  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # [2, ws*ws, ws*ws]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        # [ws*ws, ws*ws]
        self.relative_position_index = relative_coords.sum(-1)

        self.window_partition = WindowPartition(window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, *embed_shape = x.shape
        ws = self.window_size

        # 1. Partition input into windows.
        #    Shape: (B, H//ws, W//ws, ws, ws, *embed_dims)
        x_windows = self.window_partition(x)
        B, nH, nW, ws1, ws2, *_ = x_windows.shape  # ws1 == ws2 == window_size
        num_windows = nH * nW

        # 2. Reshape for window processing: (B*num_windows, ws, ws, *embed_dims)
        x_windows = x_windows.reshape(B * num_windows, ws1, ws2, *embed_shape)

        # 3. Compute Q, K, V using TCL layers.
        Q_windows = self.tcl_q(x_windows)
        K_windows = self.tcl_k(x_windows)
        V_windows = self.tcl_v(x_windows)

        # 4. Reshape back to batch + windows.
        Q_windows = Q_windows.view(B, nH, nW, ws1, ws2, *self.rank_window)
        K_windows = K_windows.view(B, nH, nW, ws1, ws2, *self.rank_window)
        V_windows = V_windows.view(B, nH, nW, ws1, ws2, *self.rank_window)

        # 5. Split into multi-heads.
        # Let head factors be h1, h2, h3 and TCL output rank be (r1, r2, r3).
        h1, h2, h3 = self.head_factors  # e.g., (2, 2, 1)
        r1, r2, r3 = self.rank_window     # e.g., (4, 4, 3)
        # Rearrange such that TCL channels are split into head dimensions and remaining factors.
        # New shape: (B, nH, nW, ws, ws, h1, h2, h3, x, y, z) where r1 = x*h1, etc.
        q = rearrange(Q_windows, 'b m n i j (x a) (y d) (z e) -> b m n i j a d e x y z',
                      a=h1, d=h2, e=h3)
        k = rearrange(K_windows, 'b m n i j (x a) (y d) (z e) -> b m n i j a d e x y z',
                      a=h1, d=h2, e=h3)
        v = rearrange(V_windows, 'b m n i j (x a) (y d) (z e) -> b m n i j a d e x y z',
                      a=h1, d=h2, e=h3)

        # 6. Compute attention scores using einsum.
        #    Here:
        #      - b: batch
        #      - m, n: window grid positions
        #      - i, j: query window spatial positions
        #      - k, l: key window spatial positions
        #      - a, d, e: head dimensions (from h1, h2, h3)
        #      - x, y, z: remaining TCL factors (flattened over window tokens)
        attn = torch.einsum(
            "b m n i j a d e x y z, b m n k l a d e x y z -> b m n a d e i j k l",
            q, k
        ) * self.scale

        # print(attn.shape)
        num_tokens = ws1 * ws2

        # 7. Compute and add relative positional bias.
        bias = self.rel_bias_table[self.relative_position_index.view(-1)]
        # (num_tokens, num_tokens, num_heads)
        bias = bias.view(num_tokens, num_tokens, self.num_heads)
        # (num_heads, num_tokens, num_tokens)
        bias = bias.permute(2, 0, 1).contiguous()
        # Reshape bias to separate head dimensions: (h1, h2, h3, num_tokens, num_tokens)
        bias = bias.view(h1, h2, h3, num_tokens, num_tokens)
        # (1, 1, 1, h1, h2, h3, num_tokens, num_tokens)
        bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Flatten query spatial dims (i,j) and key spatial dims (k,l) in attn.
        attn_flat = attn.view(B, nH, nW, h1, h2, h3, num_tokens, num_tokens)
        attn_flat = attn_flat + bias

        # 8. Apply softmax over the key token dimension.
        attn_softmax = torch.softmax(attn_flat, dim=-1)
        # print("attnetion softmax")
        # print(attn_softmax.shape)
        attn = attn_softmax.view(B, nH, nW, h1, h2, h3, ws1, ws2, ws1, ws2)
        # print("attn shape")

        # print(attn[0, 0, 0, 0, 0, 0, 0, 0, :, :].sum())

        # print(attn.shape)

        # 9. Multiply attention scores with V to get the weighted sum.
        final_output = torch.einsum(
            "b m n a d e i j k l, b m n i j a d e x y z -> b m n a d e k l x y z",
            attn, v
        )

        # print(final_output.shape)

        # 10. Rearrange output to merge head dimensions and remaining TCL factors.
        final_output_reshaped = rearrange(
            final_output, "b m n a d e k l x y z -> b m n k l (a x) (d y) (e z)"
        )

        # print(final_output_reshaped.shape)

        # 11. Reverse the window partition to reconstruct the full feature map.
        out = self.window_partition.reverse(final_output_reshaped, H, W)

        # print(out.shape)
        return out


# usage
#     x = torch.randn(1, 56, 56, 4, 4, 3, device=device)

# # Window parameters.
# window_size = 7
# # TCL configuration: for each window, input size is (7, 7, 4, 4, 3).
# # Here, we set rank_window to (4, 4, 3) and head_factors to (2, 2, 1) so that
# # the number of heads is 2*2*1 = 4.
# rank_window = (4, 4, 3)
# head_factors = (2, 2, 1)

# # Instantiate the W-MSA module.
# w_msa = WindowMSA(window_size=window_size, embed_dims=(4, 4, 3),
#                     rank_window=rank_window, head_factors=head_factors, device=device).to(device)

# x = w_msa(x)
