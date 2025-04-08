import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from Window_partition import WindowPartition
from Tensorized_Layers.TCL import TCL_extended as TCL_CHANGED


def generate_2d_attention_mask(H=8, W=8, window_size=4, shift_size=2, device="cuda"):
    """
    Generates a 2D attention mask, ending with shape:
      [1, num_win_h, num_win_w, 1, 1, 1, window_size*window_size, window_size*window_size]
    """

    # --------------------------------------------------------------------------
    # 1) Create a label mask [1, H, W, 1] and fill it using "shifted" slices
    # --------------------------------------------------------------------------
    img_mask = torch.zeros((1, H, W, 1), device=device)

    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size,  None)
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size,  None)
    )

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # --------------------------------------------------------------------------
    # 2) Window partition
    # --------------------------------------------------------------------------
    def window_partition_mask(x, window_size):
        """
        Partitions x into non-overlapping windows of size (window_size x window_size).
        Returns:
            windows: shape [num_windows * B, window_size, window_size, C]
        """
        B, H_, W_, C = x.shape
        # Reshape into windows
        x = x.view(
            B,
            H_ // window_size, window_size,
            W_ // window_size, window_size,
            C
        )
        # Permute to group each window in its own batch dimension
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x.view(-1, window_size, window_size, C)
        return windows

    # Partition into windows
    mask_windows = window_partition_mask(img_mask, window_size)
    # mask_windows => [num_windows, window_size, window_size, 1]

    # Flatten each window
    mask_windows = mask_windows.view(-1, window_size * window_size)
    # => [num_windows, window_size*window_size]

    # --------------------------------------------------------------------------
    # 3) Build the attention mask by comparing window-patch labels
    # --------------------------------------------------------------------------
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    # --------------------------------------------------------------------------
    # 4) Reshape to the final shape [1, num_win_h, num_win_w, 1, 1, 1, 49, 49]
    #    for the 56x56, window_size=7 example.
    # --------------------------------------------------------------------------
    num_win_h = H // window_size
    num_win_w = W // window_size

    # 4a) First add a batch dimension
    # => [1, num_windows, window_size*window_size, window_size*window_size]
    attn_mask = attn_mask.unsqueeze(0)

    # 4b) Reshape num_windows into (num_win_h, num_win_w)
    attn_mask = attn_mask.reshape(
        1,
        num_win_h,
        num_win_w,
        window_size * window_size,
        window_size * window_size
    )
    # => [1, num_win_h, num_win_w, 49, 49] for window_size=7

    # 4c) Finally, insert three singleton dimensions in the middle
    #     => [1, num_win_h, num_win_w, 1, 1, 1, 49, 49]
    attn_mask = attn_mask.unsqueeze(3).unsqueeze(4).unsqueeze(5)

    return attn_mask


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
        x_shifted = torch.roll(
            x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
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

        x_final = torch.roll(x_reconstructed, shifts=(
            self.shift_size, self.shift_size), dims=(1, 2))
        return x_final


class ShiftedWindowMSA(nn.Module):
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

    def __init__(self, window_size: int, embed_dims: tuple, rank_window: tuple, head_factors: tuple, device='cuda'):
        super(ShiftedWindowMSA, self).__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
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
        self.input_size_window = (32, window_size, window_size) + embed_dims

        # Instantiate TCL layers for Q, K, and V.
        self.tcl_q = TCL_CHANGED(input_size=self.input_size_window,
                                 rank=rank_window, ignore_modes=(0, 1, 2), bias=True, device=self.device , r=2)
        self.tcl_k = TCL_CHANGED(input_size=self.input_size_window,
                                 rank=rank_window, ignore_modes=(0, 1, 2), bias=True, device=self.device, r=2)
        self.tcl_v = TCL_CHANGED(input_size=self.input_size_window,
                                 rank=rank_window, ignore_modes=(0, 1, 2), bias=True, device=self.device, r=2)


        # # Create a learnable relative bias table.
        # self.rel_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size - 1) *
        #                 (2 * window_size - 1), self.num_heads, device=self.device)
        # )
        # nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # # Pre-compute relative position indices for a window.
        # coords_h = torch.arange(window_size, device=self.device)
        # coords_w = torch.arange(window_size, device=self.device)
        # coords = torch.stack(torch.meshgrid(
        #     coords_h, coords_w, indexing='ij'))  # [2, ws, ws]
        # coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        # relative_coords = coords_flatten[:, :, None] - \
        #     coords_flatten[:, None, :]  # [2, ws*ws, ws*ws]
        # relative_coords = relative_coords.permute(
        #     1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        # relative_coords[:, :, 0] += window_size - 1
        # relative_coords[:, :, 1] += window_size - 1
        # relative_coords[:, :, 0] *= 2 * window_size - 1
        # # [ws*ws, ws*ws]
        # self.relative_position_index = relative_coords.sum(-1)

        # print("shift_size is",self.shift_size)

        self.window_partition = ShiftedWindowPartition(
            window_size=self.window_size, shift_size=self.shift_size)

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

        # # 7. Compute and add relative positional bias.
        # bias = self.rel_bias_table[self.relative_position_index.view(-1)]
        # # (num_tokens, num_tokens, num_heads)
        # bias = bias.view(num_tokens, num_tokens, self.num_heads)
        # # (num_heads, num_tokens, num_tokens)
        # bias = bias.permute(2, 0, 1).contiguous()
        # # Reshape bias to separate head dimensions: (h1, h2, h3, num_tokens, num_tokens)
        # bias = bias.view(h1, h2, h3, num_tokens, num_tokens)
        # # (1, 1, 1, h1, h2, h3, num_tokens, num_tokens)
        # bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Flatten query spatial dims (i,j) and key spatial dims (k,l) in attn.
        attn_flat = attn.view(B, nH, nW, h1, h2, h3, num_tokens, num_tokens)

        # print("shape of attention flat is", attn_flat.shape)

        # print("shape of bias is", bias.shape)

        # attn_flat = attn_flat + bias
        attn_flat = attn_flat 

        # print("attention flat after add bias" , attn_flat.shape)

        mask = generate_2d_attention_mask(
            H=H, W=W, window_size=ws, shift_size=self.shift_size, device=x.device)

        # print("shape of mask is", mask.shape)
        attn_flat = attn_flat + mask



        sign_attn = torch.sign(attn_flat)
        abs_attn = torch.abs(attn_flat)


        # print("attn_flat shape afeter add to the mask",attn_flat.shape)

        # print("attention flat shape",attn_flat.shape)

        # 8. Apply softmax over the key token dimension.
        attn_softmax = torch.softmax(abs_attn, dim=-1)


        attn_softmax = sign_attn * attn_softmax

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
