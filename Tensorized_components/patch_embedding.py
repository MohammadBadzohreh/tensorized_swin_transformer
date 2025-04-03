import math
import torch.nn as nn

class Patch_Embedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_shape=(16,16,3), bias=True):
        """
        Args:
            img_size (int): The height (and width) of the input image.
            patch_size (int): The size of each patch (both height and width).
            in_chans (int): Number of input channels.
            embed_shape (tuple): Desired shape for each patch embedding. 
                                 The product of its dimensions is used as the output channel count.
            bias (bool): Whether to use bias in the convolution.
        """
        super(Patch_Embedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_shape = embed_shape
        # Calculate the flattened embedding size (e.g. 4*4*3 = 48)
        self.out_channels = math.prod(embed_shape)
        
        # Convolutional layer: partitions the image into patches and projects each patch.
        self.proj = nn.Conv2d(
            in_channels=in_chans, 
            out_channels=self.out_channels,
            kernel_size=patch_size, 
            stride=patch_size,
            bias=bias
        )

        self.norm = nn.LayerNorm(embed_shape)


    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            Tensor: Patch embeddings of shape (B, patch_H, patch_W, *embed_shape)
                    where patch_H = patch_W = img_size / patch_size.
                    For example: (16, 14, 14, 4, 4, 3)
        """
        # Apply convolution: output shape (B, out_channels, H/patch_size, W/patch_size)
        x = self.proj(x)
        B, C, H, W = x.shape

        # Reshape the channel dimension into the specified embedding shape.
        # This gives a tensor of shape (B, embed_shape[0], embed_shape[1], embed_shape[2], H, W)
        x = x.view(B, self.embed_shape[0], self.embed_shape[1], self.embed_shape[2], H, W)
        # Permute the tensor so that the spatial patch grid comes first: (B, H, W, embed_shape[0], embed_shape[1], embed_shape[2])
        x = x.permute(0, 4, 5, 1, 2, 3)

        x = self.norm(x)  # LayerNorm expects last dims to be normalized
        return x
    
    # TODO: add argument layer norm

# Example usage:
# if __name__ == "__main__":
#     # Input: batch size 16, 3 channels, 224x224 image.
#     x = torch.randn(16, 3, 224, 224)
#     # Here patch_size=16 and embed_shape=(4,4,3) result in an output of shape (16, 14, 14, 4, 4, 3)
#     patch_embed = Patch_Embedding(img_size=224, patch_size=16, in_chans=3, embed_shape=(16,16,3), bias=True)
#     out = patch_embed(x)
#     # print("Output shape:", out.shape)
#     # Expected output shape: (16, 14, 14, 4, 4, 3)


