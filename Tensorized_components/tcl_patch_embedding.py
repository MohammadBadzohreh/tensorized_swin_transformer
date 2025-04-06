import sys
sys.path.append('..')
import torch
import torch.nn as nn
from einops import rearrange

from Tensorized_Layers.TCL_CHANGED import TCL_CHANGED
# from Tensorized_Layers.TRL import TRL

class PatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, bias = True, device = 'cuda', ignore_modes = (0,1,2)):
        super(PatchEmbedding, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.bias = bias
        self.device = device
        self.ignore_modes = ignore_modes

        self.tcl_input_size = (self.input_size[0], self.input_size[2]//self.patch_size, self.input_size[3]//self.patch_size,
                                self.patch_size, self.patch_size, self.input_size[1]) # patched input image size
        self.tcl = TCL_CHANGED(input_size=self.tcl_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias, 
                            device=self.device)

    def forward(self, x):
        x = rearrange(x, 
                        'b c (p1 h) (p2 w) -> b p1 p2 h w c',
                        h=self.patch_size, w=self.patch_size) # X = [B P1 P2 H W C]
        
        x = self.tcl(x) # X = [B P1 P2 D1 D2 D3]



        return x # patches