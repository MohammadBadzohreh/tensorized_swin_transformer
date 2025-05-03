import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import rearrange
import torch.nn.functional as F

class TCL_CHANGED(nn.Module):
    def __init__(self, input_size, rank, ignore_modes=(0,), bias=True, device='cuda'):
        """
        input_size: tuple, shape of the input tensor (e.g., (7,7,4,4,3) for a window)
        rank: tuple or int, target rank for the non-ignored modes (e.g., (4,4,3))
        ignore_modes: tuple or int, indices of dimensions to leave unchanged (e.g., (0,1) to preserve spatial grid)
        bias: bool, whether to add a bias parameter.
        device: device string.
        """
        super(TCL_CHANGED, self).__init__()
        
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQERSUVWXYZ'
        self.device = device
        self.bias = bias
        
        if isinstance(input_size, int):
            self.input_size = (input_size,)
        else:
            self.input_size = tuple(input_size)
        
        if isinstance(rank, int):
            self.rank = (rank,)
        else:
            self.rank = tuple(rank)
        
        if isinstance(ignore_modes, int):
            self.ignore_modes = (ignore_modes,)
        else:
            self.ignore_modes = tuple(ignore_modes)
        
        # Remove ignored modes from the input size
        new_size = []
        for i in range(len(self.input_size)):
            if i in self.ignore_modes:
                continue
            else:
                new_size.append(self.input_size[i])
        
        # Register bias if enabled
        if self.bias:
            # Bias shape is the same as the rank tensor
            self.register_parameter('b', nn.Parameter(torch.empty(self.rank, device=self.device), requires_grad=True))
        else:
            self.register_parameter('b', None)
            
        # Register factor matrices (one per mode being contracted)
        for i, r in enumerate(self.rank):
            self.register_parameter(f'u{i}', nn.Parameter(torch.empty((r, new_size[i]), device=self.device), requires_grad=True))
        
        # Dynamically build the einsum formula for tensor contraction.
        index = 0
        formula = ''
        core_str = ''
        extend_str = ''
        out_str = ''
        # Build input part and track core (contracted) vs. extended (ignored) dimensions
        for i in range(len(self.input_size)):
            formula += alphabet[index]
            if i not in self.ignore_modes:
                core_str += alphabet[index]
            else:
                extend_str += alphabet[index]
            index += 1
            if i == len(self.input_size) - 1:
                formula += ','
        
        # Build factor matrices part and output mapping
        for l in range(len(self.rank)):
            formula += alphabet[index]
            formula += core_str[l]
            out_str += alphabet[index]
            index += 1
            if l < len(self.rank) - 1:
                formula += ','
            elif l == len(self.rank) - 1:
                formula += '->'
        formula += extend_str + out_str
        
        self.out_formula = formula
        # Uncomment the following line to inspect the generated einsum formula:
        # print("Generated einsum formula:", self.out_formula)

        self.init_param()  # Initialize parameters

    def forward(self, x):
        """
        If the input x has an extra batch dimension (i.e. its dimension equals len(input_size)+1),
        insert an ellipsis in both the input and output parts of the einsum equation.
        """


        
        if x.dim() == len(self.input_size) + 1:
            input_part, output_part = self.out_formula.split("->")
            new_formula = "..." + input_part + "->..." + output_part
        else:
            new_formula = self.out_formula
        
        operands = [x]
        for i in range(len(self.rank)):
            operands.append(getattr(self, f'u{i}'))
        
        out = torch.einsum(new_formula, *operands)
        if self.bias:
            out += self.b
        return out

    def init_param(self):
        # Initialize factor matrices using Kaiming Uniform initialization
        for i in range(len(self.rank)):
            init.kaiming_uniform_(getattr(self, f'u{i}'), a=math.sqrt(5))
        if self.bias:
            bound = 1 / math.sqrt(self.input_size[0])
            init.uniform_(self.b, -bound, bound)

