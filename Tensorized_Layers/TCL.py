
import torch
import torch.nn as nn
from torch.nn import init

import math

# torch.autograd.set_detect_anomaly(True)

class TCL(nn.Module):
    def __init__(self, input_size, rank, ignore_modes = (0,), bias = True, device = 'cuda'):
        super(TCL, self).__init__()
        
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQERSUVWXYZ'
        self.device = device
        self.bias = bias
        
        if isinstance(input_size, int):
            self.input_size = (input_size, )
        else:
            self.input_size = tuple(input_size)
        
        if isinstance(rank, int):
            self.rank = (rank, )
        else:
            self.rank = tuple(rank)
        
        if isinstance(ignore_modes, int):
            self.ignore_modes = (rank, )
        else:
            self.ignore_modes = tuple(ignore_modes)
        
        # remove ignored modes from the input size
        new_size = []
        for i in range(len(self.input_size)):
            if i in self.ignore_modes:
                continue
            else:
                new_size.append(self.input_size[i])
        
        if self.bias:
            self.register_parameter('b', nn.Parameter(torch.empty(self.rank, device=self.device), requires_grad=True))
            self.b = nn.Parameter(torch.empty(self.rank), requires_grad=True)
        else:
            self.register_parameter('b',None)
            
        # Tucker Decomposition method for TCL
                                   
        # List of all factors
        for i,r in enumerate(self.rank):
            self.register_parameter(f'u{i}', nn.Parameter(torch.empty((r, new_size[i]), device = self.device), requires_grad=True))

        # Generate formula for output :
        index = 0
        formula = ''
        core_str = ''
        extend_str = ''
        out_str = ''
        for i in range(len(self.input_size)):
            formula+=alphabet[index]
            if i not in self.ignore_modes:
                core_str+=alphabet[index]
            else:
                extend_str+=alphabet[index]   
            index+=1
            if i==len(self.input_size)-1:
                formula+=','
        
        for l in range(len(self.rank)):
            formula+=alphabet[index]
            formula+=core_str[l]
            out_str+=alphabet[index]
            index+=1
            if l < len(self.rank) - 1:
                formula+=','
            elif l == len(self.rank) - 1:
                    formula+='->'
        formula+=extend_str+out_str  
            
        self.out_formula = formula
        # print(formula) 

        self.init_param() # initialize parameters       
        
    def forward(self, x):
        operands = [x]
        for i in range(len(self.rank)):
            operands.append(getattr(self, f'u{i}'))  

        out = torch.einsum(self.out_formula, operands)
        if self.bias:
            out += self.b
        return out # You may rearrange your out tensor to your desired shapes 
    
    def init_param(self): # initialization methods by tensorly
        for i in range(len(self.rank)):
            init.kaiming_uniform_(getattr(self, f'u{i}'), a = math.sqrt(5))
        if self.bias:
            bound = 1 / math.sqrt(self.input_size[0])
            init.uniform_(self.b, -bound, bound)


class TCL_extended(nn.Module):
    def __init__(self, input_size, rank, ignore_modes = (0,), bias = True, device = 'cuda', r = 3):
        super(TCL_extended, self).__init__()
        
        self.TCLs = nn.ModuleList([TCL(input_size, rank, ignore_modes, bias, device) for _ in range(r)])
        
    def forward(self, x):
        # outputs = [tcl(x) for tcl in self.TCLs]
        return sum([tcl(x) for tcl in self.TCLs]) 
    