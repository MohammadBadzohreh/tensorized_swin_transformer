
import torch
import torch.nn as nn
from torch.nn import init

# torch.autograd.set_detect_anomaly(True)

class TRL(nn.Module):
    def __init__(self, input_size, output, rank, ignore_modes = (0,), bias = True, device = 'cuda'):
        super(TRL, self).__init__()
        
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQERSUVWXYZ'
        self.device = device
        self.bias = bias
        
        if isinstance(input_size, int):
            self.input_size = (input_size, )
        else:
            self.input_size = tuple(input_size)
            
        if isinstance(output, int):
            self.output = (output, )
        else:
            self.output = tuple(output)
        
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
        
        self.w_size = tuple(new_size) + self.output
        if self.bias:
            self.register_parameter('b', nn.Parameter(torch.empty(self.output, device = self.device), requires_grad=True))
        else:
            self.register_parameter('b',None)
            
        # Tucker Decomposition method for TRL
        
        self.register_parameter('core', nn.Parameter(torch.empty(self.rank, device = self.device), requires_grad=True))

        # List of all factors
        for i,r in enumerate(self.rank):
            self.register_parameter(f'u{i}', nn.Parameter(torch.empty((r, self.w_size[i]), device = self.device), requires_grad=True))      

        # Generate formula for w :
        
        index = 0
        formula = ''
        core_str = ''
        w_str = ''
        for i in range(len(self.core.shape)):
            formula+=alphabet[index]
            index+=1
            if i== len(self.core.shape) - 1:
                formula+=','
        core_str = formula[:len(formula)-1]
                
        for l in range(len(self.rank)):
            # change these two lines positions if it doesnt work 
            # need further investigation
            formula+=core_str[l]
            formula+=alphabet[index]
            w_str+=alphabet[index]
            index+=1
            if l < len(self.rank) - 1:
                formula+=','
            elif l == len(self.rank) - 1:
                    formula+='->'
        
        formula+=w_str
        # print(formula)
        
        self.w_formula = formula   
        operands = [self.core]
        for i in range(len(self.rank)):
            operands.append(getattr(self, f'u{i}'))  

        self.w_operands = operands
        # self.w = torch.einsum(self.w_formula, operands)
        
        # Generate formula for Generalized Inner Product of W and X:
        index = 0
        formula = ''
        mul = ''
        out_str = ''
        extend_str =''
        for i in range(len(self.input_size)):
            formula+=alphabet[index]
            if i not in self.ignore_modes:
                mul+= alphabet[index]
            else:
                extend_str+= alphabet[index]
            index+=1
            if i== len(self.input_size) - 1:
                formula+=','
        
        formula+=mul
        for i in range(len(mul),len(self.w_size)):
            formula+=alphabet[index]
            out_str+=alphabet[index]
            index+=1
            if i== len(self.w_size) - 1:
                formula+='->'
         
        formula+=extend_str+out_str       
        self.out_formula = formula
        # print(formula)
        
        self.init_param() # initialize parameters       

    def forward(self, x):
        w = torch.einsum(self.w_formula, self.w_operands)
        out = torch.einsum(self.out_formula, (x, w))
        if self.bias:
            out += self.b 
        return out # You may rearrange your out tensor to your desired shapes
    
    def init_param(self): # initialization methods by tensorly
        for i in range(len(self.rank)):
            init.normal_(getattr(self, f'u{i}'), mean=0.0, std=0.02)
        init.normal_(self.core, mean=0.0, std=0.02)
        if self.bias:
            bound = 1
            init.uniform_(self.b, -bound, bound)