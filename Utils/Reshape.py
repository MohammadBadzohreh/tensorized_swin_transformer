import torch
import torch.nn as nn
class reshape(nn.Module):
    def __init__(self, split, map_type=1, device='cpu'):
        super(reshape, self).__init__()
        
        self.split = split
        self.map_type = map_type
        self.device = device
        
    def split_into_chunks(self, tensor):
        batch_size, C, H, W = tensor.shape
        chunks = []
        
        if self.map_type == 1:
            # Approach 1 : Attached
            C_indices, H_indices, W_indices = [
                [sum(dim // self.split[i] for _ in range(j)) for j in range(self.split[i] + 1)]
                for i, dim in enumerate([C, H, W])
            ]
            
            for b in range(batch_size):
                for i in range(self.split[0]):
                    for j in range(self.split[1]):
                        for k in range(self.split[2]):
                            chunk = tensor[b,
                                           C_indices[i]:C_indices[i+1], 
                                           H_indices[j]:H_indices[j+1], 
                                           W_indices[k]:W_indices[k+1]].unsqueeze(0)
                            chunks.append(chunk)

        elif self.map_type == 2:
            # Approach 2 : Compressed
            for b in range(batch_size):
                for i in range(self.split[0]):
                    for j in range(self.split[1]):
                        for k in range(self.split[2]):
                            C_stride_indices = torch.arange(i, C, self.split[0]).to(self.device)
                            H_stride_indices = torch.arange(j, H, self.split[1]).to(self.device)
                            W_stride_indices = torch.arange(k, W, self.split[2]).to(self.device)

                            chunk = tensor[b].index_select(0, C_stride_indices)
                            chunk = chunk.index_select(1, H_stride_indices)
                            chunk = chunk.index_select(2, W_stride_indices).unsqueeze(0)
                            chunks.append(chunk)
        
        return chunks
    
    def stack_chunks_to_form_tensor(self, chunks):
        batch_size = len(chunks) // (self.split[0] * self.split[1] * self.split[2])
        result = torch.cat(chunks).view(
            batch_size, self.split[0], self.split[1], self.split[2], 
            *chunks[0].shape[1:])
        return result

    def forward(self, x):
        chunks = self.split_into_chunks(x)
        output = self.stack_chunks_to_form_tensor(chunks)
        return output