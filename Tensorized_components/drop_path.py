def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    
    keep_prob = 1 - drop_prob
    batch_size = x.shape[0]
    # shape for random mask -> (B, 1, 1, 1, 1, 1) for your 6D input
    random_tensor = keep_prob + torch.rand(
        (batch_size, 1, 1, 1, 1, 1),
        dtype=x.dtype, device=x.device
    )
    random_tensor.floor_()
    x = x / keep_prob * random_tensor
    return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)