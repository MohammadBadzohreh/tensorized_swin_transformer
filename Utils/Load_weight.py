import torch
def load_weight(model, weight_path):
    return model.load_state_dict(torch.load(weight_path))