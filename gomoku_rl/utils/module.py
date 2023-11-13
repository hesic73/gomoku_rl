import torch
import torch.nn as nn


def init_params(m: nn.Module):
    for p in m.parameters():
        if len(p.data.shape) > 1:
            nn.init.xavier_uniform_(p)
