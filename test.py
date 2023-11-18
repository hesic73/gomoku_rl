import torch

a = torch.randn(3, 4)
print(a)
b = torch.flip(a, dims=[-2])
print(b)
