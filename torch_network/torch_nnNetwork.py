import torch
from torch import nn


class torch_nn(nn.Module):

    def __init__(self):
        super(torch_nn, self).__init__()

    def forward(self, input):
        output = 1
        return input + 1


TestClass = torch_nn()
x = torch.tensor((1, 0, 5, 14))
int = 0
out = TestClass(int)
print(out)
