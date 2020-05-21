import torch
from torch import nn
import torch.nn.functional as F


class VirtualBatchNorm(nn.Module):
    """
    Applies Virtual Batch Normalization over a 4D input (a mini-batch
    of 2D inputs with additional channel dimension) as described in
    paper `Improved Techniques for Training GANs`:
    https://arxiv.org/abs/1606.03498

    .. math::

        y = \\frac{x - \\mathrm{E}[x_\\text{ref}]}{ \\sqrt{\\mathrm{Var}[x_\\text{ref}] + \\epsilon}} * \\gamma + \\beta

    VirtualBatchNorm requires two forward passes. First one is to
    calculate mean and variance over a reference batch and second
    is to calculate the actual output.

    Args:

        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
    """
    def __init__(self, num_features, eps=1e-5):
        super(VirtualBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.mean = None
        self.var = None
        self.weight = nn.parameter.Parameter(torch.Tensor(num_features))
        self.bias = nn.parameter.Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def normalize(self, x):
        y = ((x-self.mean) / torch.sqrt(self.var + self.eps) *
             self.weight.view(1, self.num_features, 1, 1) +
             self.bias.view(1, self.num_features, 1, 1))
        return y

    def forward(self, x):
        """"""
        if self.mean is None and self.var is None:
            self.mean = torch.mean(x, dim=0, keepdim=True)
            self.var = torch.var(x, dim=0, keepdim=True)
            out = self.normalize(x)
        else:
            out = self.normalize(x)
            self.mean = None
            self.var = None
        return out
