import numbers
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn import functional as F
from typing import Union, List, Tuple
from torch import Tensor, Size
from torch.nn import init


class _MMLNorm(Module):
    def __init__(self,
                 normalized_shape,
                 num_con,
                 eps=1e-5,
                 elementwise_affine=True):
        super(_MMLNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )  # type: ignore[assignment]

        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        num_features = self.normalized_shape[0]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape))
            self.bias = Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.ConBias = nn.Sequential(nn.Linear(num_con, num_features),
                                     nn.Tanh())
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, convInfo=None):
        # inputs (batch, time, dim): Tensor contains input sequences
        # import pdb; pdb.set_trace()
        b, t, d,  = input.size()
        out = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

        if convInfo is not None:
            tarBias = self.ConBias(convInfo).view(b, -1, d)
            out = out.view(b, t, d) + tarBias

        if self.elementwise_affine:
            bias = self.bias.repeat(b).view(b, 1, d)
            weight = self.weight.repeat(b).view(b, 1, d)
            return out * weight + bias
        else:
            return out

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MMLNorm(_MMLNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))
