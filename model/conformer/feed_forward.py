# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor

from conformer.activation import Swish
from conformer.modules import Linear
from conformer.cbln import CBLNorm
import functools

def get_norm_layer(layer_type='ln', num_con=2):
    if layer_type == 'ln':
        norm_layer = functools.partial(nn.LayerNorm, elementwise_affine=True)
    elif layer_type == 'cbln':
        norm_layer = functools.partial(CBLNorm, elementwise_affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)

    return norm_layer



class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
        device (torch.device): torch device (cuda or cpu)

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
            device: torch.device = 'cuda'
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.device = device
        ln = get_norm_layer(layer_type='cbln', num_con=encoder_dim)
        # self.sequential = nn.Sequential(
        #     # nn.LayerNorm(encoder_dim),
        #     ln(encoder_dim),
        #     Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
        #     Swish(),
        #     nn.Dropout(p=dropout_p),
        #     Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
        #     nn.Dropout(p=dropout_p),
        # )

        self.ln = ln(encoder_dim)
        self.linear1 = Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.swish = Swish()
        self.drop1 = nn.Dropout(p=dropout_p)
        self.linear2 = Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.drop2 = nn.Dropout(p=dropout_p)

    def forward(self, inputs, convInfo=None):
        if convInfo == None:
            x = self.ln(inputs.to(self.device))
        else:
            x = self.ln(inputs.to(self.device), convInfo.to(self.device))
        # import pdb;pdb.set_trace()
        x = self.linear1(x)
        x = self.swish(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x
        # return self.sequential(inputs.to(self.device))
