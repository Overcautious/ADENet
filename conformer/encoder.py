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
from typing import Tuple

from conformer.feed_forward import FeedForwardModule
from conformer.attention import MultiHeadedSelfAttentionModule
from conformer.convolution import (
    ConformerConvModule,
    Conv2dSubampling,
)
from conformer.modules import (
    ResidualConnectionModule,
    ResidualConnectionModule_FFN,
    ResidualConnectionModule_CONV,
    Linear,
)
from conformer.cbln import MMLNorm
import functools

def get_norm_layer(layer_type='ln', num_con=2):
    if layer_type == 'ln':
        norm_layer = functools.partial(nn.LayerNorm, elementwise_affine=True)
    elif layer_type == 'cbln':
        norm_layer = functools.partial(MMLNorm, elementwise_affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)

    return norm_layer

class CMC_ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
        device (torch.device): torch device (cuda or cpu)

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            device: torch.device = 'cuda',
            layer_type: str = 'ln',
    ):
        super(CMC_ConformerBlock, self).__init__()
        self.device = device
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.FFN = ResidualConnectionModule_FFN(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                    device=device,
                ),
                module_factor=self.feed_forward_residual_factor,
            )
        
        self.FFN_ = ResidualConnectionModule_FFN(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                    device=device,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.MHSA = ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                    device=device,
                ),
            )
        
        self.MHSA_ = ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                    device=device,
                ),
            )
        

        self.COV = ResidualConnectionModule_CONV(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                    device=device,
                ),
            )

        self.COV_ = ResidualConnectionModule_CONV(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                    device=device,
                ),
            )

        self.FFN2 = ResidualConnectionModule_FFN(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                    device=device,
                ),
                module_factor=self.feed_forward_residual_factor,
            )
        
        self.FFN2_ = ResidualConnectionModule_FFN(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                    device=device,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        ln = get_norm_layer(layer_type='cbln', num_con=encoder_dim)

        self.layer =  ln(encoder_dim)
        self.layer_ =  ln(encoder_dim)
        # self.layer =  nn.LayerNorm(encoder_dim)
        # self.layer_ =  nn.LayerNorm(encoder_dim)

        # self.sequential = nn.Sequential(
        #     ResidualConnectionModule(
        #         module=FeedForwardModule(
        #             encoder_dim=encoder_dim,
        #             expansion_factor=feed_forward_expansion_factor,
        #             dropout_p=feed_forward_dropout_p,
        #             device=device,
        #         ),
        #         module_factor=self.feed_forward_residual_factor,
        #     ),
        #     ResidualConnectionModule(
        #         module=MultiHeadedSelfAttentionModule(
        #             d_model=encoder_dim,
        #             num_heads=num_attention_heads,
        #             dropout_p=attention_dropout_p,
        #             device=device,
        #         ),
        #     ),
        #     ResidualConnectionModule(
        #         module=ConformerConvModule(
        #             in_channels=encoder_dim,
        #             kernel_size=conv_kernel_size,
        #             expansion_factor=conv_expansion_factor,
        #             dropout_p=conv_dropout_p,
        #             device=device,
        #         ),
        #     ),
        #     ResidualConnectionModule(
        #         module=FeedForwardModule(
        #             encoder_dim=encoder_dim,
        #             expansion_factor=feed_forward_expansion_factor,
        #             dropout_p=feed_forward_dropout_p,
        #             device=device,
        #         ),
        #         module_factor=self.feed_forward_residual_factor,
        #     ),
        #     nn.LayerNorm(encoder_dim),
        # )

    def forward(self, inputs: Tensor, second_input=None) -> Tensor:
        if second_input is None:
            x = self.FFN(inputs.to(self.device))
            x = self.MHSA(x)
            x = self.COV(x)
            x = self.FFN2(x)
            x = self.layer(x)
            return x
        else:
            x = self.FFN(inputs.to(self.device))
            second_input = self.FFN_(second_input.to(self.device))
            x1 = self.MHSA(x, second_input)
            y1 = self.MHSA_(second_input, x)

            x2 = self.COV(x1, y1)
            y2 = self.COV_(y1, x1)

            x3 = self.FFN2(x2, y2)
            y3 = self.FFN2_(y2, x2)

            x4 = self.layer(x3, y3)
            y4 = self.layer_(y3, x3)

            return x4, y4



class ConformerEncoder(nn.Module):
    """
    Conformer encoder first processes the input with a convolution subsampling layer and then
    with a number of conformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
        device (torch.device): torch device (cuda or cpu)

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            device: torch.device = 'cuda',
    ):
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.layers = nn.ModuleList([CMC_ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            device=device,
        ).to(device) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        # 先压缩特征
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.layers:
            outputs = layer(outputs)  # 处理后维度不变

        return outputs, output_lengths
