# wujian@2018

import torch 
import torch.nn as nn
# from visualEncoder     import visualFrontend, visualTCN, visualConv1D
import torch.nn.functional as F
# import AVID as model
from einops import repeat
from conformer.encoder import CMC_ConformerBlock


def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory torchan BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # deptorchwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x



class ConvTasNet(nn.Module):
    def __init__(self,
                 L=40,
                 N=256,
                 X=8,
                 R=4,
                 B=256,
                 H=512,
                 P=3,
                 norm="cLN",
                 num_spks=1,
                 non_linear="relu",
                 causal=False):
        super(ConvTasNet, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, padding = L // 4)
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)

        # repeat blocks
        # n x B x T => n x B x T
        # self.repeats = self._build_repeats(
        #     num_repeats = 1,
        #     num_blocks = X,
        #     in_channels=B,
        #     conv_channels=H,
        #     kernel_size=P,
        #     norm=norm,
        #     causal=causal)
        
        self.repeats = self._build_conformer(num_layers=1, encoder_dim=B)
            
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not pytorchon list
        # self.conv1x1_2 = torch.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x B x T => n x 2N x T
        self.mask = Conv1D(B, num_spks * N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d = ConvTrans1D(
            N, 1, kernel_size=L, stride=L // 2, padding = L // 4, bias=True)
        self.num_spks = num_spks

        # self.visualFrontend = self._build_video()

        # self.visualFrontend  = visualFrontend() # Visual Frontend 
        # self.visualTCN       = visualTCN(256)      # Visual Temporal Network TCN
        # self.visualConv1D    = visualConv1D(256)   # Visual Temporal Network Conv1d

        self.up = nn.Upsample(scale_factor=32, mode='nearest')

        self.connected_layer = nn.Linear(512, 256)

        self.repeats_fusion = self._build_repeats(
            num_repeats = 3,
            num_blocks = X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)

    # def _build_video(self):
    #     arch = 'av_wrapper'
    #     model_args = {
    #         'proj_dim': [512, 512, 128],
    #         'video_backbone': "R2Plus1D",
    #         "video_backbone_args": {
    #             "depth": 18
    #         },
    #         'audio_backbone': "Conv2D",
    #         "audio_backbone_args": {
    #             "depth": 10
    #         }
    #     }

    #     pretrained_net = model.__dict__[arch](**model_args)
    #     checkpoint_fn = 'AVID/AudioSet/AVID_Audioset_Cross-N1024_checkpoint.pth.tar'
    #     ckp = torch.load(checkpoint_fn, map_location='cpu')
    #     pretrained_net.load_state_dict(
    #         {k.replace('module.', ''): ckp['model'][k]
    #             for k in ckp['model']})

    #     return pretrained_net.video_model
   
    def _build_conformer(self, num_layers, encoder_dim):
        blocks = [
            CMC_ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=4,
                feed_forward_expansion_factor=4,
                conv_expansion_factor=2,
                feed_forward_dropout_p=0.1,
                attention_dropout_p=0.1,
                conv_dropout_p=0.1,
                conv_kernel_size=31,
                half_step_residual=True,
                layer_type = 'ln',
            ) for _ in range(num_layers)]
        
        return nn.Sequential(*blocks)
        

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)



    def forward(self, x , asdEmbed =None):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt

        if asdEmbed is not None:
            asdEmbed = self.up(asdEmbed.transpose(1,2))


        # import pdb; pdb.set_trace()

        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # n x 1 x S => n x N x T
        w = F.relu(self.encoder_1d(x), inplace=True)
        # n x c x T
        y = self.proj(self.ln(w))
        # n x T x c
        y = y.transpose(1, 2)
        y = self.repeats(y)
        y = y.transpose(1,2)
        if asdEmbed is not None:

            fusion = torch.cat((y, asdEmbed), dim= 1)
            fusion = self.connected_layer(fusion.transpose(1,2))
        else:
            fusion = y.transpose(1,2)

        fusion_mask = self.repeats_fusion(fusion.transpose(1,2))

        # n x 2N x T
        e = torch.chunk(self.mask(fusion_mask), self.num_spks, 1)
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(torch.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(torch.stack(e, dim=0))
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # spks x n x S
        output = [self.decoder_1d(x, squeeze=True) for x in s]
        if output[0].dim() <2:
            output[0] = output[0].unsqueeze(0)
        return output, m


def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))


def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))


def foo_conv_tas_net():
    x = torch.rand(1, 16000)
    visual_input = torch.rand(1,3, 25, 224, 224)
    cuda = torch.cuda.is_available()  
    device = torch.device('cuda' if cuda else 'cpu')
    nnet = ConvTasNet(norm="cLN", causal=False).to(device)
    # print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x.cuda())
    #x = nnet(x.cuda(), visual_input.cuda())
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    foo_conv_tas_net()
    # foo_conv1d_block()
    # foo_layernorm()
