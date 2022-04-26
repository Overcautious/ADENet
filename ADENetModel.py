import torch
import torch.nn as nn

from model.audioEncoder      import audioEncoder
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from model.conv_tas_net      import ConvTasNet
from conformer.encoder import CMC_ConformerBlock

class ADENetModel(nn.Module):
    def __init__(self, isDown, isUp):
        super(ADENetModel, self).__init__()

        self.isDown = isDown
        self.isUp = isUp

        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

        # Audio Temporal Encoder 
        self.SpeechTemporalEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])

        # # Cross modal conformer
        
        self.CMC = CMC_ConformerBlock(encoder_dim=128,
                num_attention_heads=8,
                feed_forward_expansion_factor=4,
                conv_expansion_factor=2,
                feed_forward_dropout_p=0.1,
                attention_dropout_p=0.1,
                conv_dropout_p=0.1,
                conv_kernel_size=31,
                half_step_residual=True,
                layer_type = 'MLN',)
        

        self.CB = CMC_ConformerBlock(encoder_dim=256,
                num_attention_heads=8,
                feed_forward_expansion_factor=4,
                conv_expansion_factor=2,
                feed_forward_dropout_p=0.1,
                attention_dropout_p=0.1,
                conv_dropout_p=0.1,
                conv_kernel_size=31,
                half_step_residual=True,
                layer_type = 'ln',)

        self.AsdDecoder = CMC_ConformerBlock(encoder_dim=256,
                num_attention_heads=8,
                feed_forward_expansion_factor=4,
                conv_expansion_factor=2,
                feed_forward_dropout_p=0.1,
                attention_dropout_p=0.1,
                conv_dropout_p=0.1,
                conv_kernel_size=31,
                half_step_residual=True,
                layer_type = 'ln')

        #speech enhancement
        self.ConformerConv_tasnet =  ConvTasNet(norm="cLN", causal=False)
        self.down = torch.nn.MaxPool1d(32)

    def forward_visual_frontend(self, x):
        
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)        
        x = self.SpeechTemporalEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c, x2_c = self.CMC(x1, x2)   
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2): 
        x = torch.cat((x1,x2), 2)            

        x = self.CB(x)
        return x    

    def forward_end_backend(self, x, m): 
        if m is not None:
            m = self.down(m).transpose(1,2)   
            
            x = self.AsdDecoder(x*m)
        else:
            x = self.AsdDecoder(x)

        return x    


    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_speech(self, audio_input, asd_pre=None):
        if asd_pre is not None:
            x, m = self.ConformerConv_tasnet(audio_input, asd_pre)
        else:
            x, m = self.ConformerConv_tasnet(audio_input)
            
        return x, m
    
    def forward(self ,audio_input, visual_input, audio):

        audioEmbed = self.forward_audio_frontend(audio_input)     # shape : batch-size * t * 128
        visualEmbed = self.forward_visual_frontend(visual_input)  # shape: (bs, t, 128)

        audioEmbed, visualEmbed = self.forward_cross_attention(audioEmbed, visualEmbed)

        outsAV= self.forward_audio_visual_backend(audioEmbed, visualEmbed)

        outsA = self.forward_audio_backend(audioEmbed) # reshape
        outsV = self.forward_visual_backend(visualEmbed) # reshape
        del audioEmbed, visualEmbed
        # for multi task
        if self.isDown:
            outs_speech, m = self.forward_speech(audio, outsAV)
        else:
            outs_speech, m = self.forward_speech(audio, asd_pre=None)

        if self.isUp:
            outsAV = self.forward_end_backend(outsAV, m[0])
        else:
            outsAV = self.forward_end_backend(outsAV, m=None)

        outsAV = torch.reshape(outsAV, (-1, 256))
        return outsAV, outs_speech, outsA, outsV 

