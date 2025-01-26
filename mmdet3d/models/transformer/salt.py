from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from ..builder import MODELS
from mmcv.runner import BaseModule
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, build_conv_layer

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

@MODELS.register_module()
class Spatial_Temp_Transformer(BaseModule):
    def __init__(self,
                 num_block=3,
                 num_frame=3,
                 norm_cfg=dict(type='LN'),
                 embed_dims=128,
                ):
        super().__init__()
        self.num_block=num_block
        self.num_frame=num_frame
        self.norms = nn.ModuleList()
        self.attns=nn.ModuleList()
        for i in range(num_block):
            self.attns.append(SALTBlock(embed_dims))
            self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        # self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.temp_embedding=nn.Embedding(num_frame,embed_dims)

        
    def forward(self,x):
        for i in range(self.num_block):
            x_=self.attns[i](x)
            x_+=x
            x=self.norms[i](x_)
        return x
        


class SALTBlock(nn.Module):#
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)#gn
        q = self.q(h_)#conv2d
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape#[16,256,50,50]
        q = q.reshape(b, c, h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)#conv2d,k=1

        return x+h_