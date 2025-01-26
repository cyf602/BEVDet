from tools.utils.vis_bev import vis_bev_view,vis_mask3d,vis_fut_loss,vis_flow3d
# from .bevdet import BEVStereo4D,BEVDepth4D
import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import build_neck
from mmdet.models.losses import FocalLoss
from torch.nn import functional as F
# from mmcv.cnn.bricks.transformer import build_positional_encoding
from ..builder import MODELS
from mmcv.runner import BaseModule
from mmdet3d.models.detectors.base import Base3DDetector

# https://mmengine.readthedocs.io/zh-cn/latest/migration/model.html
@DETECTORS.register_module()
class Flowworld(Base3DDetector):
    def __init__(self,numf,encoder_cfg,expansion=8,pose_encoder=None,transformer=None,train_cfg=None,test_cfg=None):
        super().__init__()
        self.numf=numf#帧数
        self.nc=17
        self.encoder = MODELS.build(encoder_cfg)
        self.pose_encoder=MODELS.build(pose_encoder)
        self.expansion=expansion
        self.class_embeds = nn.Embedding(self.nc,expansion)
        self.transformer = MODELS.build(transformer)
        
    def forward_encoder(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c
        x = x.reshape(bs*F, H, W, D * self.expansion).permute(0, 3, 1, 2)

        z, shapes = self.encoder(x)
        return z, shapes
    
    def get_pose_feature(self,**kwargs):
        rel_yaws=kwargs.get('rel_yaws')
        rel_locs=kwargs.get('rel_locs')
        # quaternions=kwargs.get('Quaternions',None)
        in_poses=torch.cat([rel_locs,rel_yaws.unsqueeze(-1)],dim=-1)
        rel_poses=self.pose_encoder(in_poses)
        return rel_poses #B,T,128
    
    def extract_feat(self, imgs):
        return super().extract_feat(imgs)
    
    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)
    
    def simple_test(self, img, img_metas, **kwargs):
        return super().simple_test(img, img_metas, **kwargs)
    
    def forward_train(self, imgs, img_metas, **kwargs):
        return super().forward_train(imgs, img_metas, **kwargs)
    
    def forward_test(self, points, img_metas, img=None, **kwargs):
        return super().forward_test(points, img_metas, img, **kwargs)
    
    def forward(self,occ_inputs,**kwargs):#occ->flow->nextocc
        z, shapes = self.forward_encoder(occ_inputs)#in:B,T,200,200,16
        poses_encs=self.get_pose_feature(**kwargs)
        
        pass