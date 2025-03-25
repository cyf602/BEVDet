import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
import os
import cv2
from mmdet3d.models.builder import DETECTORS
from mmdet3d.utils.vis_2dgt import vis_single_det_and_seg
from tools.utils.vis_bev import colors_map
@HEADS.register_module()
class FCN32s(nn.Module):

    def __init__(self,  
                n_class,  
                loss_seg=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1.0)):
        super().__init__()
        self.n_class = n_class
        # self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        # self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        self.loss_seg = build_loss(loss_seg)
        self.vis_idx=0
        from datetime import datetime
        if torch.cuda.current_device()==0:
            self.save_vis2d_root="vis/vis2d/"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'/'
            os.mkdir(self.save_vis2d_root)

    def forward(self, x):
        # output = self.pretrained_net(x)
        # x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        # score = self.bn1(self.relu(self.deconv1(x)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(x)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def loss(self,seg_preds,seg_gt,canvas):#canvas为原图 len6 list
        seg_gt=torch.stack([seg for sample in seg_gt  for seg in sample],dim=0).long()
        loss_dict = dict()
        seg_loss = self.loss_seg(seg_preds.reshape(-1,self.n_class).softmax(dim=1), seg_gt.view(-1))
        loss_dict['loss_seg'] = seg_loss
        if seg_preds.device==torch.device('cuda:0') and self.vis_idx%100!=2:
            seg_pr_1=seg_preds[:6].argmax(1).detach().cpu().numpy()
            seg_gt_1=seg_gt[:6].cpu().numpy()
            self.visseg(seg_pr_1,seg_gt_1,canvas)
        self.vis_idx+=1
        return loss_dict
    
    def visseg(self,seg_prs,seg_gts,canvas):
        seg_pr_ims=colors_map[seg_prs]
        seg_gts-=1#这里只是可视化时对应，监督时free还是0
        seg_gts[seg_gts==255]=16
        seg_gt_ims=colors_map[seg_gts]
        for i, canva in enumerate(canvas):
            segvisimg=np.concatenate([seg_gt_ims[i],canva,seg_pr_ims[i]],axis=1)
            if not cv2.imwrite(self.save_vis2d_root+f'{self.vis_idx}-{i}.png',segvisimg):
                print("vis failed!"+self.save_vis2d_root+f'{self.vis_idx}-{i}.png')