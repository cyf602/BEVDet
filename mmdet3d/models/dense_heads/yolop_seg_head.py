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
class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX
    
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
@HEADS.register_module()
class YOLOP_SEG(nn.Module):
    def __init__(self,
                nc,
                in_channel=512,
                loss_seg=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1.0)):
        super().__init__()
        self.c=in_channel
        self.up=nn.Upsample(scale_factor=2,mode='bilinear')#nearest
        self.conv1=Conv(self.c,self.c//2,3,1)
        self.csp1=Bottleneck(self.c//2,self.c//4,g=1,shortcut=False)
        self.conv2=Conv(self.c//2,self.c//4,3,1)
        self.csp2=Bottleneck(self.c//4,self.c//8,g=1,shortcut=False)
        self.classifier=Conv(self.c//8,nc,3,1)
        self.vis_idx=0
        from datetime import datetime
        if torch.cuda.current_device()==0:
            self.save_vis2d_root="vis/vis2d/"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'/'
            os.mkdir(self.save_vis2d_root)

    def forward(self,x):
        x=self.csp1(self.up(self.conv1(x)))
        x=self.csp2(self.up(self.conv2(x)))
        return self.classifier(self.up(x))

    def loss(self,seg_preds,seg_gt,canvas):#canvas为原图 len6 list
        seg_gt=torch.stack([seg for sample in seg_gt  for seg in sample],dim=0).long()
        loss_dict = dict()
        seg_loss = self.loss_seg(seg_preds.softmax(1), seg_gt)#softmax?
        loss_dict['loss_seg'] = seg_loss
        if seg_preds.device==torch.device('cuda:0') and self.vis_idx%400==2:
            seg_pr_1=seg_preds[:6].argmax(1).detach().cpu().numpy()##等待修改
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