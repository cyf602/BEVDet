# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp

import mmcv
import torch
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet.core import multi_apply
from .. import builder
from ..builder import DETECTORS
from .base import Base3DDetector
from .mvx_two_stage import MVXTwoStageDetector
from mmdet3d.utils.vis_2dgt import vis_img_and_labels


@DETECTORS.register_module()
class Det2D(MVXTwoStageDetector):
    """"""
    def __init__(self,det2d_cfg,**kwargs):
        super(Det2D,self).__init__(**kwargs)

        self.det2d=False    
        if det2d_cfg is not None:
            self.det2d=True
            self.det2t_head = builder.build_head(det2d_cfg)


    def forward_train(self,img_inputs,img_metas=None,**kwargs):
        imgs = img_inputs[0]
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        # if self.grid_mask is not None:#None
        #     imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x=x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        feats={'img_feats':x}
        outs_2d=self.det2t_head(**feats)

        gt_bboxes=kwargs['bboxes2d_xyxy']
        gt_labels=kwargs['labels2d']
        centers2d=kwargs['centers2d']
        loss2d_inputs = [gt_bboxes, gt_labels,
                                centers2d, outs_2d,img_metas]
        # vis_img_and_labels(img_inputs[0][0,::3].cpu().numpy().astype(np.uint8),gt_bboxes[0],gt_labels[0])
        losses2d = self.det2t_head.loss(*loss2d_inputs)
        return losses2d
    
    def forward_test(self,img_inputs,img_metas=None,**kwargs):
        pass