# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp
import os
# import mmcv
import torch
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from torch.nn import functional as F

# from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
#                           merge_aug_bboxes_3d, show_result)
# from mmdet.core import multi_apply
from .. import builder
from ..builder import DETECTORS
# from .base import Base3DDetector
from .mvx_two_stage import MVXTwoStageDetector
# from mmdet3d.utils.vis_2dgt import vis_img_and_labels
from torch.cuda.amp.autocast_mode import autocast#一种自动混合精度（Automatic Mixed Precision, AMP）计算的工具
import cv2
import numpy as np

@DETECTORS.register_module()
class Det2D(MVXTwoStageDetector):
    """"""
    def __init__(self,
                 det2d_cfg,
                 grid_config=None,
                 depth_net=None,
                 loss_depth_weight=1.0,
                 downsample=16,
                 **kwargs):
        super(Det2D,self).__init__(**kwargs)

        self.det2d=self.depth=False    
        if det2d_cfg is not None:
            self.det2d=True
            self.det2t_head = builder.build_head(det2d_cfg)
        if depth_net is not None:
            self.depthnet=builder.build_head(depth_net)
            self.D=torch.arange(*grid_config['depth'], dtype=torch.float).shape[0]
            self.depth=True
        self.downsample=downsample#深度图降采样为特征图 与特征图大小有关
        self.grid_config=grid_config
        self.loss_depth_weight=loss_depth_weight
        self.num_frame=1
        self.sid=False
        if torch.cuda.current_device()==0 and self.depth:
            self.depthvis_root=self.det2t_head.save_vis2d_root+'depth/'
            os.mkdir(self.depthvis_root)
            self.maxd=self.grid_config['depth'][1]#最大深度距离(m)
            self.mind=self.grid_config['depth'][0]#最小深度距离(m)
        self.visdepth_idx=0

    def forward_train(self,img_inputs,img_metas=None,**kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, _ = self.prepare_inputs(img_inputs)
        img, sensor2keyego, ego2global, intrin, post_rot, post_tran=imgs[0],sensor2keyegos[0], ego2globals[0], intrins[0], post_rots[0], post_trans[0]
        mlp_input=self.get_mlp_input(sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
        # input_curr=(x, sensor2keyego, ego2global, intrin, post_rot,
        #                        post_tran, bda,mlp_input)
        B, N, C, imH, imW = img.shape
        img = img.view(B * N, C, imH, imW)
        # if self.grid_mask is not None:#None
        #     imgs = self.grid_mask(imgs)
        x = self.img_backbone(img)

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                xshape=x[0].shape
                x=[f.view(B, N, *xshape[1:]) for f in x]
        # _, output_dim, ouput_H, output_W = x[0].shape
        # x = x.view(B, N, output_dim, ouput_H, output_W)
        feats={'img_feats':x}
        outs_2d=self.det2t_head(**feats)
        # x=x[0].flatten(0,1)
        depth=self.depthnet(x[0].flatten(0,1),mlp_input)#?应为B * N, C512, H, W
        depth=depth.softmax(dim=1)#C应为depthnet depth_channels
        gt_bboxes=kwargs['bboxes2d_xyxy']
        gt_labels=kwargs['labels2d']
        centers2d=kwargs['centers2d']
        gt_depth = kwargs['gt_depth']
        loss2d_inputs = [gt_bboxes, gt_labels,
                                centers2d, outs_2d,img_metas]
        # vis_img_and_labels(img_inputs[0][0,::3].cpu().numpy().astype(np.uint8),gt_bboxes[0],gt_labels[0])
        losses2d = self.det2t_head.loss(*loss2d_inputs)
        if self.depth:
            depth_labels = self.get_downsampled_gt_depth(gt_depth)
            fg_mask = torch.max(depth_labels, dim=1).values > 0.0
            loss_depth=self.get_depth_loss(depth_labels, depth,fg_mask)#[6,118,16,44]
            losses2d.update(dict(loss_depth=loss_depth))
            if depth.device==torch.device("cuda:0") and self.visdepth_idx%200==10:
                pr_depth=torch.argmax(depth,dim=1)
                pr_depth=pr_depth.view(B,N,*pr_depth.shape[1:])[0]
                depth_labels=depth_labels.view(B,N,*pr_depth.shape[1:],self.D)
                depth_labels=torch.argmax(depth_labels,dim=-1).cpu().numpy()[0]
                self.visdepth(gt_depth[0].cpu().numpy(),pr_depth.cpu().numpy(),depth_labels,fg_mask.view(B,N,*pr_depth.shape[1:]).cpu().numpy()[0])
            self.visdepth_idx+=1
        return losses2d
    
    def get_downsampled_gt_depth(self, gt_depths):
        """ from mmdet3d/models/necks/view_transformer.py
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)#[BN,h,dsh,w,dsw,1]
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()#[BN,h,w,1,dsh,dsw]
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2] #[6,16,44]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))#torch.where(condition, x, y) True选取x
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]#[N,h,w,D]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds,fg_mask=None):
        """
        depth_labels: [B,N,256,704]
        depth_preds: [N,118/self.D,16,44]
        """
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,#[B*N*h*w, d]
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward_test(self,img_inputs,img_metas=None,**kwargs):
        pass

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)#len=nf
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:#F
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor
    
    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 4, 4).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)#[B,6,27]
        return mlp_input
    
    def visdepth(self,depthgt,depthpr,downsp_gt,mask=None):
        """gt max=60m pr max=118格"""
        N,H,W=depthgt.shape
        depthpr[~mask]=self.D#未监督置于白色大深度值
        downsp_gt[~mask]=self.D
        ori_mask=depthgt > self.mind
        depthgt[~ori_mask]=self.maxd
        for i in range(N):
            primg=depthpr[i]/self.D*255
            primg=cv2.resize(primg,(W,H))
            gtimg=depthgt[i]/self.maxd*255
            dsp_gtimg=downsp_gt[i]/self.D*255
            dsp_gtimg=cv2.resize(dsp_gtimg,(W,H))
            depthimg=np.concatenate((gtimg,primg,dsp_gtimg),axis=-1).astype(np.uint8)
            if not cv2.imwrite(self.depthvis_root+f'depth_{self.visdepth_idx}_{i}.png',depthimg):
                print('vis depth falied:',self.depthvis_root+f'depth_{self.visdepth_idx}_{i}.png')