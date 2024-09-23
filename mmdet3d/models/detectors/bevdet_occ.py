# Copyright (c) Phigent Robotics. All rights reserved.
from tools.utils.vis_bev import vis_bev_view
from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np


@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 loss_flow=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=17,
                 use_predicter=True,
                 class_wise=False,
                 pred_occ=True,
                 pred_flow=True,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.occ_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.flow_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        if use_predicter:
            if pred_occ:
                self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim*2, num_classes),
                )
            if pred_flow:
                self.flow_predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim*2, 2),
                )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        if loss_flow is not None:
            self.loss_flow=build_loss(loss_flow)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.pred_occ=pred_occ
        self.pred_flow=pred_flow
        #vis
        self.vis_idx=0
        self.show_dir ="/root/data/chuyunfeng/BEVDet/vis/Bevdet4d_occ"
        self.tempsavedir='vis/vis_finalmask/'
        indices = np.indices((200,200, 16))#[3,x,y,z]
        self.indices=np.transpose(indices,(1,2,3,0)).reshape(-1,3)

    def loss_single(self,voxel_semantics,preds,voxel_flow,preds_flow=None,
                    mask_camera=None):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long().reshape(-1)
        if preds_flow is not None:
            preds_flow = preds_flow.view(-1, 2)
            voxel_flow = voxel_flow.reshape(-1, 2)
            non_free=(voxel_semantics!=self.num_classes-1)
            # static_class=torch.logical_and((voxel_semantics<16),voxel_semantics>=10)
            static_vox=torch.norm(voxel_flow,dim=-1)==0
            rand_tensor=torch.rand(voxel_semantics.shape,device=voxel_flow.device)
            rand_mask=rand_tensor<0.1 #10%为T的mask
            final_mask=(rand_mask*static_vox+~static_vox)*non_free#只监督这些区域
            if mask_camera is not None:
                final_mask=torch.logical_and(final_mask,mask_camera.view(-1))
            loss_['loss_flow']=self.loss_flow(preds_flow[final_mask], voxel_flow[final_mask],avg_factor=torch.sum(final_mask))
            # if preds_flow.device == torch.device('cuda:0'):
            #     self.vis_finalmask(final_mask,voxel_semantics,preds,voxel_flow,preds_flow,mask_camera.reshape(-1)*non_free)
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            # voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()#visable_mask
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            # voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        _occ_pred = self.occ_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        _flow_pred = self.flow_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        # bncdhw->bnwhdc
        if self.use_predicter:
            if self.pred_occ:
                occ_pred = self.predicter(_occ_pred)#[B,200,200,16,17]
            if self.pred_flow:
                flow_pred=self.flow_predicter(_flow_pred)    
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        flow_pred=flow_pred.half().squeeze(dim=0).cpu().numpy()
        return {'occ_results':occ_res,'flow_results':flow_pred}

    def forward_train(self,
                      points=None,
                      img_metas=None,#包含sample idx
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(#img_feats:B,32,16,200,200
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']#[B,6,256,704]
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)#depth:[6,88,16,44]
        losses['loss_depth'] = loss_depth

        _occ_pred = self.occ_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc 1,200,200,16,32
        _flow_pred = self.flow_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            if self.pred_occ:
                occ_pred = self.predicter(_occ_pred)#mlp 32->64->18
            if self.pred_flow:
                flow_pred=self.flow_predicter(_flow_pred)
        voxel_semantics = kwargs['voxel_semantics']#[B,200,200,16]
        voxel_flow=kwargs.get('voxel_flow',None)
        mask_camera = kwargs.get('maxk_camera',None)
        vismask=kwargs.get('vismask',None)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, occ_pred,voxel_flow,flow_pred,mask_camera=vismask)
        losses.update(loss_occ)
        if self.vis_idx%100==0 and self.show_dir is not None and flow_pred.device == torch.device('cuda:0'):
            B,H,W,Z,nc=occ_pred.shape
            occ_pred=occ_pred.detach().clone()
            occ_pred=occ_pred.view(-1,self.num_classes).argmax(dim=-1)
            occ_pred=occ_pred.view(-1,H,W,Z)
            vis_bev_view(occ_pred,voxel_semantics,flow_pred.detach().clone(),voxel_flow,
                            save_root=self.show_dir,idx=self.vis_idx)
            
        self.vis_idx+=1
        return losses
    
    def vis_finalmask(self,final_mask,voxel_semantics,preds,voxel_flow,preds_flow,mask_camera):
        final_mask=final_mask.detach().cpu().numpy()
        mask_camera=mask_camera.detach().cpu().numpy().reshape(-1)
        gtnormflow=torch.norm(voxel_flow,dim=1)
        prnormflow=torch.norm(preds_flow,dim=1)
        non_free=(voxel_semantics!=16).cpu().numpy()
        results=np.hstack((self.indices[non_free], voxel_semantics[non_free].detach().cpu().numpy()[:, np.newaxis]))
        np.savetxt(f'{self.tempsavedir}{self.vis_idx}_sem.txt',results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
        results=np.hstack((self.indices[final_mask], voxel_semantics[final_mask].detach().cpu().numpy()[:, np.newaxis]))
        np.savetxt(f'{self.tempsavedir}{self.vis_idx}_final_sem.txt',results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
        results=np.hstack((self.indices[final_mask], torch.ones_like(voxel_semantics)[final_mask].detach().cpu().numpy()[:, np.newaxis]))
        np.savetxt(f'{self.tempsavedir}{self.vis_idx}_final_mask.txt',results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
        results=np.hstack((self.indices[non_free], gtnormflow[non_free].detach().cpu().numpy()[:, np.newaxis]))
        np.savetxt(f'{self.tempsavedir}{self.vis_idx}_flowgt.txt',results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
        results=np.hstack((self.indices[final_mask], gtnormflow[final_mask].detach().cpu().numpy()[:, np.newaxis]))
        np.savetxt(f'{self.tempsavedir}{self.vis_idx}_final_flowgt.txt',results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
        results=np.hstack((self.indices[mask_camera], gtnormflow[mask_camera].detach().cpu().numpy()[:, np.newaxis]))
        np.savetxt(f'{self.tempsavedir}{self.vis_idx}_visable_flowgt.txt',results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
        results=np.hstack((self.indices[final_mask], prnormflow[final_mask].detach().cpu().numpy()[:, np.newaxis]))
        np.savetxt(f'{self.tempsavedir}{self.vis_idx}_final_flowpr.txt',results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
        print("save final mask for idx:",self.vis_idx)