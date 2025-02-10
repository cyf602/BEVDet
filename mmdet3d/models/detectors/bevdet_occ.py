# Copyright (c) Phigent Robotics. All rights reserved.
from tools.utils.vis_bev import vis_bev_view,vis_mask3d
from .bevdet import BEVStereo4D,BEVDepth4D
# from mmcv.runner import force_fp32
import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import build_neck
from mmdet.models.losses import FocalLoss
from mmdet3d.models.occ_loss_utils import CustomFocalLoss

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding

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
                 future_flow_loss=None,#下一帧语义损失
                 num_extraconv2d=0,
                 flow_bev_encoder_neck=None,#bev fpn处解耦
                 use_flow2d=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.occupancy_size=[200,200,16]
        self.out_dim = out_dim
        self.bev_k=int(self.img_view_transformer.grid_size[0]/self.occupancy_size[0])
        out_channels = out_dim if use_predicter else num_classes
        # self.conv_size_transfer=ConvModule(
        #     self.img_view_transformer.out_channels,
        #     out_channels,
        #     kernel_size=3,            
        #     stride=2,
        #     padding=1,
        #     bias=True,
        #     conv_cfg=dict(type='Conv2d'))
        self.num_extraconv2d=num_extraconv2d
        self.use_flow2d=use_flow2d
        if pred_occ:
            self.occ_conv = ConvModule(
                            self.img_view_transformer.out_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            conv_cfg=dict(type='Conv3d'))
        if pred_flow:
            if use_flow2d:
                self.flow_zconv=ConvModule(#处理Z维度
                                    self.occupancy_size[-1],
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True,
                                    conv_cfg=dict(type='Conv2d'))
                self.flow_conv = ConvModule(
                                    self.img_view_transformer.out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True,
                                    conv_cfg=dict(type='Conv2d'))
            else:
                self.flow_conv = ConvModule(
                                self.img_view_transformer.out_channels,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True,
                                conv_cfg=dict(type='Conv3d'))
        occ_conv2ds,flow_conv2ds=[],[]
        for i in range(self.num_extraconv2d):
            occ_conv2ds.append(
                ConvModule(
                    self.img_view_transformer.out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    conv_cfg=dict(type='Conv2d'))
            )
            flow_conv2ds.append(
                ConvModule(
                    self.img_view_transformer.out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    conv_cfg=dict(type='Conv2d'))
            )
        if self.num_extraconv2d>0:
            if pred_occ:        
                self.occ_conv2ds=nn.Sequential(*occ_conv2ds)
            if pred_flow:
                self.flow_conv2ds=nn.Sequential(*flow_conv2ds)
        self.use_predicter =use_predicter
        self.out_occ_dim=num_classes-1 if loss_occ['type']=='FocalLoss' else num_classes
        if use_predicter:
            if pred_occ:
                self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim*2, self.out_dim*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim*2, self.out_occ_dim),
                )
                # elif predictor_type=='conv2d':
                #     self.predicter = nn.Sequential(
                #         nn.Linear(self.out_dim, self.out_dim*2),
                #         nn.Softplus(),
                #         nn.Linear(self.out_dim*2, num_classes),
                #     )
            if pred_flow:
                self.flow_predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim*2),
                    nn.ReLU(),
                    nn.Linear(self.out_dim*2, self.out_dim*2),
                    nn.ReLU(),
                    nn.Linear(self.out_dim*2, 2),
                )
        if pred_flow and pred_occ and flow_bev_encoder_neck:
            self.flow_bev_encoder_neck = build_neck(flow_bev_encoder_neck)
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
        self.show_dir ="vis/Bevdet4d_occ"
        self.tempsavedir='vis/vis_finalmask/'
        indices = np.indices((200,200, 16))#[3,x,y,z]
        self.indices=np.transpose(indices,(1,2,3,0)).reshape(-1,3)
        grid_cfg=self.img_view_transformer.grid_config
        X,Y,Z=int((grid_cfg['x'][1]-grid_cfg['x'][0])/grid_cfg['x'][2]),int((grid_cfg['y'][1]-grid_cfg['y'][0])/grid_cfg['y'][2]),int((grid_cfg['z'][1]-grid_cfg['z'][0])/grid_cfg['z'][2])
        xs=torch.linspace(grid_cfg['x'][0]+grid_cfg['x'][2]/2,grid_cfg['x'][1]-grid_cfg['x'][2]/2,X).view(X,1,1).expand(X,Y,Z)#200
        ys=torch.linspace(grid_cfg['y'][0]+grid_cfg['y'][2]/2,grid_cfg['y'][1]-grid_cfg['y'][2]/2,Y).view(1,Y,1).expand(X,Y,Z)#200
        zs=torch.linspace(grid_cfg['z'][0]+grid_cfg['z'][2]/2,grid_cfg['z'][1]-grid_cfg['z'][2]/2,Z).view(1,1,Z).expand(X,Y,Z)#16
        self.local_voxel_coors=torch.stack((xs,ys,zs),-1).double().repeat(1,1,1,1,1)#para1:batch size=1

    def loss_single(self,voxel_semantics,preds_occ,voxel_flow,preds_flow=None,
                    mask_camera=None):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if preds_flow is not None:
            B,H,W,ZF,_=preds_flow.shape#当flow2d z=1
            preds_flow = preds_flow.view(-1, 2)
            voxel_flow = voxel_flow.reshape(-1, 2)
            # non_obj=(voxel_semantics>=10)#ground etc.
            obj_class=(voxel_semantics<8)
            #监督非空体素中动态体素和10%静止体素
            # static_class=torch.logical_and((voxel_semantics<16),voxel_semantics>=10)
            
            # static_vox=torch.norm(voxel_flow,dim=-1)==0
            # rand_tensor=torch.rand(voxel_semantics.shape,device=voxel_flow.device)
            # rand_mask=rand_tensor<0.1 #10%为T的mask
            # final_mask=(rand_mask*static_vox+~static_vox)*non_free#只监督这些区域
            # final_mask=torch.logical_or((rand_mask*static_class),obj_class)#只监督这些区域
            final_mask=obj_class
            if mask_camera is not None:
                final_mask=torch.logical_and(final_mask,mask_camera)
            if self.use_flow2d:
                final_mask=torch.sum(final_mask,dim=-1)
            final_mask=final_mask.view(-1)
            loss_['loss_flow']=self.loss_flow(preds_flow[final_mask], voxel_flow[final_mask],avg_factor=torch.sum(final_mask))
            #监督visible mask,nonfree
            # final_mask=torch.logical_and(mask_camera.view(-1),non_free)
            # loss_['loss_flow']=5*self.loss_flow(preds_flow[final_mask], voxel_flow[final_mask],avg_factor=torch.sum(final_mask))
            # if preds_flow.device == torch.device('cuda:0'):
                # self.vis_finalmask(final_mask,voxel_semantics,preds,voxel_flow,preds_flow,mask_camera.reshape(-1)*non_free)
        if preds_occ is not None:
            B,H,W,ZC=voxel_semantics.shape
            voxel_semantics=voxel_semantics.reshape(-1)
            free=(voxel_semantics==self.num_classes-1)    
            if self.use_mask:#mask_camera is not None
                # if isinstance(self.loss_occ,CustomFocalLoss):#focal from bevocc
                #     pass
                mask_camera = mask_camera.to(torch.int32)
                voxel_semantics=voxel_semantics.reshape(-1)
                preds_occ=preds_occ.reshape(-1,self.out_occ_dim)
                rand_tensor=torch.rand(voxel_semantics.shape,device=voxel_flow.device)
                rand_mask=torch.logical_or(torch.logical_and((rand_tensor<0.33),free),~free)#20%free和全部nonfree
                occmask=torch.logical_and(mask_camera.view(-1),rand_mask)
                # occmask=mask_camera.view(-1).to(bool)
                num_total_samples=occmask.sum()#visable_mask
                loss_occ=self.loss_occ(preds_occ,voxel_semantics,occmask, avg_factor=num_total_samples)
                loss_['loss_occ'] = loss_occ
            else:
                # voxel_semantics = voxel_semantics.reshape(-1)
                preds_occ = preds_occ.reshape(-1, self.num_classes)
                loss_occ = self.loss_occ(preds_occ, voxel_semantics,)
                loss_['loss_occ'] = loss_occ
        if preds_flow is not None  and self.vis_idx%1000==5 and preds_flow.device==torch.device('cuda:0'):
            preds_occ=preds_occ.detach().clone()
            preds_occ=preds_occ.argmax(dim=-1)
            preds_occ=preds_occ.view(-1,H,W,ZC)
            preds_flow=preds_flow.detach().clone().view(B,H,W,ZF,-1)
            voxel_flow=voxel_flow.view(B,H,W,ZF,-1)
            voxel_semantics=voxel_semantics.view(B,H,W,ZC)
            mask=final_mask.view(B,H,W,ZF)
            occmask=occmask.view(B,H,W,ZC)
            vis_bev_view(preds_occ,voxel_semantics,preds_flow,voxel_flow,flowmask=mask,
                            occmask=occmask,save_root=self.show_dir+'mask',idx=self.vis_idx)
            # vis_mask3d(voxel_semantics[0,...],occmask[0,...],pred_occ=preds_occ[0,...],
            #            pred_flow=preds_flow[0,...],save_idx=self.vis_idx,save_root='vis/vis3d/1030')
            
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    occ_threshold=0.25,#focal loss
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_res,flow_pred=None,None
        # bncdhw->bnwhdc
        if self.pred_occ:
            _occ_pred = self.occ_conv(img_feats[0])
            B,C,Z,H,W=_occ_pred.shape 
            if self.num_extraconv2d>0:
                _occ_pred=_occ_pred.transpose(1,2).reshape(-1,C,H,W)
                _occ_pred=self.occ_conv2ds(_occ_pred)
                _occ_pred=_occ_pred.reshape(-1,Z,C,H,W).transpose(1,2)
            _occ_pred=_occ_pred.permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc 1,200,200,16,32 
            occ_pred = self.predicter(_occ_pred)#mlp 32->64->18
            if isinstance(self.loss_occ,FocalLoss):
                occ_score=occ_pred.sigmoid()
                occ_score=torch.cat((occ_score, torch.ones_like(occ_score)[..., :1] * occ_threshold), dim=-1)
                occ_res=occ_score.argmax(dim=-1)                
            else:
                occ_score=occ_pred.softmax(-1)
                occ_res=occ_score.argmax(-1)
            occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)#squeeze: test时bs=1
        else:
            occ_res=None
        if self.pred_flow:
            B,C,Z,H,W=img_feats[-1].shape
            if self.use_flow2d:
                _flow_pred=self.flow_zconv(img_feats[-1].flatten(0,1)).reshape(B,C,H,W)
                _flow_pred=self.flow_conv(_flow_pred).reshape(B,C,-1,H,W)
            else:
                _flow_pred = self.flow_conv(img_feats[-1])#[B,32,16,200,200]
            if self.num_extraconv2d>0:
                _flow_pred=_flow_pred.transpose(1,2).reshape(-1,C,H,W)
                _flow_pred=self.flow_conv2ds(_flow_pred)
                _flow_pred=_flow_pred.reshape(-1,Z,C,H,W).transpose(1,2)
            _flow_pred=_flow_pred.permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc   
            flow_pred=self.flow_predicter(_flow_pred)  #[B(1),W,H,d,c]
            flow_pred=flow_pred.half().squeeze(dim=0).cpu().numpy()
            if self.use_flow2d:
                """预测2dflow时通过语义将flow对其3d"""
                flow_pred=flow_pred.repeat(Z,axis=2)
                occ_sta=(occ_res>8)[...,None].repeat(2,axis=-1)
                flow_pred[occ_sta]=0
        else:
            flow_pred=np.zeros((W,H,Z,2),dtype=np.float16)
        return [{'occ_results':occ_res,'flow_results':flow_pred}]

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
        #Conv3d 32,32 kernal(3,3,3)
        
        if self.pred_occ:
            _occ_pred = self.occ_conv(img_feats[0])
            B,C,Z,H,W=_occ_pred.shape 
            if self.num_extraconv2d>0:#额外添加两层conv2d，用处不大
                _occ_pred=_occ_pred.transpose(1,2).reshape(-1,C,H,W)
                _occ_pred=self.occ_conv2ds(_occ_pred)
                _occ_pred=_occ_pred.reshape(-1,Z,C,H//self.bev_k,W//self.bev_k).transpose(1,2)
            _occ_pred=_occ_pred.permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc 1,200,200,16,32 
            occ_pred = self.predicter(_occ_pred)#mlp 32->64->17
            device=occ_pred.device
        else:
            occ_pred=None
                
        if self.pred_flow:
            B,C,Z,H,W=img_feats[-1].shape
            if self.use_flow2d:
                _flow_pred=self.flow_zconv(img_feats[-1].flatten(0,1)).reshape(B,C,H,W)
                _flow_pred=self.flow_conv(_flow_pred).reshape(B,C,-1,H,W)
            else:
                _flow_pred = self.flow_conv(img_feats[-1])#[B,32,16,200,200]
            # B,C,Z,H,W=_flow_pred.shape 
            #[B*d,C(32),H,W]
            if self.num_extraconv2d>0:#额外添加两层conv2d，用处不大
                _flow_pred=_flow_pred.transpose(1,2).reshape(-1,C,H,W)
                _flow_pred=self.flow_conv2ds(_flow_pred)
                _flow_pred=_flow_pred.reshape(-1,Z,C,H//self.bev_k,W//self.bev_k).transpose(1,2)
            _flow_pred=_flow_pred.permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc   
            flow_pred=self.flow_predicter(_flow_pred) 
            device=flow_pred.device
        else:
            flow_pred=None
        voxel_semantics = kwargs['voxel_semantics']#[B,200,200,16]
        voxel_flow=kwargs.get('voxel_flow2d',None) if self.use_flow2d else kwargs['voxel_flow']
        # dstamp=kwargs.get('dstamp',None)
        # ego2next_mat=kwargs.get('ego2next_mat',None)
        
        # mask_camera = kwargs.get('maxk_camera',None)
        vismask=kwargs.get('vismask',None)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, occ_pred,voxel_flow,flow_pred,mask_camera=vismask)
        losses.update(loss_occ)
        # if self.vis_idx%100==5 and self.show_dir is not None and device == torch.device('cuda:0'):
        #     if occ_pred is not None:
        #         B,H,W,Z,nc=occ_pred.shape
        #         occ_pred=occ_pred.detach().clone()
        #         occ_pred=occ_pred.view(-1,self.num_classes).argmax(dim=-1)
        #         occ_pred=occ_pred.view(-1,H,W,Z)
        #     if flow_pred is not None:
        #         flow_pred=flow_pred.detach().clone()
        #     vis_bev_view(occ_pred,voxel_semantics,flow_pred,voxel_flow,
        #                     save_root=self.show_dir,idx=self.vis_idx)
            
        self.vis_idx+=1
        return losses
    
    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)#[B,64,16,w,h]
        if self.pred_flow and self.pred_occ and self.flow_bev_encoder_neck: #两个分支       
            bev_feat = self.img_bev_encoder_backbone(bev_feat)
            x=[self.img_bev_encoder_neck(bev_feat),
                self.flow_bev_encoder_neck(bev_feat)]
        else:
            x = [self.bev_encoder(bev_feat)]
        return x, depth_key_frame
    
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
        
@DETECTORS.register_module()
class BEVDepth4DOCC(BEVStereo4DOCC):
    def __init__(self, **kwargs):
        super(BEVDepth4DOCC,self).__init__(**kwargs)
        self.num_frame-=1#stereo加的再减掉
        
    def extract_stereo_ref_feat(self, x):
        assert False
        
    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img)#
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth
    
    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        if self.pred_flow and self.pred_occ and self.flow_bev_encoder_neck: #两个分支       
            bev_feat = self.img_bev_encoder_backbone(bev_feat)
            x=[self.img_bev_encoder_neck(bev_feat),
                self.flow_bev_encoder_neck(bev_feat)]
        else:
            x = [self.bev_encoder(bev_feat)]
        return x, depth_list[0]
    
@DETECTORS.register_module()
class BEVDepthformerOCC(BEVDepth4DOCC):
    def __init__(self,formerencoder=None,positional_encoding=None, **kwargs):
        super().__init__(**kwargs)
        self.bev_w,self.bev_h=self.img_view_transformer.grid_size[:2].int().tolist()
        self.real_h,self.real_w=self.img_view_transformer.grid_config['x'][1]-self.img_view_transformer.grid_config['x'][0],self.img_view_transformer.grid_config['y'][1]-self.img_view_transformer.grid_config['y'][0]
        self.frame_idx=0 #时序中的帧数
        self.numChannels=self.img_view_transformer.out_channels
        if formerencoder:
            self.tempformer=build_transformer(formerencoder)#PerceptionTransformer
            # self.tempformer=build_transformer_layer_sequence(formerencoder)#bevformer encoder
            self.bev_embedding = nn.Embedding(200*200,formerencoder['embed_dims'])
                # self.bev_h * self.bev_w, self.embed_dims)#w,h,256
            # self.query_embedding = nn.Embedding(900,formerencoder['embed_dims']*2)#self.num_query,
                                                # self.embed_dims * 2)
        if positional_encoding:
            self.positional_encoding = build_positional_encoding(
                positional_encoding)
           
    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:#F
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)#[B,32*numf,16,w,h]
        # bev_feat = self.img_bev_encoder_backbone(bev_feat)#多尺度bev torch.Size([1, 32, 16, 200, 200])torch.Size([1, 64, 8, 100, 100])torch.Size([1, 128, 4, 50, 50])
        
        dtype = bev_feat.dtype
        bs,d,z,w,h=bev_feat.shape
        new_bev_feat_list=[]#经 temp attn后的bevfeatures
        # object_query_embeds = self.query_embedding.weight.to(dtype)
        prev_bev=None
        for i in range(self.num_frame):#从前往后
            j=self.num_frame-i#变成从后向前
            
            # if not img_metas[0]['prev_bev_exists']:
            #     prev_bev = None
            # prev_bev=bev_feat[1].view(bs,-1,w*h).permute(0,2,1)#[B,4e4,512]
            if i!=self.num_frame-1:#
                istraining=self.training
                self.eval()
                with torch.no_grad():
                    # cur_bev=bev_feat[:,i*self.numChannels:(i+1)*self.numChannels,...]
                    cur_bev=bev_feat[:,(j-1)*self.numChannels:j*self.numChannels,...]
                    bev_queries = self.bev_embedding.weight.to(dtype)
                    bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                    device=bev_queries.device).to(dtype)
                    bev_pos = self.positional_encoding(bev_mask).to(dtype)
                    cur_bev=self.img_bev_encoder_backbone(cur_bev)#多尺度        
                    # x = [self.img_bev_encoder_neck(bev_feat)]
                    prev_bev=self.tempformer(#这里面就是get bev features
                        cur_bev,#原为图像特征
                        bev_queries,#[4e4,256]
                        # object_query_embeds,
                        self.bev_h,
                        self.bev_w,
                        grid_length=[self.real_h / self.bev_h,
                                        self.real_w / self.bev_w],
                        bev_pos=bev_pos,
                        # reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                        # cls_branches=self.cls_branches if self.as_two_stage else None,
                        img_metas=img_metas,
                        prev_bev=prev_bev,
                        t_idx=i,
                    )
                if istraining:
                    self.train()
            else:
                cur_bev=bev_feat[:,(j-1)*self.numChannels:j*self.numChannels,...]
                bev_queries = self.bev_embedding.weight.to(dtype)
                bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
                bev_pos = self.positional_encoding(bev_mask).to(dtype)
                cur_bev=self.img_bev_encoder_backbone(cur_bev)#多尺度            
                # x = [self.img_bev_encoder_neck(bev_feat)]
                prev_bev=self.tempformer(
                    cur_bev,
                    bev_queries,#[4e4,256]
                    # object_query_embeds,
                    self.bev_h,
                    self.bev_w,
                    grid_length=[self.real_h / self.bev_h,
                                    self.real_w / self.bev_w],
                    bev_pos=bev_pos,
                    # reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    # cls_branches=self.cls_branches if self.as_two_stage else None,
                    img_metas=img_metas,
                    prev_bev=prev_bev,
                    t_idx=i,
                )
            new_bev_feat_list.append(prev_bev.permute(0,2,1).reshape(bs,d//self.num_frame,z,w,h))#[B,4e4,256]
        # bev_feat=torch.cat(new_bev_feat_list,dim=1)#[B,32*3,16,200,200]
        x=[new_bev_feat_list[-1]]
        # if self.pred_flow and self.pred_occ and self.flow_bev_encoder_neck: #两个分支       
        #     # bev_feat = self.img_bev_encoder_backbone(bev_feat)
        #     x=[self.img_bev_encoder_neck(bev_feat),
        #         self.flow_bev_encoder_neck(bev_feat)]
        # else:
        #     x = [self.img_bev_encoder_neck(bev_feat)]
        return x, depth_list[0]
    
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
        #Conv3d 32,32 kernal(3,3,3)
        
        
        if self.pred_occ:
            _occ_pred = self.occ_conv(img_feats[0])
            B,C,Z,H,W=_occ_pred.shape 
            if self.num_extraconv2d>0:#额外添加两层conv2d，用处不大
                _occ_pred=_occ_pred.transpose(1,2).reshape(-1,C,H,W)
                _occ_pred=self.occ_conv2ds(_occ_pred)
                _occ_pred=_occ_pred.reshape(-1,Z,C,H//self.bev_k,W//self.bev_k).transpose(1,2)
            _occ_pred=_occ_pred.permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc 1,200,200,16,32 
            occ_pred = self.predicter(_occ_pred)#mlp 32->64->17
            device=occ_pred.device
        else:
            occ_pred=None
                
        if self.pred_flow:
            _flow_pred = self.flow_conv(img_feats[-1])
            B,C,Z,H,W=_flow_pred.shape 
            #[B*d,C(32),H,W]
            if self.num_extraconv2d>0:#额外添加两层conv2d，用处不大
                _flow_pred=_flow_pred.transpose(1,2).reshape(-1,C,H,W)
                _flow_pred=self.flow_conv2ds(_flow_pred)
                _flow_pred=_flow_pred.reshape(-1,Z,C,H//self.bev_k,W//self.bev_k).transpose(1,2)
            _flow_pred=_flow_pred.permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc   
            flow_pred=self.flow_predicter(_flow_pred) 
            device=flow_pred.device
        else:
            flow_pred=None
        voxel_semantics = kwargs['voxel_semantics']#[B,200,200,16]
        voxel_flow=kwargs.get('voxel_flow',None)
        # dstamp=kwargs.get('dstamp',None)
        # ego2next_mat=kwargs.get('ego2next_mat',None)
        
        # mask_camera = kwargs.get('maxk_camera',None)
        vismask=kwargs.get('vismask',None)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, occ_pred,voxel_flow,flow_pred,mask_camera=vismask)
        losses.update(loss_occ)            
        self.vis_idx+=1
        return losses