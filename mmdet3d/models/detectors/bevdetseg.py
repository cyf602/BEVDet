import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from torch import nn
from mmdet3d.core.bbox.transforms import bbox3d2result
from mmdet3d.models.utils.memory_buffer import StreamTensorMemory
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet.models.backbones.resnet import ResNet
from .bevdet import BEVDepth4D
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
@DETECTORS.register_module()
class BEVDepth4D_Multitask(BEVDepth4D):
    def __init__(self,map_grid_conf,grid_conf,streaming_cfg=None,**kwargs):
        super(BEVDepth4D_Multitask,self).__init__(**kwargs)
        self.feat_cropper = BevFeatureSlicer(kwargs['img_view_transformer']['grid_config'], map_grid_conf)    
        self.pred_seg=self.pts_bbox_head.pred_seg
        self.pred_det=self.pts_bbox_head.pred_det
        self.pred_vec=self.pts_bbox_head.pred_vec
        # if pred_seg:
        #     self.seg_head = builder.build_head(seg_head)
        if streaming_cfg:
            self.streaming_bev = streaming_cfg['streaming_bev']
        else:
            self.streaming_bev = False
        if self.streaming_bev:
            self.stream_fusion_neck = builder.build_neck(streaming_cfg['fusion_cfg'])
            self.batch_size = streaming_cfg['batch_size']
            self.bev_memory = StreamTensorMemory(
                self.batch_size,
            )
            xmin, xmax = grid_conf['x'][:2]
            ymin, ymax = grid_conf['y'][:2]
            self.roi_size=(xmax-xmin,ymax-ymin)
            bevw=int((xmax-xmin)/grid_conf['x'][2])
            bevh=int((ymax-ymin)/grid_conf['y'][2])
            x = torch.linspace(xmin, xmax, bevw)
            y = torch.linspace(ymax, ymin, bevh)
            y, x = torch.meshgrid(y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)
#https://zhuanlan.zhihu.com/p/688608681 ; https://blog.csdn.net/devil_son1234/article/details/130699031
            self.register_buffer('plane', plane.double())
            
    def update_bev_feature(self, curr_bev_feats, img_metas):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)
        fused_feats_list = []

        memory = self.bev_memory.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)
            else:
                # else, warp buffered bev feature to current pose
                prev_g2e_matrix=torch.inverse(self.plane.new_tensor(pose_memory[i]['e2g_mat'], dtype=torch.float64))
                curr_e2g_matrix=img_metas[i]['e2g_mat']
                curr2prev_matrix = prev_g2e_matrix @ torch.from_numpy(curr_e2g_matrix).to(prev_g2e_matrix.device)
                prev_coord = torch.einsum('lk,ijk->ijl', curr2prev_matrix, self.plane).float()[..., :2]

                # from (-30, 30) or (-15, 15) to (-1, 1)
                prev_coord[..., 0] = prev_coord[..., 0] / (self.roi_size[0]/2)
                prev_coord[..., 1] = -prev_coord[..., 1] / (self.roi_size[1]/2)

                warped_feat = F.grid_sample(bev_memory[i].unsqueeze(0), 
                                prev_coord.unsqueeze(0), 
                                padding_mode='zeros', align_corners=False).squeeze(0)
                new_feat = self.stream_fusion_neck(warped_feat, curr_bev_feats[i])
                fused_feats_list.append(new_feat)

        fused_feats = torch.stack(fused_feats_list, dim=0)

        self.bev_memory.update(fused_feats, img_metas)
        
        return fused_feats
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
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
        img_feats, pts_feats, depth = self.extract_feat(#l=0[B,256,h,w]
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        # if img_feats[0].device==torch.device('cuda:0'):
        #     print(img_feats[0].device,img_metas[0]['scene_name'])
        if self.streaming_bev:
            self.bev_memory.train()#[B,256,bevw,bevh]
            img_feats = [self.update_bev_feature(img_feats[0], img_metas)]
        #bev_feats:[B,256,160,160]
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            kwargs['semantic_indices'],
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        # if self.pred_seg:
        #     losses_seg=self.forward_seg_train(img_feats,kwargs['semantic_indices'])
        #     losses.update(losses_seg)
        return losses
    
    # def forward_seg_train(self,img_feats,semantic_indices):
    #     seg_bev = self.feat_cropper(img_feats[0])#[B,256,?150->200,150->400]    
    #     outs=self.seg_head(seg_bev)
    #     seg_loss_inputs = [outs,semantic_indices]
    #     seg_losses = self.seg_head.segloss(*seg_loss_inputs)
    #     return seg_losses
    
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          semantic_indices,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)#[B,256,h,w]?
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs,semantic_indices]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        
        # seg_bev = self.feat_cropper(pts_feats[0])#[B,256,?150->200,150->400]    
        # outs=self.seg_head(seg_bev)
        # seg_loss_inputs = [outs,semantic_indices]
        # segloss=self.seg_head.segloss(*seg_loss_inputs)
        # losses.update(segloss)
        return losses
    
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        if self.streaming_bev:
            self.bev_memory.eval()
            img_feats = [self.update_bev_feature(img_feats[0], img_metas)]
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts,seg_preds = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        # for result_dict, pts_bbox,seg_pred in zip(bbox_list, bbox_pts,seg_preds):
        #     result_dict['pts_bbox'] = pts_bbox
        #     result_dict['seg_preds']=seg_preds
        for i,result_dict in enumerate(bbox_list):
            if bbox_pts is not None:
                result_dict['pts_bbox']=bbox_pts[i]
            if seg_preds is not None:
                result_dict['seg_preds']=seg_preds
        return bbox_list
    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        if self.pred_det:
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            # bbox_results[0]['sample_idx']=img_metas[0]['sample_idx']
            # bbox_results[0]['idx']=img_metas[0]['idx']
        else:
            bbox_results=None
        if 'seg_pred' in outs[0][0].keys():
            seg_preds=outs[0][0]['seg_pred']
        else: seg_preds=None
        return bbox_results,seg_preds

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension
        
class BevFeatureSlicer(nn.Module):
    # crop the interested area in BEV feature for semantic map segmentation
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()

        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
                grid_conf['x'], grid_conf['y'], grid_conf['z'],
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])
            # vision 1 失败
            self.map_grid = torch.stack(torch.meshgrid(
                self.norm_map_x, self.norm_map_y), dim=2).permute(1, 0, 2)
            # self.map_grid = torch.stack(torch.meshgrid(
            #     self.norm_map_x, self.norm_map_y, indexing='xy'), dim=2)

             # vision 2 test
            # self.map_grid = torch.stack(torch.meshgrid(
            #     self.norm_map_x, self.norm_map_y), dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)
    
@DETECTORS.register_module()
class BEVDepth4DFormer_Multitask(BEVDepth4D_Multitask):
    def __init__(self,bev_w,bev_h,positional_encoding=None,formerencoder=None,**kwargs):
        super().__init__(**kwargs)
        self.bev_w=bev_w
        self.bev_h=bev_h
        # self.bev_w,self.bev_h=self.img_view_transformer.grid_size[:2].int().tolist()
        self.real_h,self.real_w=self.img_view_transformer.grid_config['x'][1]-self.img_view_transformer.grid_config['x'][0],self.img_view_transformer.grid_config['y'][1]-self.img_view_transformer.grid_config['y'][0]
        self.frame_idx=0 #时序中的帧数
        self.numChannels=self.img_view_transformer.out_channels
        if formerencoder:
            self.tempformer=build_transformer(formerencoder)#PerceptionTransformer
            self.bev_embedding = nn.Embedding(self.bev_w*self.bev_h,formerencoder['embed_dims'])
                # self.bev_h * self.bev_w, self.embed_dims)#w,h,256
            # self.query_embedding = nn.Embedding(900,formerencoder['embed_dims']*2)#self.num_query,
                                                # self.embed_dims * 2)
        if positional_encoding:
            self.positional_encoding = build_positional_encoding(
                positional_encoding)
    def extract_img_feat(self, img, img_metas, pred_prev=False, sequential=False, **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
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
        bev_feat = torch.cat(bev_feat_list, dim=1)#[B,80,160,160]
        
        dtype = bev_feat.dtype
        bs=bev_feat.shape[0]
        new_bev_feat_list=[]#经 temp attn后的bevfeatures
        # object_query_embeds = self.query_embedding.weight.to(dtype)
        prev_bev=None
        for i in range(self.num_frame):#从前往后
            j=self.num_frame-i
            
            # if not img_metas[0]['prev_bev_exists']:
            #     prev_bev = None
            # prev_bev=bev_feat[1].view(bs,-1,w*h).permute(0,2,1)#[B,4e4,512]
            if i!=self.num_frame-1:#
                istraining=self.training
                self.eval()
                with torch.no_grad():
                    cur_bev=bev_feat[:,(j-1)*self.numChannels:j*self.numChannels,...]
                    bev_queries = self.bev_embedding.weight.to(dtype)
                    bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                    device=bev_queries.device).to(dtype)
                    bev_pos = self.positional_encoding(bev_mask).to(dtype)
                    cur_bev=self.img_bev_encoder_backbone(cur_bev)#多尺度        
                    cur_bev=self.img_bev_encoder_neck(cur_bev)
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
                cur_bev=self.img_bev_encoder_neck(cur_bev)
                bs,d,w,h=cur_bev.shape
                prev_bev=self.tempformer(#[B,w*h,256]
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
            new_bev_feat_list.append(prev_bev.permute(0,2,1).reshape(bs,-1,self.bev_w,self.bev_h))#[B,w*h,256]
        # bev_feat=torch.cat(new_bev_feat_list,dim=1)#[B,32*3,16,200,200]
        x=[new_bev_feat_list[-1]]
        return x, depth_list[0]
