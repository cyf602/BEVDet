import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from torch import nn
from mmdet3d.core.bbox.transforms import bbox3d2result
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet.models.backbones.resnet import ResNet
from .bevdet import BEVDepth4D
@DETECTORS.register_module()
class BEVDepth4D_Multitask(BEVDepth4D):
    def __init__(self,seg_head,map_grid_conf,pred_det=True,pred_seg=False,**kwargs):
        super(BEVDepth4D_Multitask,self).__init__(**kwargs)
        self.feat_cropper = BevFeatureSlicer(kwargs['img_view_transformer']['grid_config'], map_grid_conf)    
        self.pred_seg=pred_seg
        self.pred_det=pred_det
        # if pred_seg:
        #     self.seg_head = builder.build_head(seg_head)
            

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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
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
    
    def forward_seg_train(self,img_feats,semantic_indices):
        seg_bev = self.feat_cropper(img_feats[0])#[B,256,?150->200,150->400]    
        outs=self.seg_head(seg_bev)
        seg_loss_inputs = [outs,semantic_indices]
        seg_losses = self.seg_head.segloss(*seg_loss_inputs)
        return seg_losses
    
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
        outs = self.pts_bbox_head(pts_feats)
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
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts,seg_preds = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox,seg_pred in zip(bbox_list, bbox_pts,seg_preds):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['seg_preds']=seg_preds
        return bbox_list
    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
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
    

