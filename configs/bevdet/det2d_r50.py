# Copyright (c) Phigent Robotics. All rights reserved.
# align_after_view_transfromation=True
# mAP: 0.3605
# mATE: 0.6189
# mASE: 0.2739
# mAOE: 0.4818
# mAVE: 0.3959
# mAAE: 0.2037
# NDS: 0.4828
# Eval time: 142.2s
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.577	0.446	0.154	0.079	0.324	0.190
# truck	0.280	0.647	0.217	0.094	0.308	0.205
# bus	0.375	0.701	0.208	0.078	0.894	0.303
# trailer	0.172	0.952	0.244	0.392	0.332	0.169
# construction_vehicle	0.090	0.754	0.424	1.018	0.107	0.341
# pedestrian	0.413	0.676	0.305	0.812	0.490	0.234
# motorcycle	0.315	0.668	0.266	0.676	0.514	0.178
# bicycle	0.273	0.534	0.281	1.066	0.198	0.010
# traffic_cone	0.564	0.415	0.334	nan	nan	nan
# barrier	0.546	0.395	0.304	0.123	nan	nan

# align_after_view_transfromation=False
# mAP: 0.3618
# mATE: 0.6168
# mASE: 0.2735
# mAOE: 0.4802
# mAVE: 0.3932
# mAAE: 0.2033
# NDS: 0.4842
# Eval time: 142.2s
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.577	0.445	0.154	0.079	0.320	0.190
# truck	0.280	0.648	0.217	0.094	0.305	0.205
# bus	0.373	0.700	0.208	0.082	0.896	0.304
# trailer	0.172	0.951	0.243	0.390	0.329	0.169
# construction_vehicle	0.091	0.750	0.422	1.008	0.109	0.339
# pedestrian	0.416	0.672	0.305	0.804	0.483	0.231
# motorcycle	0.318	0.665	0.267	0.674	0.506	0.179
# bicycle	0.274	0.533	0.281	1.070	0.199	0.010
# traffic_cone	0.567	0.411	0.334	nan	nan	nan
# barrier	0.550	0.391	0.304	0.122	nan	nan

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
occ_class_names = [#0~16
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-0., 0.),
    # 'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
batch_size=8
bev_embed_dims=256
# Model
grid_config = {
    'x': [-51.2, 51.2, 0.64],#分辨率要是8的倍数（bev fpn)
    'y': [-51.2, 51.2, 0.64],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
map_grid_conf = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
    'zbound': [-5.0,3.0,8.0],#[-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 0.5],
}
voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 0+1, 1)

model = dict(
    type='Det2D',
    grid_config=grid_config,
    loss_depth_weight=0.5,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,#输出尺寸的数量 4 for yolo det CustomFPN当前只输出单一特征图
        start_level=0,#FPN的首个输入特征图索引
        out_ids=[0]),#理论不超过输入数量
    # det2d_cfg=dict(    
    #     type='YOLOXHeadCustom',
    #     num_classes=10,
    #     in_channels=512,
    #     strides=[16],#[8, 16, 32, 64],
    #     train_cfg=dict(assigner=dict(
    #         type='SimOTAAssigner', center_radius=2.5)),
    #     test_cfg=dict(score_thr=0.01, nms=dict(
    #         type='nms', iou_threshold=0.65)),
    # ),
    seg2d_cfg=dict(
        type="FCN32s",
        n_class=len(occ_class_names),
        loss_seg=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
    ),
    depth_net=dict(
        type='DepthNet',
        in_channels=512,
        mid_channels=512,#=in_channels
        context_channels=0,#不参与预训练
        depth_channels=int((grid_config['depth'][1]-grid_config['depth'][0])/grid_config['depth'][2]),#"self.D"
        use_context=False,
        use_dcn=False, 
        aspp_mid_channels=96,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=10*grid_config['x'][2],
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=10*grid_config['x'][2],
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    # scale_lim=(1., 1.),
    # rot_lim=(-22.5, 22.5),#看起来对分割效果不好
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True,
        with_2d=False,
        seg2d_root='data/nuscenes/seg2d_mask/',
        data_aug_conf=data_config),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAugv2',
        bev_h=400,#bev 分割
        bev_w=200,
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth',#'semantic_indices',
                                'bboxes2d_xyxy','labels2d','centers2d','sem2d'],
        meta_keys=('token', 'sample_idx',
       'img_shape', 'scene_name',     # 'labels2d','bboxes2d',
        # 'bboxes2d_xyxy','centers2d','labels2d','bboxdepths2d'
        # 'pts_filename','box_mode_3d','box_type_3d'
        'canvas',#'sem2d'
        ))
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='LoadAnnotations'),
    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=[ 'img_inputs'],
                 meta_keys=('scene_name','e2g_mat','box_mode_3d','box_type_3d','sample_idx'))
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    # version="v1.0-mini",
    version="v1.0-trainval",
)

test_data_config = dict(
    pipeline=test_pipeline,
    data_root=data_root,    
    ann_file=data_root + 'bevdetv3-nuscenes-mini_infos_val.pkl',
    grid_conf=map_grid_conf,
    )

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    shuffle=True,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes-mini_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        grid_conf=map_grid_conf,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')),
    # train=dict(
    #     type='NuScenesDataset',#'CBGSDataset',    
    #     data_root=data_root,
    #     ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
    #     pipeline=train_pipeline,
    #     classes=class_names,
    #     test_mode=False,
    #     use_valid_flag=True,
    #     grid_conf=map_grid_conf,
    #     # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    #     # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    #     box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
# data['train'].update(share_data_config)
data['train']['dataset'].update(share_data_config)
# Optimizer
optimizer = dict(type='AdamW', lr=2e-2, weight_decay=1e-4)
# optimizer = dict(
#     type='AdamW',
#     lr=2e-4,
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.25),
#         }),
#     weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(interval=100, pipeline=test_pipeline)
# runner = dict(type='IterBasedRunner', max_iters=20*7724)
# evaluation = dict(interval=7724,pipeline=test_pipeline)
# checkpoint_config = dict(interval=7724)

custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=2,
    ),
]
find_unused_parameters=False