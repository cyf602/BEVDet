# Copyright (c) Phigent Robotics. All rights reserved.

# align_after_view_transfromation=False
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 8.22
# ===> barrier - IoU = 44.21
# ===> bicycle - IoU = 10.34
# ===> bus - IoU = 42.08
# ===> car - IoU = 49.63
# ===> construction_vehicle - IoU = 23.37
# ===> motorcycle - IoU = 17.41
# ===> pedestrian - IoU = 21.49
# ===> traffic_cone - IoU = 19.7
# ===> trailer - IoU = 31.33
# ===> truck - IoU = 37.09
# ===> driveable_surface - IoU = 80.13
# ===> other_flat - IoU = 37.37
# ===> sidewalk - IoU = 50.41
# ===> terrain - IoU = 54.29
# ===> manmade - IoU = 45.56
# ===> vegetation - IoU = 39.59
# ===> mIoU of 6019 samples: 36.01


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# For OpenOcc v2 we have 17 classes (including `free`)
occ_class_names = [
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
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],#0.25->10g/bs
}
occupancy_size=[200,200,16]
#for former encoder
pc_range=[grid_config['x'][0],grid_config['y'][0],grid_config['z'][0],grid_config['x'][1],grid_config['y'][1],grid_config['z'][1]]
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
# voxel_size = [0.1, 0.1, 0.2]

# numC_Trans = 32

multi_adj_frame_id_cfg = (1, 2+1, 1)
num_frame=len(range(*multi_adj_frame_id_cfg))
_dim_ = 16
expansion = 8
base_channel = 64
return_len_=15
n_e_=512#embedding的维度
model = dict(
    type='Flowworld',
    numf=len(range(*multi_adj_frame_id_cfg)),
    expansion=expansion,
    encoder_cfg=dict(
        type='Encoder2D',
        ch = base_channel, 
        out_ch = base_channel, 
        ch_mult = (1,2,4), 
        num_res_blocks = 2,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        z_channels = base_channel * 2, 
        double_z = False,
    ), 
    pose_encoder=dict(
        type = 'PoseEncoder',
        in_channels=4,
        out_channels=base_channel*2,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
    # transformer=dict(
    #     type='Spatial_Temp_Transformer',
    #     num_block=3,
    #     num_frame=num_frame,
    # ),
    vae = dict(
        type = 'VAERes2D',
        encoder_cfg=dict(
            type='Encoder2D',
            ch = base_channel, 
            out_ch = base_channel, 
            ch_mult = (1,2,4), 
            num_res_blocks = 2,
            attn_resolutions = (50,), 
            dropout = 0.0, 
            resamp_with_conv = True, 
            in_channels = _dim_ * expansion,
            resolution = 200, 
            z_channels = base_channel * 2, 
            double_z = False,
        ), 
        decoder_cfg=dict(
            type='Decoder2D',
            ch = base_channel, 
            out_ch = _dim_ * expansion, 
            ch_mult = (1,2,4), 
            num_res_blocks = 2,
            attn_resolutions = (50,), 
            dropout = 0.0, 
            resamp_with_conv = True, 
            in_channels = _dim_ * expansion,
            resolution = 200, 
            z_channels = base_channel * 2, 
            give_pre_end = False
        ),
        num_classes=17,
        expansion=expansion, 
        vqvae_cfg=dict(
            type='VectorQuantizer',
            sane_index_shape=True,
            n_e = n_e_, 
            e_dim = base_channel * 2, 
            beta = 1., 
            z_channels = base_channel * 2, 
            use_voxel=False)),
    transformer=dict(#transformer of occworld
        type = 'PlanUAutoRegTransformer',
        num_tokens=1,
        num_frames=return_len_,
        num_layers=2,
        img_shape=(base_channel*2,50,50),
        pose_shape=(1,base_channel*2),
        pose_attn_layers=2,
        pose_output_channel=base_channel*2,
        tpe_dim=base_channel*2,
        channels=(base_channel*2, base_channel*4, base_channel*8),
        temporal_attn_layers=6,
        output_channel=n_e_,
        learnable_queries=False
    ),
)
# Data
dataset_type = 'NuScenesDatasetOccpancyv2'
# dataset_type = 'TemporalNuSceneOcc'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    # dict(
    #     type='PrepareImageInputs',
    #     is_train=True,
    #     data_config=data_config,
    #     sequential=True),
    dict(type='LoadOccGTFromFilev3'),
    dict(type='FormatPoses'),
    # dict(type='LoadTempOccGTFromFile'),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics','next_voxel_semantics',
                                'voxel_flow','vismask','dstamp','dstamp2past','ego2next_mat',
                                'past_voxel_semantics','ego2past_mat','next_vismask',
                                'dstamps','occ_inputs','Quaternions','rel_yaws','rel_locs'])
]

test_pipeline = [
    # dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='LoadOccGTFromFilev3'),
    dict(type='FormatPoses'),
    dict(type='Collect3D', keys=['points', 'img_inputs','occ_inputs','Quaternions','rel_yaws','rel_locs'])
    
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['points', 'img_inputs','occ_inputs','Quaternions','rel_yaws','rel_locs'])
    #     ])
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
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    # train=dict(
    #     type='CBGSDataset',
    #     dataset=dict(
    #     data_root=data_root,
    #     ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
    #     pipeline=train_pipeline,
    #     classes=class_names,
    #     test_mode=False,
    #     use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        # box_type_3d='LiDAR')),
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
# data['train']['dataset'].update(share_data_config)#cbgs
data['train'].update(share_data_config)
# env_cfg = dict(
#     cudnn_benchmark=False,
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
#     dist_cfg=dict(backend='nccl',timeout=10800),
# )
# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    gamma=0.2,
    step=[24,])
checkpoint_config = dict(interval=1)
evaluation = dict(interval=3, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=30)

# custom_hooks = [
#     dict(
#         type='MEGVIIEMAHook',
#         init_updates=10560,
#         priority='NORMAL',
#     ),
# ]
# resume_from="work_dirs/bevdepth-selfsupervise1129/epoch_9.pth"

