_base_ = ['./det2d_r50.py']
numC=256
model=dict(
    downsample=8,
    loss_depth_weight=0.5,
    img_backbone=dict(
        # pretrained='torchvision://resnet50',
        pretrained='ckpts/resnet101-5d3b4d8f.pth',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1,2, 3),#需同时更改fpn inchannel;lss downsample
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512,1024, 2048],
        out_channels=numC,
        num_outs=1,#4 for yolo det CustomFPN当前只输出单一特征图
        start_level=0,
        out_ids=[0]),
    depth_net=dict(
        in_channels=numC,
        mid_channels=numC
    ),
    seg2d_cfg=dict(
        in_channel=numC,
        n_deconvs=3,
    ),
)