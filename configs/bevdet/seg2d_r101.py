_base_ = ['./det2d_r50.py']
numC=256
batch_size=8
model=dict(
    downsample=8,
    loss_depth_weight=0.25,
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
    # seg2d_cfg=dict(
    #     type="FCN32s",
    #     n_class=17,
    #     loss_seg=dict(
    #         type='CrossEntropyLoss',
    #         use_sigmoid=False,
    #         loss_weight=1.0),
    # ),
    seg2d_cfg=dict(
        type="YOLOP_SEG",
        n_class=17,
        in_channel=numC,
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
    ),
)
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size)

optimizer = dict(type='AdamW', lr=2e-2, weight_decay=1e-4)
resume_from='work_dirs/seg2d_r101_yolopseghead327/epoch_2.pth'