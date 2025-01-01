import pickle
from mmcv.utils import build_from_cfg
from mmcv import Config, DictAction
import mmcv
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.builder import build_dataloader
from mmdet.datasets.builder import PIPELINES as MMDET_PIPELINES
from mmdet3d.datasets.builder import PIPELINES
from debug_test import show_seg
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import math
pklpath="data/nuscenes/bevdetv3-nuscenes_infos_val.pkl"
save_dir="work_dirs/bevaugseg"
car_img_cv = cv2.imread('icon/car.png')

data=mmcv.load(pklpath)
data_infos=data['infos']
cfg = Config.fromfile('configs/bevdet/bevdet-r50-4d-depthformer-iter-detseg.py')
test_dataloader_default_args = dict(
    samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
dataset = build_dataset(cfg.data.test)
test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
data_loader = build_dataloader(dataset, **test_loader_cfg)
angle=30/180*math.pi
bda_mat=torch.tensor([
    [math.cos(angle),math.sin(-angle),0.],
    [math.sin(angle),math.cos(angle),-0.]
], dtype=torch.float)
for i,data in enumerate(data_loader):
    savepath=save_dir+f"/{i}.png"
    _,w,h=data['semantic_indices'][0].shape
    target_semantic_indices=data['semantic_indices'][0].unsqueeze(0)
    one_hot = target_semantic_indices.new_full([1,4,w,h], 0)#4为类别数
    one_hot.scatter_(1, target_semantic_indices, 1)
    semantic = one_hot.cpu().numpy().astype(np.float)
    cv2.imwrite(savepath,show_seg(semantic.squeeze(),car_img_cv))
    
    semgt=data['semantic_indices'][0].to(torch.float32)
    grid = F.affine_grid(bda_mat.unsqueeze(0), semgt.unsqueeze(0).size())#.long()grid_sampler_2d_cpu not implemented for Long
    output = F.grid_sample(semgt.unsqueeze(0), grid,mode='nearest')
    output=output.squeeze(0).long()
    savepath=save_dir+f"/{i}_bda.png"
    _,w,h=output.shape
    target_semantic_indices=output.unsqueeze(0)
    one_hot = target_semantic_indices.new_full([1,4,w,h], 0)#4为类别数
    one_hot.scatter_(1, target_semantic_indices, 1)
    semantic = one_hot.cpu().numpy().astype(np.float)
    cv2.imwrite(savepath,show_seg(semantic.squeeze(),car_img_cv))
    print(f"saved:{i}")
    if i>100:break
    pass