import os
import numpy as np
import mmcv
from tqdm import tqdm

pkl_file="data/nuscenes/bevdetv3-nuscenes_infos_train.pkl"
data=mmcv.load(pkl_file)
data_infos = data['infos']
for data_info in tqdm(data_infos):
    occ_gt = dict(np.load(data_info['occv2_path']+'/labels.npz', allow_pickle=True))
    flow3d=occ_gt['flow']
    flow3d_sq=np.linalg.norm(flow3d, axis=-1)
    W,H,Z,D=flow3d.shape

    max_indices=np.argmax(flow3d_sq,axis=-1)

    # 使用 np.arange 生成行索引
    # 这里的 row_indices 是生成 M 的行索引
    row_indices = np.arange(W).reshape(-1, 1)  # 形状为 (W, 1)

    flow2d=flow3d[row_indices,np.arange(H),max_indices]
    occ_gt['flow2d']=flow2d
    np.savez_compressed(data_info['occv2_path']+'/labels.npz',**occ_gt)
