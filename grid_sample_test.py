import mmcv
import torch
import numpy as np
import os
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import torch.nn.functional as F
from tools.utils.vis_bev import indices# 可视化用indices
# 模拟下一帧位置向当前帧采样后的结果
grid_cfg = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],#0.25->10g/bs
}
X,Y,Z=int((grid_cfg['x'][1]-grid_cfg['x'][0])/grid_cfg['x'][2]),int((grid_cfg['y'][1]-grid_cfg['y'][0])/grid_cfg['y'][2]),int((grid_cfg['z'][1]-grid_cfg['z'][0])/grid_cfg['z'][2])
grid_x, grid_y = torch.meshgrid(torch.arange(X), torch.arange(Y), indexing='xy')
loc2d=torch.stack([grid_x,grid_y,torch.ones_like(grid_x),torch.ones_like(grid_x)],dim=-1).to(torch.float).view(1,X,Y,-1,1)#-1 ->4
# grid_z,grid_y, grid_x = torch.meshgrid(torch.arange(Z), torch.arange(Y),torch.arange(X), indexing='ij')
grid_x,grid_y, grid_z = torch.meshgrid(torch.arange(X), torch.arange(Y),torch.arange(Z), indexing='ij')
loc3d=torch.stack((grid_x, grid_y, grid_z,torch.ones_like(grid_x)), dim=-1).to(torch.float).view(1,X,Y,Z,4,1)
feat2bev = torch.zeros((4,4),dtype=loc2d.dtype)#类比solofusion
feat2bev[0, 0] = grid_cfg['x'][-1]
feat2bev[1, 1] = grid_cfg['y'][-1]
feat2bev[0, 3] = grid_cfg['x'][0] #这样只适配自车在中心的情况
feat2bev[1, 3] = grid_cfg['y'][0] 
feat2bev[2, 2] = 1
if False:#useloc3d
    feat2bev[2,2]=grid_cfg['z'][-1]
    feat2bev[2,3]=grid_cfg['z'][0]
feat2bev[3, 3] = 1
def mask2onehot(mask,nc):#torch 源代码是在gridsample B*Z,-1,H,W 是2Dgrid_smaple
    one_hot=torch.zeros([*mask.shape,nc])
    one_hot.scatter_(-1,mask.long().unsqueeze(-1),1.)
    return one_hot

ann_file='data/nuscenes/bevdetv3-nuscenes_infos_val.pkl'
save_root='vis/vis3d/grid_sp'
if not os.path.exists(save_root):
    os.makedirs(save_root)
data = mmcv.load(ann_file, file_format='pkl')
data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
data_infos = data_infos#[:50]#[::2]
last_e2gmat=None
last_occ=None
last_occ_onehot=None
for i,data_info in enumerate(data_infos):
    occ_gt_path = data_info['occv2_path']
    occ_gt_path = os.path.join(occ_gt_path, "labels.npz")
    occ_labels = np.load(occ_gt_path)
    input_dict={}
    input_dict['voxel_semantics'] = occ_labels['semantics'].astype(np.int64)#[200,200,16]        
    input_dict['voxel_flow']=occ_labels['flow']
    input_dict['vismask']=occ_labels['vismask']
    e2g_mat=torch.from_numpy(transform_matrix(data_info['ego2global_translation'],Quaternion(data_info['ego2global_rotation']))).to(torch.float)
    vismask=torch.from_numpy(input_dict['vismask']).to(torch.float)
    if last_e2gmat is None:
        last_e2gmat=e2g_mat
        last_occ=torch.from_numpy(input_dict['voxel_semantics']).unsqueeze(0)#[B,h,w,z]
        last_occ_onehot=mask2onehot(last_occ,nc=17)#[B,200,200,16,17]
        # one_hot = torch.nn.functional.one_hot(last_occ, 17)#这和上面是一样的
        continue

    #3D采样
    cur2past=torch.inverse(feat2bev)@torch.inverse(last_e2gmat)@e2g_mat@feat2bev
    last_loc=cur2past@loc3d
    # normalize_factor #缩放至[-1,1]才能grid sample loc:xyz
    normalize_factor = torch.tensor([X - 1.0, Y - 1.0, Z-1.0])
    last_loc3D = last_loc[...,:3,0] / normalize_factor.view(1, 1, 1, 3)*2 - 1.0#[B,h,w,16,3]
    sampled_occ_last=F.grid_sample(last_occ_onehot.permute(0,4,3,2,1),last_loc3D,align_corners=False)#gridsample与常规笛卡尔坐标系相反 x对应 width（最后一个空间维度），y 对应 height（倒数第二维度）。
    sampled_occ_res=torch.argmax(sampled_occ_last,dim=1)[0]#[(B),(C),H,W,Z]

    #2D采样
    B,H,W,Z,C=last_occ_onehot.shape
    last_occ_onehot2D=last_occ_onehot.permute(0,3,4,2,1).reshape(B*Z,C,W,H)#[B,h,w,z,c]->(Bz)cwh
    last_loc2D=last_loc[...,0,:2,0]/normalize_factor[:2].view(1,1,2)*2-1.0#[B,200,200,2]
    sampled_occ_last2D=F.grid_sample(last_occ_onehot2D,last_loc2D.expand(16,*last_loc2D.shape[1:]),align_corners=False)#gridsample与常规笛卡尔坐标系相反 x对应 width（最后一个空间维度），y 对应 height（倒数第二维度）。
    sampled_occ_last2D=sampled_occ_last2D.reshape(B,Z,C,W,H).permute(0,2,3,4,1)
    sampled_occ_res2D=torch.argmax(sampled_occ_last2D,dim=1)[0]

    outsave=f'{i}_occ.txt'
    nonfree=input_dict['voxel_semantics']<16
    results = np.hstack((indices[nonfree], input_dict['voxel_semantics'][nonfree][:, np.newaxis]))
    np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')

    outsave=f'{i}_occ_last.txt'
    nonfree=sampled_occ_res<16
    results = np.hstack((indices[nonfree],sampled_occ_res[nonfree][:, np.newaxis]))    
    np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')

    outsave=f'{i}_occ_last2Dgs.txt'
    nonfree=sampled_occ_res2D<16
    results = np.hstack((indices[nonfree],sampled_occ_res2D[nonfree][:, np.newaxis]))    
    np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
    pass