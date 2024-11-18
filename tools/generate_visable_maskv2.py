import os
os.environ["CUDA_VISIBLE_DEVIES"]="3"
import glob
import numpy as np
import math
import torch
import copy
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes
from .ray_iou.ego_pose_extractor import EgoPoseDataset
import mmcv
from torch.utils.cpp_extension import load
# import multiprocessing
dvr = load("dvr", sources=["tools/ray_iou/lib/dvr/dvr.cpp", "tools/ray_iou/lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
center=np.array([100,100,6])
num=0
# from projects.mmdet3d_plugin.datasets.ray_metrics import main as ray_based_miou
# from projects.mmdet3d_plugin.datasets.ray_metrics import process_one_sample,generate_coords
# X,Y,Z=200,200,16
_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4
free_id=16
# occupancy_size=[0.4, 0.4, 0.4],
# point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4],
# xs=torch.linspace(point_cloud_range[0]+occupancy_size[0]/2,point_cloud_range[3]-occupancy_size[0]/2,X).view(X,1,1).expand(X,Y,Z)#200
# ys=torch.linspace(point_cloud_range[1]+occupancy_size[1]/2,point_cloud_range[4]-occupancy_size[1]/2,Y).view(1,Y,1).expand(X,Y,Z)#200
# zs=torch.linspace(point_cloud_range[2]+occupancy_size[2]/2,point_cloud_range[5]-occupancy_size[2]/2,Z).view(1,1,Z).expand(X,Y,Z)#16
# local_voxel_coors=torch.stack((xs,ys,zs),-1).double().repeat(1,1,1,1,1)#para1:batch size=1

# xs=torch.linspace(0,X-1,X)#.view(X,1,1).expand(X,Y,Z)#200
# ys=torch.linspace(0,Y-1,Y)#.view(1,Y,1).expand(X,Y,Z)#200
# zs=torch.linspace(0,Z-1,Z)#.view(1,1,Z).expand(X,Y,Z)#200
# view_grid=torch.stack((xs,ys,zs),-1).double().repeat(1,1,1,1,1)#para1:batch size=1
Vend_size=4#每个结尾处周围部分点设为mask中
def process_voxel_slice(view_grid, center, free_id, x_range=None):
    x_voxels, y_voxels, z_voxels = view_grid.shape
    x_center, y_center, z_center = center
    voxels_mask_slice = np.zeros_like(view_grid, dtype=bool)
    t_values = np.linspace(0, 1, 768).reshape(-1, 1)

    for i in range(x_voxels):
        for j in range(y_voxels):
            for k in range(z_voxels):
                if view_grid[i, j, k] != free_id:
                    # indices = np.round(
                    #     np.array([i, j, k])
                    #     + t_values * (np.array([x_center, y_center, z_center]) - np.array([i, j, k]))
                    # ).astype(int)#线段采样一些点 round成voxel索引
                    indices = np.round(
                        np.array([x_center, y_center, z_center])
                        + t_values * (np.array([i, j, k])-np.array([x_center, y_center, z_center]))
                    ).astype(int)#线段采样一些点 round成voxel索引
                    mask = ~np.all(indices == [i, j, k], axis=1)
                    indices = indices[mask]
                    if (view_grid[indices[:, 0], indices[:, 1], indices[:, 2]] != free_id).sum() > 1:
                        #线段上有其他非空点就舍弃当前线段
                        continue
                    voxels_mask_slice[indices[:, 0], indices[:, 1], indices[:, 2]] = True
                    voxels_mask_slice[i, j, k] = True
                    voxels_mask_slice[max(i-Vend_size,0):min(x_voxels-1,i+Vend_size),
                                      max(j-Vend_size,0):min(y_voxels-1,j+Vend_size),
                                      max(k-Vend_size,0):min(z_voxels-1,k+Vend_size)] = True
    return voxels_mask_slice.astype(np.bool)

def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)
    
    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)

def process_one_npz(batch,i,data_infos):
    global center,num
    if num%100==0:print(num)
    num+=1
    vismask=np.zeros((200,200,16),dtype=np.bool)
    token = batch[0][0]
    output_origin = batch[1]
    T = output_origin.shape[1]
    assert data_infos[i]['token']==token
    # data_id = sample_tokens.index(token)
    # info = data_infos[data_id]
    info =data_infos[i]
    occ_gt = dict(np.load(info['occv2_path']+'/labels.npz', allow_pickle=True))
    gt_semantics = occ_gt['semantics']
    instances=occ_gt['instances']
    occ_pred = copy.deepcopy(gt_semantics)
    occ_pred[gt_semantics < free_id] = 1
    occ_pred[gt_semantics == free_id] = 0
    occ_pred = torch.from_numpy(occ_pred).permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()#[1,1,16,200,200]
    coord_indexs=[]
    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]
    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
    for t in range(T): 
        lidar_origin = output_origin[:, t:t+1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ_pred.cuda(),#[1,1,16,200,200]
                output_origin_render.cuda(),#[1,1,3]
                output_points_render.cuda(),#[1,14040,3]
                output_tindex_render.cuda(),#[1,14040]
                [1, 16, 200, 200],
                "test"
            )
            pred_dist *= _voxel_size
        coord_index = coord_index[0, :, :].int().cpu()  # [N, 3] int体素序号
        coord_indexs.append(coord_index)
    coord_indexs=torch.cat(coord_indexs, dim=0).cpu().numpy()#[8N,3]
    coord_indexs=np.unique(coord_indexs,axis=0)#end of rays 11w->2.8w
    choose_labels=gt_semantics[tuple(coord_indexs.T)]
    choose_labels_idx=np.where(choose_labels!=16)
    coord_indexs=coord_indexs[choose_labels_idx]#非空端点
    vismask[tuple(coord_indexs.T)]=True
    # indices=np.round(coord_indexs).astype(int).reshape(-1,3)
    #在每个ray上采样,要保证端点采样
    t_values = np.linspace(0, 1, 41)
    for coord_index in coord_indexs:    
        for t_value in t_values:###遮挡检测
            indice=np.round(center+t_value*(coord_index-center)).astype(int)
            if gt_semantics[indice[0],indice[1],indice[2]]!=16:
                break
            else:
                vismask[indice[0],indice[1],indice[2]]=True
    # indices = np.round(
    #                 center
    #                 + t_values * (coord_indexs-center)
    #             ).astype(int).reshape(-1,3)#线段采样一些点 round成voxel索引
    # indices[:,0]=np.clip(indices[:,0],0,199)
    # indices[:,1]=np.clip(indices[:,1],0,199)
    # indices[:,2]=np.clip(indices[:,2],0,15)
    # indices=np.unique(indices,axis=0)
    # mask = ~np.all(indices == coord_indexs, axis=1)
    # indices = indices[mask]
    # indices=coord_indexs
    # vismask[tuple(indices.T)]=True
    
    #补全照射到的instances
    # addmask=np.zeros_like(vismask)
    # visidxs=np.unique(instances[vismask])
    # for visidx in visidxs:
    #     # if visidx==0:continue#empty
    #     if gt_semantics[instances==visidx][0]>=10:
    #         continue #非常规detection类别 包含barrier，cone
    #     vismask[instances==visidx]=True
    
    
    
    #save
    occ_gt['vismask']=vismask
    save_path=info['occv2_path'].replace('openocc_v2','openocc_v2_mask21_1017')
    if save_path[-4:]!=".npz":
        save_path=save_path+"/labels.npz"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.savez_compressed(save_path,**occ_gt)
    print(save_path)
    data.close()
  
def gen_vis_mask_atT(center,coord_indexs,vismask):
    coord_indexs=np.unique(coord_indexs.cpu().numpy(),axis=0)#end of rays 11w->2.8w
    choose_labels=gt_semantics[tuple(coord_indexs.T)]
    choose_labels_idx=np.where(choose_labels!=16)
    coord_indexs=coord_indexs[choose_labels_idx]#非空端点
    vismask[tuple(coord_indexs.T)]=True
    # indices=np.round(coord_indexs).astype(int).reshape(-1,3)
    #在每个ray上采样,要保证端点采样(用于遮挡判断除外)
    t_values = np.linspace(0, 0.9, 11).tolist()
    for coord_index in coord_indexs:                  
        # if 10<=gt_semantics[coord_index[0],coord_index[1],coord_index[2]]<=13:
        #     #处理地面类别'driveable_surface', 'other_flat', 'sidewalk','terrain',
        #     vismask[coord_index[0],coord_index[1],coord_index[2]:coord_index[2]+3]=True
        if gt_semantics[coord_index[0],coord_index[1],coord_index[2]]<16:
            #等分采样（此处可能仅用于遮挡判断）
            obstacled=False#是否遮挡
            for t_value in t_values:###遮挡检测
                indice=np.round(center+t_value*(coord_index-center)).astype(int)
                if indice[2]>15 or indice[0]>199 or indice[1]>199 or min(indice)<0:
                    obstacled=True#出界判断？scene~416/417 42/43
                    break
                if gt_semantics[indice[0],indice[1],indice[2]]!=16:
                    obstacled=True
                    break
            #     else:
            #         vismask[indice[0],indice[1],indice[2]]=True

            #等距采样
            if obstacled:
                continue
            dist=np.linalg.norm(coord_index-center)#单位 格
            if gt_semantics[coord_index[0],coord_index[1],coord_index[2]]<10:
                dist_add_values=np.linspace(0, min(4,0.5*dist), 6)#包含首尾
            elif gt_semantics[coord_index[0],coord_index[1],coord_index[2]]>13:
                dist_add_values=np.linspace(0, min(4,0.5*dist), 6)
            else:
                continue
            t_values_add=(1-dist_add_values/dist).tolist()
            for t_value in t_values_add:
                indice=np.round(center+t_value*(coord_index-center)).astype(int)
                # if gt_semantics[indice[0],indice[1],indice[2]]!=16:    
                #     break
                vismask[indice[0],indice[1],indice[2]]=True
            # dist_add_values=np.linspace(max(0,dist-50), dist, 11)
            # t_values_add=dist_add_values/dist
            # # dist=min(5,0.1*np.linalg.norm(coord_index-center))
            # for t_value in t_values_add:
            #     indice=np.round(center+t_value*(coord_index-center)).astype(int)
            #     vismask[indice[0],indice[1],indice[2]]=True
            # t_values=np.sort((np.concatenate((t_values,t_values_add))))#unique函数自动排序
            # t_values.extend(t_values_add)
            # t_values.sort()
        # for t_value in t_values:###遮挡检测
        #     indice=np.round(center+t_value*(coord_index-center)).astype(int)
        #     if gt_semantics[indice[0],indice[1],indice[2]]!=16:
        #         break
        #     else:
        #         vismask[indice[0],indice[1],indice[2]]=True

    # indices = np.round(
    #                 center
    #                 + t_values * (coord_indexs-center)
    #             ).astype(int).reshape(-1,3)#线段采样一些点 round成voxel索引
    # indices[:,0]=np.clip(indices[:,0],0,199)
    # indices[:,1]=np.clip(indices[:,1],0,199)
    # indices[:,2]=np.clip(indices[:,2],0,15)
    # indices=np.unique(indices,axis=0)
    # mask = ~np.all(indices == coord_indexs, axis=1)
    # indices = indices[mask]
    # indices=coord_indexs
    # vismask[tuple(indices.T)]=True
    
    #补全照射到的instances
    # visidxs=np.unique(instances[vismask])
    # for visidx in visidxs:
    #     # if visidx==0:continue#empty
    #     if gt_semantics[instances==visidx][0]>=10:
    #         continue #非常规detection类别 包含barrier，cone
    #     vismask[instances==visidx]=True
    
    #补充对地面类别的处理 'driveable_surface', 'other_flat', 'sidewalk','terrain’
    ground_idxs=np.where(np.logical_and(np.logical_and((10<=gt_semantics),(gt_semantics<=13)),vismask))
    i=1
    while i<=1:#将上方若干格置True
        upperz=np.clip(i+ground_idxs[2],0,15)
        vismask[ground_idxs[0],ground_idxs[1],upperz]=True
        i+=1
            
if __name__=="__main__":
    # root=Path("/root/data/chuyunfeng/OccNet_/data/nuscenes/openocc_v2")
    # save_root=Path("data/nuscenes/openocc_v2_mask")
    # npzfiles=glob.glob(os.path.join(root,"**/*.npz"),recursive=True)[:10]
    lidar_rays=generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)
    # center=np.array([100,100,2.5])
    # for i,npzfile in tqdm(enumerate(npzfiles)):
    #     data=np.load(npzfile)
    #     sem_gt=data['semantics']
    #     flow_gt=data['flow']
    #     save_data=dict(data)
    #     vismask=process_voxel_slice(sem_gt,center,free_id=16)
    #     save_data['vis_mask']=vismask
    #     savedir=save_root/Path(npzfile).relative_to(root)
    #     if not savedir.parent.exists():
    #         savedir.parent.mkdir(parents=True)
    #     np.savez_compressed(savedir,**save_data)

        # pcd_gt = process_one_sample(sem_gt, lidar_rays, lidar_origins, flow_gt)#[8*N,4]

        # gt_dict=generate_coords(sem_gt, lidar_rays, lidar_origins,flow_gt)
    ann_file='data/nuscenes/bevdetv3-nuscenes_infos_train.pkl'
    print("processing:",ann_file)
    data = mmcv.load(ann_file)
    data_infos = data['infos']
    data_infos=sorted(data_infos,key= lambda x:x['timestamp'])    
    # data_infos=data_infos[:10000]
    # num_workers=16
    # files_per_worker=len(data_infos)//num_workers
    # sub_files_split=[]
    # pool=multiprocessing.Pool(processes=16)
    
    ego_pose_dataset = EgoPoseDataset(data_infos, dataset_type='openocc_v2')

    data_loader_kwargs={
        "pin_memory": False,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 8,
    }

    data_loader = DataLoader(
        
        ego_pose_dataset,
        **data_loader_kwargs,
    )

    # sample_tokens = [info['token'] for info in data_infos]
    center=np.array([100,100,6])
    # with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    #     for i, batch in tqdm(enumerate(data_loader), ncols=50):
    #         executor.submit(process_one_npz, batch,i,data_infos)
        # {executor.submit(process_one_npz, element): element for element in tqdm(enumerate(data_loader), ncols=50)}
    # for i, batch in tqdm(enumerate(data_loader), ncols=50):
    #     for data in batch:
    #         results=pool.apply_async(process_one_npz,data)
    for i, batch in tqdm(enumerate(data_loader), ncols=50):
    # for i, batch in enumerate(data_loader):
        vismask=np.zeros((200,200,16),dtype=np.bool)
        token = batch[0][0]
        output_origin = batch[1]
        T = output_origin.shape[1]
        assert data_infos[i]['token']==token
        # data_id = sample_tokens.index(token)
        # info = data_infos[data_id]
        info =data_infos[i]
        if os.path.exists(info['occv2_path'].replace('openocc_v2','openocc_v2_nextrend1112')):
            print(i,"finisned")
            continue
        # else:
        #     print(i,"unfinished")
        #     continue
        if i%40==0:
            print(info['occv2_path'].split('/')[4],'time:',info['timestamp'])
        occ_gt = dict(np.load(info['occv2_path']+'/labels.npz', allow_pickle=True))
        gt_semantics = occ_gt['semantics']
        instances=occ_gt['instances']
        occ_pred = copy.deepcopy(gt_semantics)
        occ_pred[gt_semantics < free_id] = 1
        occ_pred[gt_semantics == free_id] = 0
        occ_pred = torch.from_numpy(occ_pred).permute(2, 1, 0)
        occ_pred = occ_pred[None, None, :].contiguous().float()#[1,1,16,200,200]
        coord_indexs=[]
        offset = torch.Tensor(_pc_range[:3])[None, None, :]
        scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]
        lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
        for t in range(T): 
            lidar_origin = output_origin[:, t:t+1, :]  # [1, 1, 3]起点？
            lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

            output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
            output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
            output_tindex_render = lidar_tindex  # [1, N], all zeros

            with torch.no_grad():
                pred_dist, _, coord_index = dvr.render_forward(
                    occ_pred.cuda(),#[1,1,16,200,200]
                    output_origin_render.cuda(),#[1,1,3]
                    output_points_render.cuda(),#[1,14040,3]
                    output_tindex_render.cuda(),#[1,14040]
                    [1, 16, 200, 200],
                    "test"
                )
                pred_dist *= _voxel_size
            coord_index = coord_index[0, :, :].int().cpu()  # [N, 3] int体素序号
            gen_vis_mask_atT(output_origin_render[0][0].cpu().numpy(),coord_index,vismask)
        
        
        #save
        occ_gt['vismask']=vismask
        save_path=info['occv2_path'].replace('openocc_v2','openocc_v2_nextrend1112')
        if save_path[-4:]!=".npz":
            save_path=save_path+"/labels.npz"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.savez_compressed(save_path,**occ_gt)
        #按时序保存
        # save_path=os.path.join(*save_path.split('/')[:-2])+'/'+str(i)+".npz"
        # np.savez_compressed(save_path,**occ_gt)