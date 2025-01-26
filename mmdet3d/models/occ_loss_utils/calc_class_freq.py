import mmcv
from tqdm import tqdm
import numpy as np
import json
pkl_path='data/nuscenes/bevdetv3-nuscenes_infos_train.pkl'
data_infos=mmcv.load(pkl_path)['infos']
num=data_infos.__len__()
num_class=17
total_voxels_class=[0]*num_class
total_voxels_class_ratios=[]
with open('mmdet3d/models/occ_loss_utils/class_voxel_freq.json','a') as f:
    f.write('\n'+'mask_v1cam_1216:' + '\n')
    f.close()

for i,data_info in tqdm(enumerate(data_infos)):
    datapath=(data_info['occv2_path']+'/labels.npz')#.replace('openocc_v2','openocc_v2_nextrend1112')
    data=np.load(datapath)
    sem_gt=data['semantics'][data['vismask']]
    for j in range(num_class):
        total_voxels_class[j]+=np.sum(sem_gt==j)
totalvoxs=sum(total_voxels_class)
for i,class_vox in enumerate(total_voxels_class):
    total_voxels_class_ratios.append(class_vox/totalvoxs)
with open('mmdet3d/models/occ_loss_utils/class_voxel_freq.json','a') as f:
    f.write(json.dumps(str(total_voxels_class)) + '\n')
    for total_voxel,voxel_ratio in zip(total_voxels_class,total_voxels_class_ratios):
        f.write(json.dumps(str(total_voxel))+' '+str(voxel_ratio)+',\n')