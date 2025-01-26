import torch
import numpy as np
import mmcv
from tqdm import tqdm
import os
#采用occv1的mask作为visible mask

if __name__=="__main__":
    # root=Path("/root/data/chuyunfeng/OccNet_/data/nuscenes/openocc_v2")
    # save_root=Path("data/nuscenes/openocc_v2_mask")
    # npzfiles=glob.glob(os.path.join(root,"**/*.npz"),recursive=True)[:10]
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


    ann_file='data/nuscenes/bevdetv3-nuscenes_infos_val.pkl'
    print("processing:",ann_file)
    data = mmcv.load(ann_file)
    data_infos = data['infos']
    data_infos=sorted(data_infos,key= lambda x:x['timestamp']) 
    # data_infos=data_infos[:200]
    for i,info in tqdm(enumerate(data_infos)):
        occ_gt = dict(np.load(info['occv2_path']+'/labels.npz', allow_pickle=True))
        occ_gtv1=np.load(info['occ_path']+'/labels.npz', allow_pickle=True)
        occ_gt['vismask']=occ_gtv1['mask_camera'].astype(bool)
        save_path=info['occv2_path'].replace('openocc_v2','openocc_v2_v1mask1216')
        if save_path[-4:]!=".npz":
            save_path=save_path+"/labels.npz"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.savez_compressed(save_path,**occ_gt)