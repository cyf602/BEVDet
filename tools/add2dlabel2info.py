import mmcv
import numpy as np
import os
from tqdm import tqdm
pkl_path='data/nuscenes/bevdetv3-nuscenes_infos_val.pkl'
label2d_root='data/nuscenes/nusc_2d_yolo'
data=mmcv.load(pkl_path)
for info in tqdm(data['infos']):
    for cam,detail in info['cams'].items():
            
        label2dpath=os.path.join(label2d_root,*detail['data_path'][:-4].split('/')[-2:])+'.txt'
        if not os.path.exists(label2dpath):
            detail['labels2d']=np.array([])
            detail['bboxes2d']=np.array([]).reshape(0,4)
            continue
        
        labelstrs=open(label2dpath).readlines()
        bboxes=[]#xywh
        labels=[]#标签
        for lablestr in labelstrs:
            labellist=list(map(float,lablestr.split()))
            bboxes.append(labellist[1:])
            labels.append(int(labellist[0]))
        detail['labels2d']=np.array(labels)
        detail['bboxes2d']=np.array(bboxes)
mmcv.dump(data,pkl_path)
# mmcv.dump(data,pkl_path)