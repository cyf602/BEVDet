import mmcv
import numpy as np
import argparse
from tqdm import tqdm
"""
对多个pkl中需求的字段进行融合
"""
parser = argparse.ArgumentParser()
parser.add_argument('--source-path1', type=str, default='data/nuscenes/bevdetv3-nuscenes_infos_train.pkl', help='pkl1 path')#QAF2d
parser.add_argument('--source-path2', type=str, default='/root/autodl-fs/pkls/nuscenes2d-stream_temporal_infos_train.pkl', help='pkl2 path')#bevdet
parser.add_argument('--save-path', type=str, help='pkl save path')
opt = parser.parse_args()

save_path=opt.source_path1 if opt.save_path is None else opt.save_path
data1=mmcv.load(opt.source_path1)#base 将2中参数给1
data2=mmcv.load(opt.source_path2)
# data1_infos=data1['infos']
data2_infos=data2['infos']
length=len(data1['infos'])
assert(length==len(data2_infos))
for i in tqdm(range(length)):
    data1['infos'][i]['bboxes2d_xyxy']=data2_infos[i]['bboxes2d']
    data1['infos'][i]['labels2d']=data2_infos[i]['labels2d']
    data1['infos'][i]['centers2d']=data2_infos[i]['centers2d']
    data1['infos'][i]['bboxdepths2d']=data2_infos[i]['depths']
mmcv.dump(data1,save_path)