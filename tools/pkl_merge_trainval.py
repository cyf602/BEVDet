import mmcv
import numpy as np
import argparse
from tqdm import tqdm
"""
对多个pkl中需求的字段进行融合
"""
parser = argparse.ArgumentParser()
parser.add_argument('--source-path1', type=str, default='data/nuscenes/bevdetv3-nuscenes_infos_train.pkl', help='pkl1 path')#QAF2d
parser.add_argument('--source-path2', type=str, help='pkl2 path')#bevdet
parser.add_argument('--save-path', type=str, help='pkl save path')
opt = parser.parse_args()

save_path=opt.source_path1.replace('train','trainval') if opt.save_path is None else opt.save_path
path2=opt.source_path1.replace('train','val') if opt.source_path2 is None else opt.source_path2
data1=mmcv.load(opt.source_path1)#base 将2中参数给1
data2=mmcv.load(path2)
# data1_infos=data1['infos']
data1['infos'].extend(data2['infos'])
mmcv.dump(data1,save_path)