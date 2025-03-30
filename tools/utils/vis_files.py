import torch
import numpy as np
import cv2
import os
import mmcv
import glob

from vis_bev import vis_gt_txt

root='data/nuscenes/openocc_v2_sky330'
save_vis_root=root+'_vis'
if not os.path.exists(save_vis_root):  
    os.mkdir(save_vis_root)
npz_files=glob.glob(os.path.join(root,'**/**.npz'),recursive=True)
for npz_file in npz_files:
    vis_gt_txt(npz_file,save_vis_root)
    # data=np.load(npz_file)
    pass