import cv2
import os
from tqdm import tqdm
import shutil
import glob
import numpy as np
#用于验证2Dgt的正确性
json_path='data/nuscenes/nuscenes_infos_val.coco.json'
img_root='data/nuscenes/'
save_root="test/vis_2d_labels_in_model/"
label_root='data/nuscenes/nusc_2d_yolo'
vis_id=0
os.makedirs(save_root,exist_ok=True)
cam_types=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT']
colormap={
    0:(255, 158, 0),  # Orange
    1:(255, 99, 71),  # Tomato
    2: (233, 150, 70),  # Darksalmon
    3: (255, 69, 0),  # Orangered
    4: (255, 140, 0),  # Darkorange
    5: (112, 128, 144),  # Slategrey
    6: (255, 61, 99),  # Red
    7: (220, 20, 60),  # Crimson
    8: (0, 0, 230),  # Blue
    9: (47, 79, 79),  # Darkslategrey
}    
id2name={
    0:'car', 1:'truck', 2:'construction_vehicle', 3:'bus', 
    4:'trailer', 5:'barrier', 6:'motorcycle', 
    7:'bicycle', 8:'pedestrian', 9:'traffic_cone'
}
#export_to_2d时的类别
nus_categories = {0:'car', 1:'truck', 2:'trailer', 3:'bus', 4:'construction_vehicle',
                  5:'bicycle', 6:'motorcycle', 7:'pedestrian', 8:'traffic_cone',
                  9:'barrier'}


def vis_img_and_labels(imgs,bboxes,labels=None):
    """
    bboxes,labels: len=6 list, torch shape[n,4],[n]
    imgs:[6,3,267,704] np array uint8
    """
    global vis_id
    assert len(imgs)==len(bboxes)==6
    l=len(imgs)
    for i,cam in enumerate(cam_types):
        img=imgs[i].transpose(1,2,0).copy()
        for j,bbox in enumerate(bboxes[i]):#tensor [N,4]
            bbox=bbox.int().cpu().numpy()
            catid=int(labels[i][j]) if labels is not None else 0
            color=colormap[catid]
            cls_name=nus_categories[catid]
            cv2.rectangle(img,bbox[:2],bbox[2:],color,2)
        cv2.imwrite(save_root+str(vis_id)+str(cam)+'.jpg',img)
        print('save 2d img and labels:',save_root+str(vis_id)+str(cam)+'.jpg')
    vis_id+=1
    
def vis_single_det_and_seg(img,bboxes=None,labels=None,seg=None,cam="un",idx=None):
    """为一张图可视化检测框和分割
    
    """
    img=np.array(img)
    global vis_id
    if idx is None:
        idx=vis_id 
    if bboxes is not None:
        for j,bbox in enumerate(bboxes):#np [N,4]
            bbox=bbox.astype(np.int16)#.cpu().numpy()
            catid=int(labels[j]) if labels is not None else 0
            color=colormap[catid]
            cls_name=nus_categories[catid]
            cv2.putText(img,cls_name,bbox[:2],fontFace=2,fontScale=1.,color=color)
            cv2.rectangle(img,bbox[:2],bbox[2:],color,2)
        cv2.imwrite(save_root+str(vis_id)+str(cam)+'.jpg',img)
    if seg:
        # seg=seg.transpose(1,0)#to 704 256
        W,H=seg.shape
        seg_pic=np.ones((W,H,3),dtype=np.uint8)*255
        for typeid in range(10):
            color=colormap[typeid]
            seg_pic[seg==typeid]=color
        cv2.imwrite(save_root+str(idx)+str(cam)+'seg.jpg',seg_pic)
    vis_id+=1