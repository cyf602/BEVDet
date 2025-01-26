import torch
import torch.nn as nn
pth_path='ckpts/bevdet-r50-4d-stereo-cbgs.pth'
save_path='ckpts/bevdet-r50-bevenc.pth'
checkpoint=torch.load(pth_path)
selected={}
for k,v in checkpoint.items():
    pass
torch.save(selected,save_path)
