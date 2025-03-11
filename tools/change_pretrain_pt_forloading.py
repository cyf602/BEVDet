from collections import OrderedDict
import torch

pth_path='ckpts/det2d_depth_b16_310ep5.pth'
dest_pth_path="ckpts/bevdet-r50-4d-depth-cbgs.pth"
dest_dict=torch.load(dest_pth_path)['state_dict']
weights = torch.load(pth_path)
stat_dict=weights['state_dict']
new_stat_dict=OrderedDict()
for k,v in stat_dict.items():
    if k in dest_dict.keys():
        new_stat_dict[k]=stat_dict[k]
    else:
        print(k)
        if k.split('.')[0]=='depthnet':
            newk='img_view_transformer.'+k.replace('depthnet','depth_net')
            assert newk in dest_dict.keys()
            new_stat_dict[newk]=stat_dict[k]
            pass

weights['state_dict']=new_stat_dict
torch.save(weights,pth_path.split('.')[0]+'_fitted.pth')