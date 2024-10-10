import numpy as np
import pickle
import random
from mmcv import dump
from tqdm import tqdm
flow_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian',
]
_voxel_size=0.4
def convert2cloud(pkldata,length,output_root='test_results/clouds/1008/'):
    gtdata=pkldata['gt_dict']
    prdata=pkldata['pr_dict']
    for i in tqdm(range(length)):
        gt_coors=gtdata[i]['coord_indexs']
        pr_coors=prdata[i]['coord_indexs']
        gt_sem=gtdata[i]['selected_label'].reshape(-1)
        pr_sem=prdata[i]['selected_label'].reshape(-1)
        gt_dist=gtdata[i]['pred_dists']
        pr_dist=prdata[i]['pred_dists']
        
        tp_class_mask=gt_sem==pr_sem #类别正确
        dist_error=np.abs(pr_dist-gt_dist)
        
        tp_dist_mask=(dist_error<2).reshape(-1)
        non_free_gt=gt_sem!=16
        non_free_pr=pr_sem!=16
        final_mask=np.logical_and(tp_dist_mask,tp_class_mask)
        final_mask=np.logical_and(final_mask,non_free_gt)
        gt_flow_select=gtdata[i]['selected_flow'][final_mask]
        pr_flow_select=prdata[i]['selected_flow'][final_mask]
        flow_error=np.linalg.norm(gt_flow_select-pr_flow_select,axis=1)
        flow_gt_norm=np.linalg.norm(gt_flow_select,axis=1)
        flow_pr_norm=np.linalg.norm(pr_flow_select,axis=1)
        mAVE=np.mean(flow_error)    
        flow_error=np.clip(flow_error,0,4)
        results = np.hstack((pr_coors[final_mask]*_voxel_size, flow_error[:, np.newaxis]))
        pr_sem_results=np.hstack((pr_coors[non_free_pr]*_voxel_size, pr_sem[non_free_pr][:, np.newaxis]))
        gt_sem_results=np.hstack((gt_coors[non_free_gt]*_voxel_size, gt_sem[non_free_gt][:, np.newaxis]))
        flow_gt_norm=np.hstack((gt_coors[final_mask]*_voxel_size, flow_gt_norm[:, np.newaxis]))
        flow_pr_norm=np.hstack((pr_coors[final_mask]*_voxel_size, flow_pr_norm[:, np.newaxis]))
        np.savetxt(output_root+str(i)+'flow_error.txt', results, fmt='%.7f', delimiter=',', header='x,y,z,value', comments='')
        np.savetxt(output_root+str(i)+'gt_flow.txt', flow_gt_norm, fmt='%.7f', delimiter=',', header='x,y,z,value', comments='')
        np.savetxt(output_root+str(i)+'pr_flow.txt', flow_pr_norm, fmt='%.7f', delimiter=',', header='x,y,z,value', comments='')
        np.savetxt(output_root+str(i)+'pr_sem.txt', pr_sem_results, fmt='%.2f', delimiter=',', header='x,y,z,value', comments='')
        np.savetxt(output_root+str(i)+'gt_sem.txt', gt_sem_results, fmt='%.2f', delimiter=',', header='x,y,z,value', comments='')
    
        

def choose_few_data(pkldata):
    outdict={}
    for k,v in pkldata.keys():
        outdict[k]=[]
    for k,v in pkldata.items():
        if random.randint(1,10)==7:
            outdict[k].append(v)
    dump(outdict,"test_results/resultckpt905_1-10.pkl")        

            
            

if __name__=="__main__":
    pkl_path='test_results/resultckpt1008_epoch30.pkl'
    choosedata=False
    with open(pkl_path,"rb") as f:
        pkldata=pickle.load(f)
    if choosedata:
        choose_few_data(pkldata)
    pass
    length=len(pkldata['gt_dict'])
    convert2cloud(pkldata,length)
    # coors_pred=pkldata['coors_pred']
    # coors_gt=pkldata['coors_gt']
    
    
    
