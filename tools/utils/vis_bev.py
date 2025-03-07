import torch
import numpy as np
import cv2
import os
import mmcv
from datetime import datetime
now=datetime.now()
time_str = now.strftime("%Y-%m-%d-%H:%M:%S")
white_board=np.ones((200,200,3))*255
colors_map=np.array([
            [0, 150, 245, 255],  # car                  blue
            [160, 32, 240, 255],  # truck                purple
            [135, 60, 0, 255],  # trailer              brown
            [255, 255, 0, 255],  # bus                  yellow
            [0, 255, 255, 255],  # construction_vehicle cyan
            [255, 192, 203, 255],  # bicycle              pink
            [200, 180, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [255, 120, 50, 255],  # barrier              orangey
            [255, 0, 255, 255],  # driveable_surface    dark pink
            [175,   0,  75, 255],       # other_flat           dark red
            [75, 0, 75, 255],  # sidewalk             dard purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
            [255, 255, 255, 255],  # free             white
        ], dtype=np.uint8)[:, :3]
V_MAX_THR=-1
def vis_occ(semantics, flows,use_minv_thr=True,v_max_thr=-1):
    H, W, D = semantics.shape
    semantics_valid=(semantics!=16)#0~16类别号
    # semantics_valid = np.logical_not(semantics == 0)#
    # print("semantics_valid: ", semantics_valid.shape)
    # d = np.arange(D).reshape(1, 1, D)#0~D
    # d = np.repeat(d, H, axis=0)#
    # d = np.repeat(d, W, axis=1).astype(np.float32)#H,W,D D维度上1~D
    # d = d * semantics_valid
    d=torch.arange(D).repeat(H,W,1).to(semantics.device)*semantics_valid
    selected = torch.argmax(d, axis=-1)#最高点序号？

    # selected_torch = torch.from_numpy(selected)
    # semantics_torch = torch.from_numpy(semantics)

    sem_occ_bev_torch = torch.gather(semantics, dim=2,
                                    index=selected.unsqueeze(-1))#最高点语义
    non_free_pillar=(sem_occ_bev_torch!=16).unsqueeze(-1)
    flows*=non_free_pillar
    # print("occ_bev_torch: ", occ_bev_torch.shape)
    # flow_occ_bev_x=torch.gather(flows[...,0], dim=2,index=selected.unsqueeze(-1)).cpu().numpy()
    # flow_occ_bev_y=torch.gather(flows[...,1], dim=2,index=selected.unsqueeze(-1)).cpu().numpy()
    flow_occ_bev_x=torch.max(flows[...,0],dim=2).values.unsqueeze(-1).cpu().numpy()
    flow_occ_bev_y=torch.max(flows[...,1],dim=2).values.unsqueeze(-1).cpu().numpy()
    flow_occ_bev_v=np.sqrt(flow_occ_bev_x**2+flow_occ_bev_y**2)
    occ_bev = sem_occ_bev_torch.cpu().numpy()#[B,200,200,1]
    if v_max_thr>0:
        max_v_thr=v_max_thr#按指定阈值可视化
    else:
        max_v_thr=max(np.max(flow_occ_bev_v),0.1)
        V_MAX_THR=max_v_thr
    min_v_thr=0.1#小于该值的不画出来
    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = colors_map[occ_bev].astype(np.uint8)#bev视角下各最高点语义颜色
    occ_bev_vis = occ_bev_vis.reshape(H,W,3)#[::-1, ::-1, :3]#::-1倒序？
    occ_bev_vis = cv2.resize(occ_bev_vis,(1024,1024))
    v_vis_bases=[]#x,y,v 按最大速度的比例画
    v_vis_minthrs=[]#
    v_vis_maxthrs=[]#按与预设值的比值画
    channel_change=np.array([0,0,1])[None,None,:]
    for v in (flow_occ_bev_x,flow_occ_bev_y,flow_occ_bev_v):
        maxv=max(np.max(v),1e-6)
        v_vis_base=v/maxv*255*channel_change
        v_vis_maxthr=v/max_v_thr*255*channel_change
        v=v.copy()
        # v[v<min_v_thr]=0
        v[v<max(maxv/10,1e-1)]=0
        v_vis_bases.append(cv2.resize(v_vis_base,(1024,1024)))
        v_vis_maxthrs.append(cv2.resize(v_vis_maxthr,(1024,1024)))
        
                    
    # flow_occ_bev_x_vis=flow_occ_bev_x/np.max(flow_occ_bev_x+1e-6)*255
    # flow_occ_bev_y_vis=flow_occ_bev_y/np.max(flow_occ_bev_y+1e-6)*255
    # flow_occ_bev_v_vis=flow_occ_bev_v/np.max(flow_occ_bev_v+1e-6)*255
    # # flow_occ_bev_x_vis[flow_occ_bev_x_vis<min_v_thr]=0
    # # flow_occ_bev_y_vis[flow_occ_bev_y_vis<min_v_thr]=0
    # # flow_occ_bev_v_vis[flow_occ_bev_v_vis<min_v_thr]=0
    # flow_occ_bev_x_vis[flow_occ_bev_x_vis<np.max(flow_occ_bev_x+1e-6)/10]=0
    # flow_occ_bev_y_vis[flow_occ_bev_y_vis<np.max(flow_occ_bev_y+1e-6)/10]=0
    # flow_occ_bev_v_vis[flow_occ_bev_v_vis<np.max(flow_occ_bev_v+1e-6)/10]=0
    # # flow_occ_bev_x_vis=np.minimum(1,flow_occ_bev_x/max_v_thr)*255
    # # flow_occ_bev_y_vis=np.minimum(1,flow_occ_bev_y/max_v_thr)*255
    # # flow_occ_bev_v_vis=np.minimum(1,flow_occ_bev_v/max_v_thr)*255
    # channel_change=np.array([0,0,1])[None,None,:]
    # flow_occ_bev_x_vis=flow_occ_bev_x_vis*channel_change#[200,200,1]*[1,1,3]
    # flow_occ_bev_y_vis=flow_occ_bev_y_vis*channel_change
    # flow_occ_bev_v_vis=flow_occ_bev_v_vis*channel_change
    # flow_occ_bev_x_vis=cv2.resize(flow_occ_bev_x_vis,(1024,1024))
    # flow_occ_bev_y_vis=cv2.resize(flow_occ_bev_y_vis,(1024,1024))
    # flow_occ_bev_v_vis=cv2.resize(flow_occ_bev_v_vis,(1024,1024))
    return occ_bev_vis,np.concatenate(v_vis_bases,axis=1),np.concatenate(v_vis_maxthrs,axis=1)

def vis_bev_view(occ_preds=None,occ_gts=None,flow_preds=None,flow_gts=None,idx=0,
                 flowmask=None,occmask=None,save_root='/root/data/chuyunfeng/OccNet_/vis/bev_tiny'):
    """
    occ_preds/occ_gts:[B,200,200,16] int
    low_preds/flow_gts:[B,200,200,16,2]
    """
    V_MAX_THR=-1
    if occ_preds is None:
        occ_preds=torch.zeros_like(occ_gts)
    if flow_preds is None:
        flow_preds=torch.zeros_like(flow_gts)
    flowmask_pic=white_board.copy()
    occmask_pic=white_board.copy()
    if flowmask is not None:
        flowmask=torch.sum(flowmask,dim=-1)[0,...]
        flowmask=(flowmask>0).cpu().numpy()#.astype(np.uint8)
        flowmask_pic[flowmask]=np.array([0,0,255])
    flowmask_pic=cv2.resize(flowmask_pic,(1024,1024))
    if occmask is not None:
        occmask=torch.sum(occmask,dim=-1)[0,...]
        occmask=(occmask>0).cpu().numpy()
        occmask_pic[occmask]=np.array([255,0,0])
    occmask_pic=cv2.resize(occmask_pic,(1024,1024))
    bs=occ_gts.shape[0]
    # v_max_thr=torch.max(flow_gts).item()
    occ_gt_vis,flow_gt_vis,flow_gt_vis_mthr=vis_occ(occ_gts[0],flow_gts[0])
    occ_preds_vis,flow_pred_vis,flow_pred_vis_mthr=vis_occ(occ_preds[0],flow_preds[0],v_max_thr=V_MAX_THR)
    # row1=np.concatenate([occ_gt_vis,flow_gt_vis,flowmask_pic],axis=1)
    # row2=np.concatenate([occ_preds_vis,flow_pred_vis,flowmask_pic],axis=1)
    row3=np.concatenate([occmask_pic,occ_gt_vis,flow_gt_vis_mthr,flowmask_pic],axis=1)
    row4=np.concatenate([occmask_pic,occ_preds_vis,flow_pred_vis_mthr,flowmask_pic],axis=1)
    # if row2.shape!=(1024,4096,3) or row1.shape!=(1024,4096,3):
    #     print("!!incorrect vis shape:",idx,"-",row1.shape,row2.shape)
    final_image = np.concatenate([row3,row4], axis=0)
    mmcv.imwrite(final_image, os.path.join(save_root+time_str,"%d_0.jpg" % idx))
    # mmcv.imwrite(row1, os.path.join(save_root+time_str,"%d_gt.jpg" % idx))
    # mmcv.imwrite(row2, os.path.join(save_root+time_str,"%d_pred.jpg" % idx))
    pass   


def vis_mask3d(occ_gt,mask,save_idx=0,pred_occ=None,
               pred_flow=None,save_root='vis/vis3d'):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    X,Y,Z=200,200,16
    voxel_size=0.4
    indices = np.indices((X, Y, Z))#[3,x,y,z]
    indices=np.transpose(indices,(1,2,3,0))
    indices=indices*voxel_size
    outsave=f'{save_idx}_semgt_masknonfree.txt'
    gtnonfree=(occ_gt!=16).cpu().numpy()
    prnonfree=(pred_occ!=16).cpu().numpy()
    mask=mask.cpu().numpy()
    results = np.hstack((indices[mask*gtnonfree], occ_gt.cpu().numpy()[mask*gtnonfree][:, np.newaxis]))
    np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
    outsave=f'{save_idx}_semgt_mask.txt'
    results = np.hstack((indices[mask], occ_gt.cpu().numpy()[mask][:, np.newaxis]))
    np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
    outsave=f'{save_idx}_sempr_nonfree.txt'
    results = np.hstack((indices[prnonfree], pred_occ.cpu().numpy()[prnonfree][:, np.newaxis]))
    np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
    outsave=f'{save_idx}_semgt_nonfree.txt'
    results = np.hstack((indices[gtnonfree], occ_gt.cpu().numpy()[gtnonfree][:, np.newaxis]))
    np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
    # outsave=f'{save_idx}_flowpr_nonfree.txt'
    # results = np.hstack((indices[nonfree], pred_flow.cpu().numpy()[nonfree][:, np.newaxis]))
    # np.savetxt(os.path.join(save_root,outsave),results,fmt='%.2f',delimiter=',', header='x,y,z,value', comments='')
    
    
def vis_flow_gt(data_infos,savevis_root="vis/flowgt/"):
    #可视化3d和2dflow
    max_v=3
    for i,data_info in enumerate(data_infos):
        occ_gt = dict(np.load(data_info['occv2_path']+'/labels.npz', allow_pickle=True))
        flow3d=occ_gt['flow']
        flow2d=occ_gt['flow2d']
        semantics=occ_gt['semantics']
        flow3d_sq=np.linalg.norm(flow3d, axis=-1)
        W,H,Z,D=flow3d.shape
        free=(semantics==16)#0~16类别号
        # d=np.arange(Z).repeat(H,W,1)*(~free)
        # selected = np.argmax(d, axis=-1)#最高点序号？
        flow3d_v=np.linalg.norm(flow3d,axis=-1)
        flow3d_bev_v=np.max(flow3d_v,axis=-1)
        flow3d_bev_x=np.max(flow3d[...,0],axis=-1)
        flow3d_bev_y=np.max(flow3d[...,1],axis=-1)
        flow2d_bev_v=np.sqrt(flow2d[...,0]**2,flow2d[...,1]**2)
        flow2d_bev_x=flow2d[...,0]
        flow2d_bev_y=flow2d[...,1]
        flows=[]
        channel_change=np.array([0,0,1])[None,None,:]
        for v2d,v3d in zip([flow2d_bev_x,flow2d_bev_y,flow2d_bev_v],[flow3d_bev_x,
                flow3d_bev_y,flow3d_bev_v]):
            max_v=np.max(v3d)
            v2d=v2d[...,None]/max_v*255*channel_change
            v3d=v3d[...,None]/max_v*255*channel_change
            flows.append(cv2.resize(v2d,(1024,1024)))
            flows.append(cv2.resize(v3d,(1024,1024)))
        flow2ds=np.concatenate(flows[::2],axis=1)
        flow3ds=np.concatenate(flows[1::2],axis=1)
        flowspic=np.concatenate([flow2ds,flow3ds],axis=0)
        mmcv.imwrite(flowspic,savevis_root+str(i)+".png")

def vis_gt_txt(npz_file,save_root='vis/vis3d/gt'):
    #将单一文件转为txt 用于cloudcompare可视化
    occ_data=np.load(npz_file)
    occgt=occ_data['semantics']
    vismask=occ_data['vismask']

def vis_gt_txts(infos,save_root='vis/vis3d/gt'):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for info in infos:
        npzfile=os.path.join(info['occv2_path'],'labels.npz')
        vis_gt_txt(npzfile,save_root)

if __name__=="__main__":
    pkl_file="data/nuscenes/bevdetv3-nuscenes_infos_train.pkl"
    savevis_root="vis/flowgt/"
    data=mmcv.load(pkl_file)
    data_infos = data['infos']

    # vis_flow_gt(data_infos)

            



