# Copyright (c) OpenMMLab. All rights reserved.
import json
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs
from mmdet.apis.test import collect_results_cpu
import time
from mmcv.runner import get_dist_info
from mmdet3d.core.evaluation.seg_metric import IntersectionOverUnion
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from mmdet3d.utils.logger import get_root_logger


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def onehot_encoding(logits, dim=1):
    if len(logits.shape)==3:
        logits=logits.unsqueeze(0)
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    logger = get_root_logger()
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    map_enable = True
    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()
        
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results [{'pts_bbox'}]
            # if isinstance(result[0], tuple):
            #     result = [(bbox_results, encode_mask_results(mask_results))
            #               for bbox_results, mask_results in result]
            # # This logic is only used in panoptic segmentation test.
            # elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            #     for j in range(len(result)):
            #         bbox_results, mask_results = result[j]['ins_results']
            #         result[j]['ins_results'] = (
            #             bbox_results, encode_mask_results(mask_results))
        if 'seg_preds' in result[0].keys():
            if  result[0]['seg_preds'] == None:
                map_enable = False
        else:
            map_enable = False
        
        
        if map_enable:
            pred = result[0]['seg_preds']
            pred = onehot_encoding(pred)
            num_cls = pred.shape[1]
            indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)
            pred_semantic_indices = torch.sum(pred * indices, axis=1).int()#转类别编号
            target_semantic_indices = data['semantic_indices'][0].cuda()

            semantic_map_iou_val(pred_semantic_indices,
                                 target_semantic_indices)
        results.append(result)#??

        if rank == 0:
            batch_size = 1#len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    if map_enable:
        import prettytable as pt
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        if rank == 0:
            tb = pt.PrettyTable()
            tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
            tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                        round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                        round(mIoU.cpu().numpy().item(), 4)])
            print('\n')
            #print(tb)
            logger.info(tb)

            seg_dict = dict(
                Validation_num=len(dataset),
                Divider=round(scores[1:].cpu().numpy()[0], 4),
                Pred_Crossing=round(scores[1:].cpu().numpy()[1], 4),
                Boundary=round(scores[1:].cpu().numpy()[2], 4),
                mIoU=round(mIoU.cpu().numpy().item(), 4)
            )
            print("seg_evaluate results:",seg_dict)
            with open('segmentation_result.json', 'a') as f:
                f.write(json.dumps(str(seg_dict)) + '\n')
                
    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)