# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys

from mmdet3d.core.evaluation.seg_metric import IntersectionOverUnion
from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box
from mmdet3d.utils.logger import get_root_logger
sys.path.insert(0, '/root/data/chuyunfeng/bev/BEVDet/')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
import warnings
from mmcv import Config, DictAction,ProgressBar
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import numpy as np
#from mmdet3d.apis import single_gpu_test
# from projects.mmdet3d_plugin.apis.test import single_gpu_test
from mmdet3d.datasets import build_dataset,build_dataloader
# from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet3d.apis.test import multi_gpu_test, onehot_encoding
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
from PIL import Image
import cv2
from nuscenes.utils.data_classes import Box as NuScenesBox
import pyquaternion
from nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
car_img = Image.open('icon/car.png')
car_img_cv = cv2.imread('icon/car.png')
pic=np.zeros((200,200,3))
mapped_class_names= ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
def show_seg(labels, car_img):

    PALETTE = [[255, 255, 255], [220, 20, 60], [0, 0, 128], [0, 100, 0],
               [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
               [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64]]
    mask_colors = np.array(PALETTE)
    img = np.zeros((200, 400, 3))

    for index, mask_ in enumerate(labels):
        color_mask = mask_colors[index]
        mask_ = mask_.astype(bool)
        img[mask_] = color_mask

    # 这里需要水平翻转，因为这样才可以保证与在图像坐标系下，与习惯相同

    img = np.flip(img, axis=0)
    # 可视化小车
    car_img = np.where(car_img == [0, 0, 0], [255, 255, 255], car_img)[16: 84, 5:, :]
    car_img = cv2.resize(car_img.astype(np.uint8), (30, 16))
    img[img.shape[0] // 2 - 8: img.shape[0] // 2 + 8, img.shape[1] // 2 - 15: img.shape[1] // 2 + 15, :] = car_img

    return img

def format_one_bbox(result,data,sample_id):
    boxes = result['boxes_3d'].tensor.numpy()#det['boxes_3d'].tensor.numpy()
    scores = result['scores_3d'].numpy()
    labels = result['labels_3d'].numpy()
    sample_token = data[sample_id]['token']

    trans = data[sample_id]['cams'][
        'CAM_FRONT']['ego2global_translation']
    rot = data[sample_id]['cams'][
        'CAM_FRONT']['ego2global_rotation']
    rot = pyquaternion.Quaternion(rot)
    annos = list()
    for i, box in enumerate(boxes):
        name = mapped_class_names[labels[i]]
        center = box[:3]
        wlh = box[[4, 3, 5]]
        box_yaw = box[6]
        box_vel = box[7:].tolist()
        box_vel.append(0)
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
        nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
        nusc_box.rotate(rot)
        nusc_box.translate(trans)
        if np.sqrt(nusc_box.velocity[0]**2 +
                    nusc_box.velocity[1]**2) > 0.2:
            if name in [
                    'car',
                    'construction_vehicle',
                    'bus',
                    'truck',
                    'trailer',
            ]:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = DefaultAttribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = DefaultAttribute[name]
        nusc_anno = dict(
            sample_token=sample_token,
            translation=nusc_box.center.tolist(),
            size=nusc_box.wlh.tolist(),
            rotation=nusc_box.orientation.elements.tolist(),
            velocity=nusc_box.velocity[:2],
            detection_name=name,
            detection_score=float(scores[i]),
            attribute_name=attr,
        )
    pass

def single_gpu_vis(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    cv2_style=True,
                    show_mask_gt = True
                    ):
    map_enable = True
    logger = get_root_logger()

    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result_data=[]
            if False and 'seg_preds' in result[0].keys() and result[0]['seg_preds'] is not None:
                semantic = result[0]['seg_preds']
                semantic = onehot_encoding(semantic).cpu().numpy()
                # 使用cv2进行可视化
                imname = f'{out_dir}/{data["img_metas"][0].data[0][0]["sample_idx"]}.png'
                logger.info(f'saving: {imname}')
                cv2.imwrite(imname, show_seg(semantic.squeeze(), car_img_cv))

                if show_mask_gt:
                    target_semantic_indices = data['semantic_indices'][0].unsqueeze(0)
                    one_hot = target_semantic_indices.new_full(semantic.shape, 0)
                    one_hot.scatter_(1, target_semantic_indices, 1)
                    semantic = one_hot.cpu().numpy().astype(np.float)
                    imname = f'{out_dir}/{data["img_metas"][0].data[0][0]["sample_idx"]}_gt.png'
                    logger.info(f'saving: {imname}')
                    cv2.imwrite(imname, show_seg(semantic.squeeze(), car_img_cv))
            if result[0]['pts_bbox'] is not None:
                results.append([result[0]])
                # nusc_boxes=format_one_bbox(result[0]['pts_bbox'],data,i)
                # for ind, cam in enumerate(cams):
                #     sample_data_token = sample['data'][cam]

                #     # if sensor_modality in ['lidar', 'radar']:
                #     #     assert False
                #     # elif sensor_modality == 'camera':
                #     # Load boxes and image.
                #     boxes = [NuScenesBox(record['translation'], record['size'], pyquaternion.Quaternion(record['rotation']),
                #                 name=record['detection_name'], token='predicted') for record in
                #             pred_data['results'][sample_toekn] if record['detection_score'] > 0.3]

                #     data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                #                                                                 box_vis_level=box_vis_level, pred_anns=boxes)
                #     _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
                #     data = cv2.imread(data_path)

                #     # Show boxes.
                #     if with_anns:
                #         for box in boxes_pred:
                #             c = get_color(box.name)
                #             box.render_cv2(data, view=camera_intrinsic, normalize=True, colors=(c, c, c))
                #         result_data.append(data)

            # compose result data
            # first_row = result_data[:3]
            # second_row = result_data[3:]
            # np.hstack(first_row)
            # cam_img = np.vstack((np.hstack(first_row), np.hstack(second_row)))
        prog_bar.update()    
        
    # print("L100")
    return results
                
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default='projects/configs/bevformer/bevformer_small_seg_det.py', help='test config file path')
    parser.add_argument('--checkpoint', default='ckpts/epoch_18.pth', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        #nargs='+',
        default=True,
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', default='/root/data/chuyunfeng/BEVFormer_segmentation_detection/work_dirs/show_dir', help='directory where results will be saved')
    #parser.add_argument('--show-dir', default=None, help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # dataset.data_infos=dataset.data_infos[:100]
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        # nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        # assert False
        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        assert False
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        #outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)
    outputs=single_gpu_vis(model, data_loader, args.show, args.show_dir)
    rank, _ = get_dist_info()
    if rank == 0:
    #     if args.out:
    #         print(f'\nwriting results to {args.out}')
    #         assert False
    #         #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        # if args.format_only:
        result_files, tmp_dir=data_loader.dataset.format_results(outputs, **kwargs)
        print(result_files,tmp_dir)#json路径，None

    #     if args.eval:
    #         eval_kwargs = cfg.get('evaluation', {}).copy()
    #         print(eval_kwargs)
    #         print('.................')
    #         # hard-code way to remove EvalHook args
    #         for key in [
    #                 'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
    #                 'rule'
    #         ]:
    #             eval_kwargs.pop(key, None)
    #         eval_kwargs.update(dict(metric=args.eval, **kwargs))
    #         print(eval_kwargs)
    #         print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
