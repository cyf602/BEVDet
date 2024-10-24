# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .builder import DATASETS


@DATASETS.register_module()
class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()#~7723*16
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):#全是0？
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {#各类出现的sample占总出现次数的比例 sum=1
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / max(v,1e-6) for v in class_distribution.values()]#sum>1 ~13
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):#每个类出现的sampleid
            sample_indices += np.random.choice(cls_inds,#在每类出现样本中选取一定比例的sample组成新数据
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)#此处改变了epoch的样本数
