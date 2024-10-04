import torch
from typing import Optional
from torchmetrics.metric import Metric
from torchmetrics.functional.classification import stat_scores

class IntersectionOverUnion(Metric):
    """Computes intersection-over-union."""

    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = 'none',
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        self.add_state('true_positive', default=torch.zeros(
            n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(
            n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(
            n_classes), dist_reduce_fx='sum')
        self.add_state('support', default=torch.zeros(
            n_classes), dist_reduce_fx='sum')

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        tps, fps, _, fns, sups = stat_scores(
            prediction, target, num_classes=self.n_classes, reduce='macro', mdmc_reduce='global').t()
        self.true_positive += tps
        self.false_positive += fps
        self.false_negative += fns
        self.support += sups

    def compute(self):
        scores = torch.zeros(
            self.n_classes, device=self.true_positive.device, dtype=torch.float32)

        for class_idx in range(self.n_classes):
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            # If this class is absent in the target (no support) AND absent in the pred (no true or false
            # positives), then use the absent_score for this class.
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        # Remove the ignored class index from the scores.
        if (self.ignore_index is not None) and (0 <= self.ignore_index < self.n_classes):
            scores = torch.cat([scores[:self.ignore_index],
                               scores[self.ignore_index+1:]])

        return scores