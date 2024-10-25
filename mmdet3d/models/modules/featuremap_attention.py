import torch
from mmdet.models.utils.builder import TRANSFORMER

@TRANSFORMER.register_module()
class FeatureMapAttention:
    def __init__(self) -> None:
        pass