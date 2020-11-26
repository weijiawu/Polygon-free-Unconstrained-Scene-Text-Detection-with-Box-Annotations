"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .st800k import TextSegmentation
from .TextSegmentation_total import TextSegmentation_total
from .Curved_Synthtext_attention import TextSegmentation_attention
datasets = {
    'st800k': TextSegmentation,
    'st800k_total': TextSegmentation_total,
    'st800k_attention': TextSegmentation_attention

}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
