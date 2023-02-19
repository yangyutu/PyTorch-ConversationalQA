from torch import Tensor, device
import gzip
import os
import csv

from pytorch_lightning.callbacks import Callback


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
