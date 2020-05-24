"""
Various simple utilities to facilitate the HW.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
import numpy as np
import os.path as osp

from torch.utils.data import DataLoader
from .pointcloud_dataset import PointcloudDataset


def make_data_loaders(top_data_dir, batch_size, use_small_data=True):
    """Make pytorch dataloaders for HW4.
    :param top_data_dir: (string), directory containing the pointcloud related data (.npz)
    :param batch_size: int size of each batch
    :param use_small_data: boolean, Students use True.
    :return: a dictionary for train/test/val keys to corresponding dataloaders for a in_out/PointcloudDataset.
    """
    data_loaders = dict()
    splits = ['train', 'test', 'val']
    for split in splits:
        if use_small_data:
            raw_data = np.load(osp.join(top_data_dir, split + '_data_small.npz'), allow_pickle=True)
        else:
            raw_data = np.load(osp.join(top_data_dir, split + '_data.npz'), allow_pickle=True)

        pcs = raw_data['pcs']
        part_masks = raw_data['part_masks'].astype(np.long)
        model_names = raw_data['model_names']
        data = PointcloudDataset(pcs, part_masks, model_names)
        data_loaders[split] = DataLoader(data, batch_size=batch_size, shuffle=split=='train')
    return data_loaders


class AverageMeter(object):
    """Computes and stores the average/sum and current value of some quantity of interest.
    Typically this is used to log the loss value of different batches that make an epoch.

    Example:
        loss_meter = AverageMeter()
        for batch in ... :
            batch_loss = ...  # average loss for examples of current batch.
            loss_meter.update(batch_loss, len(batch))
        print(loss_meter.avg)
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        :param val: primitive python type (int, float, boolean)
        :param n: (int, default 1) if val is the result of a computation (e.g., average of batch) of multiple
        items, then n, should reflect the number of those items.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save any number of torch items that have a state_dict property.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load any number of torch items from a checkpoint holding their (saved)
    state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch
