"""Various point-cloud oriented utilities to facilitate the HW.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

from torch.utils.data import Dataset

class PointcloudDataset(Dataset):
    def __init__(self, pointclouds, part_masks, model_names):
        """ Constructor.
        :param pointclouds: iterable of N point-clouds, each being K x 3 points (floats). Typically, this is
         a numpy-array (or a list of size N).
        :param part_masks: part-labels for each provided point of each pointcloud. Assumes same order as
        `pointlouds`.
        :param model_names: iterable of size N, with strings indicating the names/IDs of the provided
        point-clouds.
        """
        super(PointcloudDataset, self).__init__()
        self.pointclouds = pointclouds
        self.part_masks = part_masks
        self.model_names = model_names

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, index):
        pcs = self.pointclouds[index]
        part_masks = self.part_masks[index]
        model_names = self.model_names[index]

        return {'point_cloud': pcs,
                'part_mask': part_masks,
                'model_name': model_names,
                'index': index}