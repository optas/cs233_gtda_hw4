"""
Part-Aware PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from ..in_out.utils import AverageMeter
from ..losses.chamfer import chamfer_loss

# In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
# one that is written in pure pytorch:
# from ..losses.nn_distance import chamfer_loss (uncomment)

class PartAwarePointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, part_classifier):
        """ Part-aware AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        :param part_classifier: nn.Module acting as the second decoding branch that classifies the point part
        labels.
        """
        super(PartAwarePointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.part_classifier = part_classifier

