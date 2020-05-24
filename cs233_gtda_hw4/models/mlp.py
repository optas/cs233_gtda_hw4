"""
Multi-layer perceptron.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""


from torch import nn


class MLP(nn.Module):
    """ Multi-layer perceptron. That is a k-layer deep network where each layer is a fully-connected layer, with
    (optionally) batch-norm, a non-linearity and dropout.

    Students: again, you can use this scaffold to make a generic MLP that can be used with multiple-hyper parameters
    or, opt for a perhaps simpler custom variant that just does so for HW4. For HW4 do not use batch-norm, drop-out
    or other non-requested features, for the non-bonus question.
    """

    def __init__(self, in_feat_dims, out_channels, b_norm=False, dropout_rate=0, non_linearity=nn.ReLU(inplace=True)):
        """Constructor
        :param in_feat_dims: input feature dimensions
        :param out_channels: list of ints describing each the number hidden/final neurons.
        :param b_norm: True/False, or list of booleans
        :param dropout_rate: int, or list of int values
        :param non_linearity: nn.Module
        """
        super(MLP, self).__init__()
        raise NotImplementedError
