from .ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
chamfer_raw = dist_chamfer_3D.chamfer_3DDist()

def chamfer_loss(pc_a, pc_b):
    """ Compute the chamfer loss for batched pointclouds.
    :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
    :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
    :return: B floats, indicating the chamfer distances
    """
    dist_a, dist_b, idx_a, idx_b = chamfer_raw(pc_a, pc_b)
    dist = dist_a.mean(1) + dist_b.mean(1) # reduce separately, sizes of points can be different
    return dist