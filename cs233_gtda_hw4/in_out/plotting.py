"""
Various plotting utilities to facilitate the HW.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image


def plot_3d_point_cloud(pc, show=True, show_axis=True, in_u_sphere=True, marker='.', s=8, alpha=.8,
                        figsize=(5, 5), elev=10, azim=45, axis=None, title=None, *args, **kwargs):
    """Plot a 3d point-cloud via matplotlib.
    :param pc: N x 3 numpy array storing the x, y, z coordinates of a 3D pointcloud with N points.

    Students Note you can use the default other parameters, or explore their effect for better visualizations.
    The **kwargs are used by matplotlib's scatter, check the "c" (color) parameter when you play with the
    part-predictions.
    """
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def transparent_png_to_rgb_via_pil(filename, background_color='white'):
    """Helper function to load the .png images of the models.
    :param filename: the .png filename
    :param background_color: color of background of resulting RBG image.
    :return: an RGB version of the loaded png.
    """
    png = Image.open(filename)
    background = Image.new("RGB", png.size, background_color)
    background.paste(png, mask=png.split()[3])
    return background


def plot_2d_embedding_in_grid_greedy_way(two_dim_emb, image_files, big_dim=2500, small_dim=200,
                                         save_file=None, transparent_pngs=False, background='white'):
    """ Use this to plot/paste images on a large canvas according to a 2D embedding. E.g. according to the TSNE
    vectors corresponding to the (models) represented by the images.
    Input:
        two_dim_emb: (N x 2) numpy array: arbitrary 2D embedding of data.
        image_files: (list) of strings pointing to images on the hard drive. Specifically image_files[i] should be an
            image associated with the datum whose coordinates are given in two_dim_emb[i].
        big_dim:     (int) height of output 'big' rectangular grid/canvas image.
        small_dim:   (int) height to which each individual 'small' image/thumbnail will be resized before is put on the
            canvas.
        transparent_pngs: (boolean, default False) are the thumbnail images on the drive .pngs with an alpha?
        background_color: color of background of resulting canvas image.
    """
    ceil = np.ceil
    mod = np.mod


    def _scale_2d_embedding(two_dim_emb):
        two_dim_emb -= np.min(two_dim_emb, axis=0)  # scale x-y in [0,1]
        two_dim_emb /= np.max(two_dim_emb, axis=0)
        return two_dim_emb

    x = _scale_2d_embedding(two_dim_emb)
    out_image = np.array(Image.new("RGB", (big_dim, big_dim),  background))

    if transparent_pngs:
        im_loader = transparent_png_to_rgb_via_pil
    else:
        im_loader = lambda x: Image.open(x).convert('RGB')

    occupied = set()
    for i, im_file in enumerate(image_files):
        #  Determine location on grid
        a = ceil(x[i, 0] * (big_dim - small_dim) + 1)
        b = ceil(x[i, 1] * (big_dim - small_dim) + 1)
        a = int(a - mod(a - 1, small_dim) - 1)
        b = int(b - mod(b - 1, small_dim) - 1)

        if (a, b) in occupied:
            continue    # Spot already filled (drop=>greedy).
        else:
           occupied.add((a, b))

        fig = im_loader(im_file)
        fig = fig.resize((small_dim,small_dim), Image.LANCZOS)

        try:
            out_image[a:a + small_dim, b:b + small_dim, :] = fig
        except:
            pass

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)

    return out_image