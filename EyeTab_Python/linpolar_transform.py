import numpy as np

from math import ceil, pi, cos, sin

### Adapted from original with licence: ###
'''
Copyright (c) Helio Perroni Filho <xperroni@gmail.com>

This file is part of  Python Log Polar Transform Redux (PLPTR).

PLPTR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PLPTR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PLPTR. If not, see <http://www.gnu.org/licenses/>.
'''

_transforms = {}

def _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s):
    
    # Checks if this transform has been requested before.
    transform = _transforms.get((i_0, j_0, i_n, j_n, p_n, t_n))

    # If the transform is not found...
    if transform == None:
        i_k = []
        j_k = []
        p_k = []
        t_k = []

        # Scans the transform across its coordinate axes. At each step
        # calculates the reverse transform back into the cartesian coordinate
        # system, and if the coordinates fall within the boundaries of the
        # input image, records both coordinate sets.
        for p in range(0, p_n):
            for t in range(0, t_n):
                t_rad = t * t_s

                i = int(i_0 + p * sin(t_rad))
                j = int(j_0 + p * cos(t_rad))

                if 0 <= i < i_n and 0 <= j < j_n:
                    i_k.append(i)
                    j_k.append(j)
                    p_k.append(p)
                    t_k.append(t)

        # Creates a set of two "fancy-indices", one for retrieving pixels from
        # the input image, and other for assigning them to the transform.
        transform = ((np.array(p_k), np.array(t_k)), (np.array(i_k), np.array(j_k)))
        _transforms[i_0, j_0, i_n, j_n, p_n, t_n] = transform

    return transform


def linpolar(image, trans_h=None, trans_w=None):

    # Middle of linpolar transform is image center
    i_0, j_0 = image.shape[1] / 2, image.shape[0] / 2

    # Shape of the input image.
    (i_n, j_n) = image.shape[:2]
    
    # The distance d_c from the transform's focus (i_0, j_0) to the image's
    # farthest corner (i_c, j_c). This is used below as the default value for
    # trans_h, and also to calculate the iteration step across the transform's p
    # dimension.
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5
    
    if trans_h == None:
        # The default value to trans_h is defined as the distance d_c.
        trans_h = int(ceil(d_c))
    
    if trans_w == None:
        # The default value to trans_w is defined as the width of the image.
        trans_w = j_n
    
    # The scale factors determine the size of each "step" along the transform.
    # p_s = log(d_c) / trans_h
    p_s = d_c / trans_h
    t_s = 2.0 * pi / trans_w
    
    # Recover the transform fancy index from the cache, creating it if not found.
    (pt, ij) = _get_transform(i_0, j_0, i_n, j_n, trans_h, trans_w, p_s, t_s)

    # The transform's pixels have the same type and depth as the input's.
    transformed = np.zeros((trans_h, trans_w) + image.shape[2:], dtype=image.dtype)

    # Applies the transform to the image via numpy fancy-indexing.
    transformed[pt] = image[ij]
    return transformed
