from __future__ import division, print_function
import numpy as np

from .util import parallel, shift


# Distance functions

def SAD(block_R, block_L):
    diff = np.sum(np.abs(block_R.reshape(-1) - block_L.reshape(-1)))

    return diff

def euclidean_SAD(block_R, block_L):
    distance = np.linalg.norm(block_R - block_L, axis=2)
    diff = np.sum(distance)

    return diff


# Block matching core

def disparity_energy(d_val):
    left_s = shift(left, d_val, 'left')
    if len(right.shape) == 2:
        pad = bsize//2
    else:
        pad = ((bsize//2,bsize//2),(bsize//2,bsize//2),(0,0))
    right_pad = np.pad(right, pad, mode='constant')
    left_s_pad = np.pad(left_s, pad, mode='constant')
    energy = np.zeros(shape=(height, width), dtype=np.float32)
    for h in range(height):
        for w in range(width):
            block_R = right_pad[h:(h + bsize), w:(w + bsize)]
            block_L = left_s_pad[h:(h + bsize), w:(w + bsize)]
            energy[h, w] = diff_fn(block_R, block_L)

    return energy

def compute_energies(img_L, img_R, numDisparities, blockSize):
    assert img_L.shape == img_R.shape
    global left, right, bsize, diff_fn, height, width
    left, right, bsize = img_L, img_R, blockSize
    height, width = img_L.shape[:2]
    if len(left.shape) == 2:
        diff_fn = SAD
    elif len(left.shape) == 3:
        diff_fn = euclidean_SAD
    else:
        raise Exception
    energies = parallel(disparity_energy, range(numDisparities))
    energies = np.asarray(energies, dtype=np.float32)
    # convert (n,H,W) -> (H,W,n)
    energies = np.transpose(energies, axes=[1,2,0])

    return energies

def compute_disparity(img_L, img_R, numDisparities, blockSize):
    energies = compute_energies(img_L, img_R, numDisparities, blockSize)
    disparity = np.argmin(energies, axis=2)

    return disparity
