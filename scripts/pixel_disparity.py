from __future__ import division, print_function
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from disparity import block_matching

parser = argparse.ArgumentParser()
parser.add_argument('--num_disparities', default=16, type=int)
parser.add_argument('--block_size', default=5, type=int)
ARGS = parser.parse_args()


def main():
    # load left and right images
    image_left = Image.open('../data/test/tsukuba_L.png').convert('L')
    image_right = Image.open('../data/test/tsukuba_R.png').convert('L')
    # convert to float, range 0-1
    image_left = np.asarray(image_left, dtype=np.float32) / 255.
    image_right = np.asarray(image_right, dtype=np.float32) / 255.

    # compute binocular disparity in pixel space
    disparity = block_matching.compute_disparity(
        image_left, image_right, numDisparities=ARGS.num_disparities,
        blockSize=ARGS.block_size
    )

    # visualize
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(image_right, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('original image')
    axes[1].imshow(disparity)
    axes[1].axis('off')
    axes[1].set_title('disparity map')
    plt.show()

if __name__ == '__main__':
    main()