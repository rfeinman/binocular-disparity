from __future__ import division, print_function
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--num_disparities', default=16, type=int)
parser.add_argument('--block_size', default=5, type=int)
ARGS = parser.parse_args()


def main():
    # load left and right images
    image_left = cv2.imread('../data/test/tsukuba_L.png', 0)
    image_right = cv2.imread('../data/test/tsukuba_R.png', 0)

    # compute binocular disparity in pixel space
    matcher = cv2.StereoBM_create(
        numDisparities=ARGS.num_disparities, blockSize=ARGS.block_size
    )
    disparity = matcher.compute(image_left, image_right)
    # convert fixed-point representation to float
    disparity = disparity.astype(np.float32) / 16.

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