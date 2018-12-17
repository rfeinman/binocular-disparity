from __future__ import division, print_function
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from disparity import cnn

parser = argparse.ArgumentParser()
parser.add_argument('--num_disparities', default=16, type=int)
ARGS = parser.parse_args()


def main():
    # load left and right images
    image_left = np.array(Image.open('../data/test/tsukuba_L.png'))
    image_right = np.array(Image.open('../data/test/tsukuba_R.png'))

    # compute binocular disparity using CNN
    disparity = cnn.compute_disparity(
        image_left, image_right, numDisparities=ARGS.num_disparities,
        shift_mode='before'
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