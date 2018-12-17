from __future__ import division, print_function
import argparse
import sys
import os
import shutil
import csv
import cv2
import numpy as np

from disparity import util, data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/middlebury', type=str)
parser.add_argument('--results_dir', default='../results/block_matching', type=str)
parser.add_argument('--nb_samples', default=None, type=int)
ARGS = parser.parse_args()


def rgb2gray(im):
    assert len(im.shape) == 3
    gray = 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]
    gray = np.round(gray).astype('uint8')
    return gray

def BM_disparity(img_L, img_R, numDisparities, blockSize):
    matcher = cv2.StereoBM_create(numDisparities, blockSize)
    disparity = matcher.compute(rgb2gray(img_L), rgb2gray(img_R))
    disparity = disparity.astype(np.float32) / 16.

    return disparity.astype(np.int64)

def write(results):
    fname = os.path.join(ARGS.results_dir, "results.csv")
    with open(fname, mode='a') as f:
        scores_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        scores_writer.writerow(results)

def check_overwrite():
    val = input("Results directory '%s' already exists. Would you like to "
                "overwrite? [y/n]: " % ARGS.results_dir)
    if val in ['y', 'n']:
        return val
    else:
        print("Must enter 'y' or 'n'.")
        return check_overwrite()

def main():

    # initialize results directory
    if os.path.isdir(ARGS.results_dir):
        if check_overwrite() == 'n':
            sys.exit(0)
        shutil.rmtree(ARGS.results_dir)
    os.mkdir(ARGS.results_dir)

    # load the middlebury dataset
    print('Loading image data...')
    samples = data.load_middlebury_dataset(
        ARGS.data_dir, nb_samples=ARGS.nb_samples, max_size=400
    )

    # add header to results csv
    write(['spearman_BM','pearson_BM'])

    # loop through samples
    for i in range(len(samples)):
        print('Processing image #%i' % i)
        img_L, img_R, _, disp_R = samples[i]
        height, width, _ = img_L.shape

        # baseline block matching disparity
        disparity_BM = BM_disparity(img_L,img_R,numDisparities=128,blockSize=5)
        # compute scores
        spearman = util.score_disparity(disparity_BM, disp_R, mode='spearman')
        pearson = util.score_disparity(disparity_BM, disp_R, mode='pearson')
        print('rho_BM: %0.3f' % spearman)

        # save results
        write([spearman, pearson])
        fname = os.path.join(ARGS.results_dir, 'disp%0.3i_BM.npy'%i)
        np.save(fname, disparity_BM.astype(np.int16))

if __name__ == '__main__':
    main()