from __future__ import division, print_function
import argparse
import warnings
import os
import cv2
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K

from disparity import cnn, mrf, util, data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/middlebury', type=str)
parser.add_argument('--save_dir', default=None, type=str)
parser.add_argument('--nb_samples', default=None, type=int)
parser.add_argument('--shift_mode', default='before', type=str)
parser.add_argument('--gpu_id', default='', type=str)
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

def save_results(
        fname, thresh, spearman_BMs, spearman_CNNs, spearman_CRFs, pearson_BMs,
        pearson_CNNs, pearson_CRFs
):
    df = pd.DataFrame()
    df['thresh'] = thresh
    df['spearman_BM'] = spearman_BMs
    df['spearman_CNN'] = spearman_CNNs
    df['spearman_CRF'] = spearman_CRFs
    df['pearson_BM'] = pearson_BMs
    df['pearson_CNNs'] = pearson_CNNs
    df['pearson_CRFs'] = pearson_CRFs
    df.to_csv(fname, index=False)

def main():
    assert ARGS.gpu_id in ['', '0', '1', '2']
    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=ARGS.gpu_id)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    assert ARGS.shift_mode in ['before', 'after']
    if ARGS.save_dir is None:
        warnings.warn("results wont be saved")
    else:
        if os.path.isdir(ARGS.save_dir):
            warnings.warn('overwriting existing results folder')
            shutil.rmtree(ARGS.save_dir)
        os.mkdir(ARGS.save_dir)
        results_file = os.path.join(ARGS.save_dir, 'results.csv')
    print('Loading image data...')
    samples = data.load_middlebury_dataset(
        ARGS.data_dir, nb_samples=ARGS.nb_samples, max_size=400
    )
    # results placeholders
    spearman_BMs = []
    spearman_CNNs = []
    spearman_CRFs = []
    pearson_BMs = []
    pearson_CNNs = []
    pearson_CRFs = []
    thresh = []
    # loop through samples
    for i in range(len(samples)):
        print('Processing image #%i' % i)
        img_L, img_R, _, disp_R = samples[i]
        height, width, _ = img_L.shape

        # baseline block matching disparity
        disparity_BM = BM_disparity(img_L, img_R, numDisparities=128, blockSize=5)
        # compute scores
        spearman = util.score_disparity(disparity_BM, disp_R, mode='spearman')
        pearson = util.score_disparity(disparity_BM, disp_R, mode='pearson')
        spearman_BMs.append(spearman)
        pearson_BMs.append(pearson)
        print('rho_BM: %0.3f' % spearman)

        print('Computing disparities...')
        # compute disparity energies
        energies = cnn.compute_energies(
            img_L, img_R, numDisparities=120, shift_mode=ARGS.shift_mode
        )
        # select disparity threshold
        print('Selecting best threshold...')
        numDisparities = util.select_disparity_threshold(energies)
        print('best threshold: %i' % numDisparities)
        thresh.append(numDisparities)
        energies = energies[:,:,:numDisparities]
        disparity_CNN = np.argmin(energies, axis=2)
        # compute scores
        spearman = util.score_disparity(disparity_CNN, disp_R, mode='spearman')
        pearson = util.score_disparity(disparity_CNN, disp_R, mode='pearson')
        spearman_CNNs.append(spearman)
        pearson_CNNs.append(pearson)
        print('rho_CNN: %0.3f' % spearman)

        # perform CRF smoothing
        print('Performing CRF smoothing...')
        # perform smoothing
        smoother = mrf.GradientDescent(
            height, width, numDisparities, session=sess, alpha=2., beta=0.2
        )
        disparity_CRF = smoother.decode_MAP(energies, lr=0.01, iterations=100)
        # compute new scores
        spearman = util.score_disparity(disparity_CRF, disp_R, mode='spearman')
        pearson = util.score_disparity(disparity_CRF, disp_R, mode='pearson')
        spearman_CRFs.append(spearman)
        pearson_CRFs.append(pearson)
        print('rho_CRF: %0.3f' % spearman)

        # save results
        if ARGS.save_dir is not None:
            save_results(
                results_file, thresh, spearman_BMs, spearman_CNNs, spearman_CRFs,
                pearson_BMs, pearson_CNNs, pearson_CRFs
            )
            np.save(
                os.path.join(ARGS.save_dir, 'disp%0.3i_BM.npy'%i),
                disparity_BM.astype(np.int16)
            )
            np.save(
                os.path.join(ARGS.save_dir, 'disp%0.3i_CNN.npy'%i),
                disparity_CNN.astype(np.int16)
            )
            np.save(
                os.path.join(ARGS.save_dir, 'disp%0.3i_CRF.npy'%i),
                disparity_CRF.astype(np.int16)
            )

if __name__ == '__main__':
    main()