from __future__ import division, print_function
import csv
import os
import shutil
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend as K

from disparity import wrangle_data, CNN, MRF, util

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/middlebury', type=str)
parser.add_argument('--results_dir', default='../results', type=str)
parser.add_argument('--nb_samples', default=None, type=int)
parser.add_argument('--gpu_id', default='', type=str)
ARGS = parser.parse_args()

def save(fname, results):
    with open(fname, mode='a') as f:
        scores_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        scores_writer.writerow(results)

def save_disparity(fname, disparity):
    # save matplotlib figure
    plt.figure()
    plt.imshow(disparity)
    plt.axis('off')
    plt.savefig(fname)
    plt.close()
    # save numpy array
    np.save(fname+'.npy', disparity.astype(np.int16))

def main():
    assert ARGS.gpu_id in ['', '0', '1', '2']
    # set up tf session
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=ARGS.gpu_id)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options)
    )
    K.set_session(sess)

    # definitions
    numDisparities = 120
    bp_num_iters = 20
    scores_file_name = os.path.join(ARGS.results_dir, "results.csv")

    # clear results dir
    if os.path.isdir(ARGS.results_dir):
        warnings.warn('overwriting existing results folder')
        shutil.rmtree(ARGS.results_dir)
    os.mkdir(ARGS.results_dir)

    # load the middlebury data
    samples = wrangle_data.load_middlebury_dataset(
        ARGS.data_dir, nb_samples=ARGS.nb_samples, max_size=400
    )

    # add header to results csv
    save(
        scores_file_name,
        ['thresh', 'spearman_CNN', 'pearson_CNN', 'spearman_CRF', 'pearson_CRF']
    )

    # loop through the samples and perform inference
    for samp_ix in range(len(samples)):
        print('Sample #%i' % samp_ix)
        # get params for this iteration
        img_L, img_R, disp_L, disp_R = samples[samp_ix]
        height, width, _ = img_L.shape

        # compute disparity energies
        print('Computing disparity energies...')
        energies = CNN.compute_energies(
            img_L, img_R, numDisparities=numDisparities
        )

        # compute disparity threshold
        print('Computing disparity threshold...')
        num_beliefs = util.select_disparity_threshold(energies)
        print('Best disparity threshold: %i' % num_beliefs)
        energies = energies[:,:,:num_beliefs]

        # compute disparity
        disparity_CNN = np.argmin(energies, axis=2)
        fname = os.path.join(ARGS.results_dir, 'disp%0.3i_CNN'%samp_ix)
        save_disparity(fname, disparity_CNN)

        # compute initial scores
        init_disp_pearson = util.score_disparity(disparity_CNN, disp_R, mode='pearson')
        init_disp_spearman = util.score_disparity(disparity_CNN, disp_R, mode='spearman')

        # initialize MRF
        mrf = MRF.LoopyBP(height, width, num_beliefs)
        disparity_MRF = mrf.decode_MAP(energies, iterations=bp_num_iters)

        fname = os.path.join(ARGS.results_dir, 'disp%0.3i_CRF' % samp_ix)
        save_disparity(fname, disparity_MRF)

        # compute final scores
        fin_disp_pearson = util.score_disparity(disparity_MRF, disp_R, mode='pearson')
        fin_disp_spearman = util.score_disparity(disparity_MRF, disp_R, mode='spearman')

        # save iteration results to CSV
        save(
            fname=scores_file_name,
            results=[num_beliefs, init_disp_spearman,
                     init_disp_pearson, fin_disp_spearman, fin_disp_pearson]
        )

if __name__== "__main__":
    main()