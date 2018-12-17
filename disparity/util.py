from __future__ import division, print_function
import warnings
import multiprocessing as mp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def parallel(f, x):
    p = mp.Pool()
    y = p.map(f, x)
    p.close()
    p.join()

    return y

def shift(img, k, mode='left'):
    img = img.copy()
    assert type(k) == int
    if k == 0:
        pass
    elif mode == 'right':
        img = np.roll(img, k, axis=1)
        img[:,:k] = 0
    elif mode == 'left':
        img = np.roll(img, -k, axis=1)
        img[:,-k:] = 0
    else:
        raise Exception("mode must be 'right' or 'left'")

    return img

def normalize_features(feats):
    assert len(feats.shape) == 4
    N, H, W, F = feats.shape
    feats = feats.copy().reshape(N*H*W,F)
    feats -= feats.mean(axis=0)
    feats /= feats.std(axis=0)
    feats = feats.reshape(N,H,W,F)

    return feats

def score_disparity(disparity, ground_truth, mode='spearman'):
    assert disparity.shape == ground_truth.shape
    assert len(disparity.shape) == 2
    if mode == 'spearman':
        rho, p = stats.spearmanr(disparity.flatten(), ground_truth.flatten())
    elif mode == 'pearson':
        rho, p = stats.pearsonr(disparity.flatten(), ground_truth.flatten())
    else:
        raise Exception
    if p > 0.05:
        warnings.warn('correlation p-value=%0.2f is greater than 0.05' % p)

    return rho

def show_result(img, d, gt, numDisparities=None, score_mode='spearman'):
    rho = score_disparity(d, gt, score_mode)
    fig, axes = plt.subplots(1, 3, figsize=(15,3))
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('image')
    if numDisparities is None:
        axes[1].imshow(d, vmin=0)
    else:
        axes[1].imshow(d, vmin=0, vmax=numDisparities)
    axes[1].axis('off')
    axes[1].set_title('prediction (rho=%0.3f)' % rho)
    axes[2].imshow(gt)
    axes[2].axis('off')
    axes[2].set_title('ground truth')

def softmax(X, theta=1.0, axis=None):
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p

def energies_to_probs(energies):
    # z-score
    energies = energies - energies.mean()
    energies = energies / energies.std()
    # compute softmax
    probs = softmax(-energies, axis=2)

    return probs

def energy_entropy(energies):
    assert len(energies.shape) == 3
    n = energies.shape[2]
    probs = energies_to_probs(energies)
    # (h,w,n) -> (n,h,w)
    probs = np.transpose(probs, axes=[2,0,1])
    entropy = stats.entropy(probs)
    # standardize across different cardinalities
    entropy /= stats.entropy(np.ones(n))

    return entropy

# def energy_entropy(energies):
#     h, w, n = energies.shape
#     # -> (h*w,n)
#     energies = energies.reshape(h*w, n)
#     # -> (n,h*w)
#     energies = np.transpose(energies)
#     energies = energies.max(axis=0) - energies
#     probs = softmax(energies, axis=0)
#     entropy = stats.entropy(probs)
#     entropy = entropy.reshape(h, w) / stats.entropy(np.ones(n))
#
#     return entropy

def get_percentile(thresh):
    entropy = energy_entropy(eng[:,:,:thresh])
    p = np.percentile(entropy.flatten(), 75)

    return p

def select_disparity_threshold(energies, min_thresh=10, alpha=0.05):
    assert len(energies.shape) == 3
    assert 0 <= alpha <= 1
    global eng
    eng = energies
    candidates = np.arange(min_thresh, energies.shape[2])
    percentiles = parallel(get_percentile, candidates)
    #thresh = candidates[np.argmin(percentiles)]
    p_min, p_max = np.min(percentiles), np.max(percentiles)
    cap = p_min + alpha*(p_max-p_min)
    ix = np.where(percentiles <= cap)[0]
    thresh = candidates[ix[0]]

    return thresh
