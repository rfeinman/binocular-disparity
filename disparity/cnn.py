from __future__ import division, print_function
from PIL import Image
import numpy as np
from keras import models
from keras.applications import vgg16

from .util import parallel, shift, normalize_features

vgg_feats = None

def disparity_energy(d_val):
    feats_Ls = shift(feats_L, d_val, 'left')
    energy = np.linalg.norm(feats_R-feats_Ls, axis=2)

    return energy

def disparity_energy1(d_val):
    energy = np.linalg.norm(feats[0]-feats[d_val+1], axis=2)

    return energy

def interpolate_energy(energy):
    energy = Image.fromarray(energy)
    energy = energy.resize((width, height), Image.BILINEAR)
    energy = np.array(energy)

    return energy

def check_imgs(img_L, img_R):
    assert len(img_L.shape) == 3 and len(img_R.shape) == 3
    assert img_L.dtype == np.uint8 and img_R.dtype == np.uint8
    assert img_L.max() > 1 and img_R.max() > 1

def compute_energies(
        img_L, img_R, numDisparities, shift_mode='before',
        normalize_feats=False, interpolate=True
):
    # check inputs
    check_imgs(img_L, img_R)
    global height, width
    height, width, _ = img_L.shape
    imgs = [img_R, img_L]
    if shift_mode == 'before':
        global feats
        # get shifted images
        for i in range(numDisparities - 1):
            img_Ls = shift(img_L, i, 'left')
            imgs.append(img_Ls)
        imgs = np.stack(imgs, axis=0)
        # extract cnn features
        feats = cnn_features(imgs, normalize_feats)
        energies = parallel(disparity_energy1, range(numDisparities))
    elif shift_mode == 'after':
        global feats_R, feats_L
        # extract cnn features
        imgs = np.stack(imgs, axis=0)
        feats_R, feats_L = cnn_features(imgs, normalize_feats)
        # compute disparity energies
        energies = parallel(disparity_energy, range(numDisparities))
    else:
        raise Exception
    # perform bi-linear interpolation if needed
    if energies[0].shape != (height, width) and interpolate:
        energies = parallel(interpolate_energy, energies)
    # finalize energies array
    energies = np.asarray(energies, dtype=np.float32)
    energies = np.transpose(energies, axes=[1,2,0])

    return energies

def cnn_features(imgs, normalize=False):
    assert len(imgs.shape) == 4
    assert imgs.dtype == np.uint8
    assert imgs.max() > 1
    global vgg_feats
    if vgg_feats is None:
        vgg_feats = get_vgg()
    # pre-process images for vgg
    imgs = vgg16.preprocess_input(imgs)
    # compute vgg features
    feats = vgg_feats.predict(imgs)
    # normalize if needed
    if normalize:
        feats = normalize_features(feats)

    return feats

def get_vgg(layer='block2_conv2'):
    model = vgg16.VGG16(weights='imagenet', include_top=False)
    assert layer in [l.name for l in model.layers]
    model_feats = models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer).output
    )

    return model_feats