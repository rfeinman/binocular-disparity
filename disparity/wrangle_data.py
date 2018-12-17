from __future__ import division, print_function
import os
import re
from struct import unpack
import numpy as np
from PIL import Image

from .util import parallel


def read_pfm(file, debug=False):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        _type = f.readline().decode('latin-1')
        if "PF" in _type:
            channels = 3
        elif "Pf" in _type:
            channels = 1
        else:
            print("ERROR: Not a valid PFM file", file=sys.stderr)
            sys.exit(1)
        if (debug):
            print("DEBUG: channels={0}".format(channels))

        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)
        if (debug):
            print("DEBUG: width={0}, height={1}".format(width, height))

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        if (debug):
            print("DEBUG: BigEndian={0}".format(BigEndian))

        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)

        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)

    return img, height, width

def interpolate(vec):
    mask = np.isinf(vec)
    vec[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), vec[~mask]
    )

    return vec

def load_disparity_GT(file):
    # read raw data from pfm file
    D, height, width = read_pfm(file)
    assert len(D) == height*width
    # convert to numpy array
    D = np.asarray(D).reshape(height, width)
    D = D[::-1]
    # fill 'inf' values via interpolation
    for i in range(height):
        D[i] = interpolate(D[i])

    return D

def load_one_sample(folder_name):
    folder = os.path.join(datadir, folder_name)
    img_L = Image.open(os.path.join(folder, 'im0.png'))
    img_R = Image.open(os.path.join(folder, 'im1.png'))
    disp_L = Image.fromarray(
        load_disparity_GT(os.path.join(folder, 'disp0.pfm'))
    )
    disp_R = Image.fromarray(
        load_disparity_GT(os.path.join(folder, 'disp1.pfm'))
    )
    if maxsize is not None:
        img_L.thumbnail((maxsize, maxsize))
        img_R.thumbnail((maxsize, maxsize))
        disp_L.thumbnail((maxsize, maxsize))
        disp_R.thumbnail((maxsize, maxsize))

    return np.array(img_L), np.array(img_R), np.array(disp_L), np.array(disp_R)

def load_middlebury_dataset(data_dir, nb_samples=None, max_size=None):
    global datadir, maxsize
    datadir, maxsize = data_dir, max_size
    # create list of folders
    folders = [elt for elt in os.listdir(data_dir) if 'perfect' in elt]
    folders = sorted(folders)
    # select subset if needed
    if nb_samples is not None:
        assert type(nb_samples) == int and nb_samples > 0
        folders = folders[:nb_samples]
    # load all samples in parallel
    samples = parallel(load_one_sample, folders)

    return samples