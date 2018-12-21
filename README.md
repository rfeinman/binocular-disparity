# Computing Binocular Disparity with Convolutional Neural Networks and Conditional Random Fields

This is a code repository for computing binocular disparity with a combination of Convolutional Neural Networks and Conditional Random Fields. The disparity model requires **zero labeled training examples**. A pre-trained ImageNet CNN is used for feature extraction. This project was assembled for Joan Bruna's 2018 NYU course "Inference and Representation." NOTE: the repository has only been tested with Python3.

<img src="documents/model_diagram.png" width="706" height="300">

## Requirements & Setup

Make sure that all requirements are installed on your machine before you run the code. A full list of requirements can be found in `requirements.txt`. To install the software, run the following command to clone the repository into a folder of your choice:
```
git clone https://github.com/rfeinman/binocular-disparity.git
```
On UNIX machines, after cloning this repository, it is recommended that you add the repository to your `PYTHONPATH` environment variable to enable imports from any folder:
```
export PYTHONPATH="/path/to/binocular-disparity:$PYTHONPATH"
```

## Usage Example

The following code demo shows how to compute disparity for a left-right
image pair.

```python
import numpy as np
from disparity import cnn, crf, util

# Create a function to load your left and right image.
image_left, image_right = load_images()
height, width, _ = image_left.shape

# Compute disparity energies for a left-right image pair.
# This returns an array of size (height, width, numDisparities)
energies = cnn.compute_energies(image_left, image_right, numDisparities=120)

# Select an optimal disparity threshold based on energy entropy
threshold = util.select_disparity_threshold(energies)
energies = energies[:,:,:threshold]

# Compute the initial disparity for each pixel by finding the disparity value
# with minimum energy at that pixel
disparity = np.argmin(energies, axis=2)

# Perform MAP inference with loopy BP (max-product message passing)
smoother = crf.MaxProductLBP(height, width, num_beliefs=threshold)
disparity = smoother.decode_MAP(disparity, iterations=30)
```

## Benchmark dataset

Our experiment scripts use the Middlebury stereo dataset. To obtain the dataset,
download the zip file at the following link:

<http://www.cns.nyu.edu/~reuben/files/middlebury.zip>.

Then, unzip the folder and place it inside `data/`.

### Script

To run the model on the whole Middlebury dataset, use the experiment script
`scripts/middlebury_experiment.py`. You can select which CRF inference algorithm
to use with the `--crf_alg` parameter (options are gradient descent, max-product
loopyBP, sum-product loopyBP).


### Results

<img src="documents/results.png" width="700" height="992">