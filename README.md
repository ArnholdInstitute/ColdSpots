
<h1 align="center">Cold Spots</h1>

<p align="center">
<a href="https://sinai-survey.tk">
<img src="https://i.imgur.com/0ccpCe0.jpg" height="150" />
</a>
</p>

This repository contains all code necessary to reproduce cold spot results.

Clone the repository along with all submodules:
```
git clone --recursive https://github.com/ArnholdInstitute/ColdSpots.git
```

# Installation

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

If the above gives you a certificate error execute
`curl https://bootstrap.pypa.io/get-pip.py | python`

```
git submodule update --init --recursive
```


### Tensorbox

Build the C++ auxilary functions:

```
cd TensorBox/utils 
make 
make hungarian
```

If the above throws an error, find the path of nsync_cv.h and nsync_mu.h inside the venv folder and find mutex.h 
(should be at /venv/local/lib/python2.7/site-packages/tensorflow/include/tensorflow/core/platform/default 
or rather you can look at the error message). In the mutex.h folder you will see it says include nysnc_cv.h
and nsync_mu.h. Replace these with the appropriate file paths. Then do `make hungarian`.

If you want to run on a GPU, you'll also need to `pip install tensorflow-gpu`.

# Inference

Below is an example of how to get the bounding boxes for each building within an image:

```
python .\demo.py
```

# Discussion and Model Description
This project uses the exemplary Tensorbox object detection library by Russell Stewart to draw 
bounding boxes around buildings in images.  Other NN libraries were included
for the purpose of comparison; testing revealed that Tensorbox provided the best results.

Tensorbox is supposedly an implementation of the GoogLeNet/OverFeat-based algorithm discussed here:
https://arxiv.org/pdf/1506.04878.pdf .  The algorithm described uses the GoogLeNet neural
network to encode an image into a block of high-level features, then feeds those features into
a set of chained LSTM modules trained to predict bounding boxes in order of decreasing confidence.

Training uses a loss function called "Hungarian loss", which penalizes bounding-box sequences
which a) do not have overlap with ground truth, b) are not in decreasing order of confidence,
and c) have large L1 deviation ("taxicab metric distance") from ground truth.  This is to deal
with the fact that the LSTM modules will generally make _multiple_ bounding-box predictions 
for a single object -- we need to penalize later predictions, and in general want to penalize
too many predictions for one object. This is distinct from the older OverFeat model, which 
attempted to heuristically merge multiple bounding-box predictions into a single box.

We say supposedly, however, since Tensorbox is designed to be flexible in its implementation.
It does not require that the NN which encodes the image into high-level features is actually
based on the GoogLeNet model (which uses Inception modules as the base element in each layer).
Tensorbox additionally supports ResNet encodings and MobileNet encodings.

# Training

This calls a file of default arguments and hyperparameters located in hypes.json;
if you want to change things around regarding e.g. where data is located, learning rates,
or the like, this is the file to fiddle with.
```
python .\train.py
```



# Model Description

```
This model is very unexpected. First the image is passed through google_net. Then the output
of one of the earlier layers is taken and input into a 5 step multi-level RNN. The outputs are 
supposed to be potential location of boxes and also confidences. The loss function is the hungarian 
loss. One could look for reference to this paper: https://arxiv.org/pdf/1506.04878.pdf .
```
