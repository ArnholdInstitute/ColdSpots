
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

### Tensorbox

Build the C++ auxilary functions:

```
cd TensorBox/utils && make && cd ../../
```

If you want to run on a GPU, you'll also need to `pip install tensorflow-gpu`.

# Inference

Below is an example of how to get the bounding boxes for each building within an image:

```Python
import pdb, json, cv2, numpy as np
from scipy.misc import imread
from TensorBox.predict import TensorBox

# Read in image (RGB format)
img = imread('data/liberia_sample_940.jpg')

# Reconstruct the model
description = json.load(open('weights/tensorbox/description.json'))
model = TensorBox('weights/tensorbox/' + description['weights'])

# Infer buildings
result = model.predict_image(img, description['threshold'])

orig = img.copy()

# Plot the boxes on the original image
for box in result.values[:, :4].round().astype(int):
    cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (0,0,255))

space = np.zeros([orig.shape[0], 5, 3])
cv2.imwrite('with_annotated_buildings.jpg', np.concatenate([orig, space, img], axis=1))
```

![Imgur](https://i.imgur.com/6mgiIGo.jpg)
