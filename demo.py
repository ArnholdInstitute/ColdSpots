#!/usr/bin/env python

import pdb, json, cv2, numpy as np
from scipy.misc import imread
from TensorBox.predict import TensorBox

# Read in image (RGB format)
img = imread('data/liberia_sample_940.jpg')

# Reconstruct the model, will automatically download weights
model = TensorBox()

description = json.load(open('weights/tensorbox/description.json'))

# Infer buildings
result = model.predict_image(img)
result = result[result.score > description['threshold']]

orig = img.copy()

# Plot the boxes on the original image
for box in result.values[:, :4].round().astype(int):
    cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (0,0,255))

space = np.zeros([orig.shape[0], 5, 3])
cv2.imwrite('with_annotated_buildings.jpg', np.concatenate([orig, space, img], axis=1))