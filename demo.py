#!/usr/bin/env python
import pandas as pd
import pdb, json, cv2, numpy as np
from scipy.misc import imread
#import matplotlib.pyplot as plt
from TensorBox.predict import TensorBox
import os

# Read in image (RGB format)
img = imread('data/liberia_sample_940.jpg')
img = imread('data/senegal_sample_113.jpg')
# Reconstruct the model, will automatically download weights
model = TensorBox()

description = json.load(open('weights/tensorbox/description.json'))


#orig_img = img.copy()[:,:,:3]
#img = imresize(orig_img, (model.H["image_height"], model.H["image_width"]), interp='cubic')
#feed = {model.x_in: img}


#os.chdir('/home/atlas_user/ColdSpots/TensorBox')

import tensorflow as tf, os, cv2, pdb, numpy as np, time, json, pandas
from scipy.misc import imread, imresize
from scipy import misc
print(os.getcwd())

from TensorBox.train import build_forward
from TensorBox.utils.annolist import AnnotationLib as al
from TensorBox.utils.train_utils import add_rectangles, rescale_boxes
from shapely.geometry import MultiPolygon, box
from subprocess import check_output


orig_img = img.copy()[:,:,:3]
img = imresize(orig_img, (model.H["image_height"], model.H["image_width"]), interp='cubic')
feed = {model.x_in: img}

(np_pred_boxes, np_pred_confidences) = model.session.run([model.pred_boxes, model.pred_confidences], feed_dict=feed)
new_img, rects, all_rects = add_rectangles(model.H, [img], np_pred_confidences, np_pred_boxes, use_stitching=True, rnn_len=model.H['rnn_len'], min_conf=description['threshold'], tau=0.25, show_suppressed=False)



# Infer buildings
result = model.predict_image(img,description['threshold'])
#result = result[result.score > description['threshold']]


print(result.shape)

orig = img.copy()

# Plot the boxes on the original image
for box in result.values[:, :4].round().astype(int):
    cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (0,0,255))

cv2.imwrite('data/SavedImages/senegal_sample_113.jpg', img)
#misc.imshow(img)

space = np.zeros([orig.shape[0], 5, 3])
cv2.imwrite('with_annotated_buildings.jpg', np.concatenate([orig, space, img], axis=1))
