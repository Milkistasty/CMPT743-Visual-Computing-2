from random import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2
from math import *

# Generate default bounding boxes based on specified layers, scales, and dimensions
def default_box_generator(layers, large_scale, small_scale):
    # The total number of default boxes is determined by the layer configurations and scales.
    # For each position on each feature map layer, multiple boxes with different sizes and aspect ratios are generated.
    # These boxes serve as anchors for detecting objects of various shapes and sizes in the image.

    # Calculate the total number of boxes
    box_num = sum([layer ** 2 for layer in layers]) * 4
    # Initialize the array to store boxes
    boxes = np.zeros((box_num, 8))
    # Counter for boxes
    index = 0
    for i, layer in enumerate(layers):
        # Determine the size of each cell in this layer
        step = 1.0 / layer
        # Sizes for the current layer
        lsize = large_scale[i]
        ssize = small_scale[i]
        for y in range(layer):
            for x in range(layer):
                # Center coordinates
                cx = (x + 0.5) * step
                cy = (y + 0.5) * step
                # Generate boxes
                box_sizes = [
                    (ssize, ssize),
                    (lsize, lsize),
                    (lsize * sqrt(2), lsize / sqrt(2)),
                    (lsize / sqrt(2), lsize * sqrt(2))
                ]
                for w, h in box_sizes:
                    # Calculate box coordinates
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2
                    # Store box attributes
                    boxes[index] = np.array([cx, cy, w, h, xmin, ymin, xmax, ymax])
                    index += 1
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
# Compute Intersection Over Union (IOU) between default bounding boxes and a specified bounding box
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1, box_r), iou(box_2, box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


# Match ground truth bounding boxes and classes with default boxes based on IOU
def match(ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max):
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box

    # This function assigns ground truth boxes to default boxes if their IOU exceeds a certain threshold.
    # It updates the annotations for the bounding box offsets and class confidences.

    # compute iou between the default bounding boxes and the ground truth bounding box
    # Calculate IOUs for all default boxes
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)

    # Find indices of boxes with IOU greater than the threshold
    ious_true_indices = np.where(ious > threshold)[0]

    # Function to update annotation for a given index
    def update_annotation(index):
        # Calculate and update box annotation
        px, py, pw, ph = boxs_default[index, :4]
        tx = (x_min + x_max) / 2 - px
        ty = (y_min + y_max) / 2 - py
        tw = np.log((x_max - x_min) / pw)
        th = np.log((y_max - y_min) / ph)
        ann_box[index] = [tx, ty, tw, th]

        # Update confidence: reset to background, then set class
        ann_confidence[index, :] = 0
        ann_confidence[index, cat_id] = 1

    # Update annotations for boxes with IOU above threshold
    for index in ious_true_indices:
        update_annotation(index)

    # Ensure at least one box is updated
    if len(ious_true_indices) == 0:
        max_iou_index = np.argmax(ious)
        update_annotation(max_iou_index)

    return ann_box, ann_confidence

class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # The '__getitem__' method returns a preprocessed image and its corresponding ground truth annotations.

        img_name = os.path.join(self.imgdir, self.img_names[index])
        ann_name = os.path.join(self.anndir, self.img_names[index][:-3] + "txt")

        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)  # one-hot vectors
        ann_confidence[:, -1] = 1  # the default class for all cells is set to "background"

        img = cv2.imread(img_name)
        img_h, img_w, img_c = img.shape

        # Randomg cropping 
        crop_threshold = 0.1
        if self.train:
            ax = int(np.random.rand() * img_w * crop_threshold)
            ay = int(np.random.rand() * img_h * crop_threshold)
            bx = int(img_w - np.random.rand() * img_w * crop_threshold)
            by = int(img_h - np.random.rand() * img_h * crop_threshold)

            img = img[ay:by, ax:bx, :]
            img_h = by - ay
            img_w = bx - ax

        # Resize the img to 320*320
        img = cv2.resize(img, (320, 320))
        img = np.transpose(img, (2, 0, 1))

        with open(ann_name, 'r') as annotations_txt:
            annotations = annotations_txt.readlines()

        for i, annotation in enumerate(annotations):
            cat_id, x_min, y_min, w, h = map(float, annotation.split())
            x_max = x_min + w
            y_max = y_min + h

            if self.train:
                x_min = max(x_min - ax, 0)
                y_min = max(y_min - ay, 0)
                x_max = min(x_max - ax, img_w)
                y_max = min(y_max - ay, img_h)

            x_min, y_min, x_max, y_max = x_min / img_w, y_min / img_h, x_max / img_w, y_max / img_h
            
            # Annotations are processed to match default boxes with ground truth boxes based on IOU.
            match(ann_box, ann_confidence, self.boxs_default, self.threshold, int(cat_id), x_min, y_min, x_max, y_max)
        if self.train:
            return img, ann_box, ann_confidence
        else:
            return img, ann_box, ann_confidence, int(self.img_names[index][:-4])
