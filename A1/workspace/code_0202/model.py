import os
import random
import numpy as np

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


# reference: https://github.com/zherenx/cmpt743-a3-SSD/blob/master/model.py


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    # Flatten inputs for loss computation
    batch_size, num_of_boxes, _ = ann_confidence.shape

    pred_confidence_flat = pred_confidence.view(-1, pred_confidence.size(-1))
    ann_confidence_flat = ann_confidence.view(-1, ann_confidence.size(-1))

    pred_box_flat = pred_box.view(-1, 4)
    ann_box_flat = ann_box.view(-1, 4)

    is_obj = (ann_confidence_flat[:, -1] == 0)
    noobj = ~is_obj  #

    cls_loss = F.binary_cross_entropy(pred_confidence_flat, ann_confidence_flat, reduction='none')

    cls_loss = cls_loss.mean(dim=1)

    cls_loss = cls_loss[is_obj].mean() + 3 * cls_loss[noobj].mean()

    box_loss = torch.abs(pred_box_flat[is_obj] - ann_box_flat[is_obj]).mean()

    total_loss = cls_loss + box_loss
    
    return total_loss


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        self.class_num = class_num  # num_of_classes, in this assignment, 4: cat, dog, person, background

        # TODO: define layers
        # all the conv block and batch norm for base layers on the left-most route
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        # first split
        self.conv14 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn14 = nn.BatchNorm2d(256)
        self.conv15 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(256)
        # second split
        self.conv16 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn16 = nn.BatchNorm2d(256)
        self.conv17 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn17 = nn.BatchNorm2d(256)
        # third split
        self.conv18 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn18 = nn.BatchNorm2d(256)
        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn19 = nn.BatchNorm2d(256)

        # Red Route
        # for the red 1 route
        self.conv20 = nn.Conv2d(256, 16, kernel_size=1, stride=1)
        # for the red 2 route
        self.conv21 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        # for the red 3 route
        self.conv22 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        # for the red 4 route
        self.conv23 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)

        # Blue Route
        # for the blue 1 route
        self.conv24 = nn.Conv2d(256, 16, kernel_size=1, stride=1)
        # for the blue 2 route
        self.conv25 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        # for the blue 3 route
        self.conv26 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        # for the blue 4 route
        self.conv27 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]

        x = x / 255.0  # normalize image. If you already normalized your input image in the dataloader, remove this line.

        # TODO: define forward

        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?

        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        # bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        # Passing through the base layers
        x1 = F.relu(self.bn1(self.conv1(x)))  # [N, 64, 160, 160]
        x2 = F.relu(self.bn2(self.conv2(x1)))  # [N, 64, 160, 160]
        x3 = F.relu(self.bn3(self.conv3(x2)))  # [N, 64, 160, 160]
        x4 = F.relu(self.bn4(self.conv4(x3)))  # [N, 128, 80, 80]
        x5 = F.relu(self.bn5(self.conv5(x4)))  # [N, 128, 80, 80]
        x6 = F.relu(self.bn6(self.conv6(x5)))  # [N, 128, 80, 80]
        x7 = F.relu(self.bn7(self.conv7(x6)))  # [N, 256, 40, 40]
        x8 = F.relu(self.bn8(self.conv8(x7)))  # [N, 256, 40, 40]
        x9 = F.relu(self.bn9(self.conv9(x8)))  # [N, 256, 40, 40]
        x10 = F.relu(self.bn10(self.conv10(x9)))  # [N, 512, 20, 20]
        x11 = F.relu(self.bn11(self.conv11(x10)))  # [N, 512, 20, 20]
        x12 = F.relu(self.bn12(self.conv12(x11)))  # [N, 512, 20, 20]
        x13 = F.relu(self.bn13(self.conv13(x12)))  # [N, 256, 10, 10]

        # First split
        x14 = F.relu(self.bn14(self.conv14(x13)))  # [N, 256, 10, 10]
        x15 = F.relu(self.bn15(self.conv15(x14)))  # [N, 256, 5, 5]

        # Second split
        x16 = F.relu(self.bn16(self.conv16(x15)))  # [N, 256, 5, 5]
        x17 = F.relu(self.bn17(self.conv17(x16)))  # [N, 256, 3, 3]

        # Third split
        x18 = F.relu(self.bn18(self.conv18(x17)))  # [N, 256, 3, 3]
        x19 = F.relu(self.bn19(self.conv19(x18)))  # [N, 256, 1, 1]

        # Red Route 1 after conv block 19
        red1 = self.conv20(x19)  # [N, 16, 1, 1]
        red1 = red1.view(red1.size(0), -1, 1)  # Reshape to [N, 16, 1]
        # Red Route 2 from first split
        red2 = self.conv21(x13)  # [N, 16, 10, 10]
        red2 = red2.view(red2.size(0), red2.size(1), -1)  # Reshape to [N, 16, 100]
        # Red Route 3 from second split
        red3 = self.conv22(x15)  # [N, 16, 5, 5]
        red3 = red3.view(red3.size(0), red3.size(1), -1)  # Reshape to [N, 16, 25]
        # Red Route 4 from third split
        red4 = self.conv23(x17)  # [N, 16, 3, 3]
        red4 = red4.view(red4.size(0), red4.size(1), -1)  # Reshape to [N, 16, 9]

        # Concatenate the routes for bboxes
        bboxes = torch.cat([red1, red2, red3, red4], dim=2)  # [N, 16, 135]
        # Permute
        bboxes = bboxes.permute(0, 2, 1).contiguous()  # [N, 135, 16]
        # Reshape
        bboxes = bboxes.view(bboxes.size(0), -1, 4)  # [N, 540, 4]

        # Blue Route 1 after conv block 19
        blue1 = self.conv24(x19)  # [N, 16, 1, 1]
        blue1 = blue1.view(blue1.size(0), -1, 1)  # Reshape to [N, 16, 1]
        # Blue Route 2 from first split
        blue2 = self.conv25(x13)  # [N, 16, 10, 10]
        blue2 = blue2.view(blue2.size(0), blue2.size(1), -1)  # Reshape to [N, 16, 100]
        # Blue Route 3 from second split
        blue3 = self.conv26(x15)  # [N, 16, 5, 5]
        blue3 = blue3.view(blue3.size(0), blue3.size(1), -1)  # Reshape to [N, 16, 25]
        # Blue Route 4 from third split
        blue4 = self.conv27(x17)  # [N, 16, 3, 3]
        blue4 = blue4.view(blue4.size(0), blue4.size(1), -1)  # Reshape to [N, 16, 9]

        # Concatenate the routes for confidences
        confidences = torch.cat([blue1, blue2, blue3, blue4], dim=2)  # [N, 16, 135]
        # Permute
        confidences = confidences.permute(0, 2, 1).contiguous()  # [N, 135, 16]
        # Reshape
        confidences = confidences.view(confidences.size(0), -1, self.class_num)  # [N, 540, 4]
        # Applying softmax
        confidences = F.softmax(confidences, dim=2)  # [N, 540, 4]

        return confidences,bboxes



