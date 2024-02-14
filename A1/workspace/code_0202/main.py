import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32
DEVICE = 'cuda'

boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.load_state_dict(torch.load('network-100.pth', map_location=torch.device('cuda')))
network.to(DEVICE)
cudnn.benchmark = True


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.to(DEVICE)
            ann_box = ann_box_.to(DEVICE)
            ann_confidence = ann_confidence_.to(DEVICE)

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        val_metric = Metric()
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, image_id = data
            images = images_.to(DEVICE)
            ann_box = ann_box_.to(DEVICE)
            ann_confidence = ann_confidence_.to(DEVICE)

            pred_confidence, pred_box = network(images)

            pred_confidence_ = pred_confidence[0].detach().cpu()
            pred_box_ = pred_box[0].detach().cpu()
            cls_score, cls_name = pred_confidence_.max(1)
            pred_confidence_ = pred_confidence_.numpy()
            pred_box_ = pred_box_.numpy()

            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)

            suppressed_boxes, suppressed_confidences, pred_cat_ids, corresponding_default_boxes = non_maximum_suppression(
                pred_confidence_,
                pred_box_, boxs_default)
            val_metric.update(suppressed_boxes, suppressed_confidences, pred_cat_ids, ann_confidence_[0].numpy(),
                               ann_box_[0].numpy(), boxs_default, images_[0].numpy())
        val_metric.compute_mAP()
        val_metric.compute_f1_score()
        val_metric.reset_states()

        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')
            print('done.')


else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth', map_location=torch.device('cpu')))
    network.eval()
    test_metric = Metric()
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_,image_id = data
        images = images_.to(DEVICE)
        ann_box = ann_box_.to(DEVICE)
        ann_confidence = ann_confidence_.to(DEVICE)

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu()
        pred_box_ = pred_box[0].detach().cpu()
        cls_score,cls_name = pred_confidence_.max(1)
        pred_confidence_ = pred_confidence_.numpy()
        pred_box_ = pred_box_.numpy()
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        print(f"Processing image {image_id}")
        suppressed_boxes,suppressed_confidences, pred_cat_ids, corresponding_default_boxes = non_maximum_suppression(pred_confidence_,
                                                                                              pred_box_, boxs_default)
        test_metric.update(suppressed_boxes,suppressed_confidences, pred_cat_ids,ann_confidence_[0].numpy(),
                           ann_box_[0].numpy(), boxs_default, images_[0].numpy())


        visualize_pred("test", suppressed_boxes,suppressed_confidences, pred_cat_ids, corresponding_default_boxes,
                              ann_confidence_[0].numpy(), ann_box_[0].numpy(), boxs_default, images_[0].numpy(),
                              image_id)

        # visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        cv2.waitKey(10)

        save_predicted_boxes(suppressed_boxes, pred_cat_ids, image_id)
        
    test_metric.compute_mAP()
    test_metric.compute_f1_score()
    test_metric.reset_states()


