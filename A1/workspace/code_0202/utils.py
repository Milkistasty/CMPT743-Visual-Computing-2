from collections import Counter

import numpy as np
import cv2
import torch
from dataset import iou
import os

# reference: https://github.com/zherenx/cmpt743-a3-SSD/blob/master/utils.py


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# use [blue green red] to represent different classes

def generate_box(x, y, w, h):
    # box = np.array([x, y, w, h, x-w/2.0, y-h/2.0, x+w/2.0, y+h/2.0], dtype=float)
    # box = np.transpose(np.vstack((x, y, w, h, x-w/2.0, y-h/2.0, x+w/2.0, y+h/2.0)))
    box = np.column_stack((x, y, w, h, x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0))
    box = np.clip(box, a_min=0, a_max=1)
    return box


def get_actual_boxes(boxes, boxs_default):
    actual_boxes = np.column_stack((
        boxs_default[:, 2] * boxes[:, 0] + boxs_default[:, 0],
        boxs_default[:, 3] * boxes[:, 1] + boxs_default[:, 1],
        boxs_default[:, 2] * np.exp(boxes[:, 2]),
        boxs_default[:, 3] * np.exp(boxes[:, 3])
    ))

    A = generate_box(actual_boxes[:, 0], actual_boxes[:, 1], actual_boxes[:, 2], actual_boxes[:, 3])
    return A


def convert_to_real_scale(boxes, w, h):
    boxes1 = np.zeros(boxes.shape)
    boxes1[:, ::2] = boxes[:, ::2] * w
    boxes1[:, 1::2] = boxes[:, 1::2] * h
    return boxes1.astype(int)


def visualize_pred(windowname, pred_boxes, pred_confidence, cat_ids, corresponding_default_boxes, ann_confidence,
                   ann_box,
                   boxs_default, image_, image_id=0, show=True):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)
    image1,image2,image3,image4 = image.copy(),image.copy(),image.copy(),image.copy()
    h, w, c = image.shape
    gt_boxes = get_actual_boxes(ann_box, boxs_default)
    gt_boxes1 = convert_to_real_scale(gt_boxes, w, h)

    if len(pred_boxes) == 0:
        print("Detection fail. image-[%d] no box detected; Msg uplink from visualize_pred()" % (image_id))
        return

    pred_boxes = convert_to_real_scale(pred_boxes, w, h)
    corresponding_default_boxes = convert_to_real_scale(corresponding_default_boxes, w, h)

    class_num = 3
    thickness = 2
    # draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i, j] == 1:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)

                color = colors[j]
                image1 = cv2.rectangle(image1, (round(gt_boxes1[i, 4]), round(gt_boxes1[i, 5])),
                                       (round(gt_boxes1[i, 6]), round(gt_boxes1[i, 7])), color, thickness)
                ious = iou(boxs_default, gt_boxes[i, 4], gt_boxes[i, 5], gt_boxes[i, 6], gt_boxes[i, 7])
                ious_true = ious > 0.5
                selected_boxes = boxs_default[ious_true, :]
                if not selected_boxes.size:
                    best_index = np.argmax(ious)
                    selected_boxes = boxs_default[best_index, :].reshape(1, -1)
                selected_boxes = convert_to_real_scale(selected_boxes, w, h)
                for box in selected_boxes:
                    image2 = cv2.rectangle(image2, (round(box[4]), round(box[5])), (round(box[6]), round(box[7])),
                                           color, thickness)

    # pred
    for i in range(len(pred_boxes)):
        # TODO:
        # image3: draw network-predicted bounding boxes on image3
        # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
        color = colors[cat_ids[i]]
        image3 = cv2.rectangle(image3, (round(pred_boxes[i, 4]), round(pred_boxes[i, 5])),
                               (round(pred_boxes[i, 6]), round(pred_boxes[i, 7])), color, thickness)

        image4 = cv2.rectangle(image4,
                               (round(corresponding_default_boxes[i, 4]), round(corresponding_default_boxes[i, 5])),
                               (round(corresponding_default_boxes[i, 6]), round(corresponding_default_boxes[i, 7])),
                               color, thickness)

    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h * 2, w * 2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    if show:
        cv2.imshow("11", image)



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.1, threshold=0.5):
    # TODO: non maximum suppression
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.

    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes, and write a new visualization function for that.

    # change the box to [x_min, y_min, x_max, y_max] format
    actual_boxes = get_actual_boxes(box_, boxs_default)

    suppressed_boxes = []
    suppressed_confidences = []  # new array to collect confidence
    pred_cat_ids = []
    corresponding_default_boxes = []

    keep_indices = np.ones(len(confidence_), dtype=bool)

    while keep_indices.any():
        num_of_boxes, num_of_classes = confidence_.shape
        max_index = np.unravel_index(np.argmax(confidence_[keep_indices, :-1], axis=None),
                                     (num_of_boxes, num_of_classes - 1))

        if confidence_[max_index] > threshold:
            x_confidence = confidence_[max_index[0], :-1]  # discard background
            max_confidence = np.max(x_confidence)  # get the max confidence
            x_box = actual_boxes[max_index[0], :]
            corresponding_default_box = boxs_default[max_index[0], :]

            keep_indices[max_index[0]] = False

            suppressed_boxes.append(x_box)
            suppressed_confidences.append(max_confidence)  # add the max confidence
            pred_cat_ids.append(np.argmax(x_confidence))
            corresponding_default_boxes.append(corresponding_default_box)
            ious = iou(actual_boxes[keep_indices], x_box[4], x_box[5], x_box[6], x_box[7])
            ious_true = ious > overlap
            # update keep_indices
            keep_indices[np.where(keep_indices)[0][ious_true]] = False
        else:
            break

    return np.array(suppressed_boxes), np.array(suppressed_confidences), np.array(pred_cat_ids), np.array(corresponding_default_boxes)


def save_predicted_boxes(pred_boxes, cat_ids, image_id):
    path = "data/test/"+"/images/"
    out_path = "predicted_boxes/"

    if not os.path.exists(out_path):
            os.makedirs(out_path)  # Create the directory if it does not exist

    image_id = image_id.item()

    # Correctly format the image file name
    img_name = path + str(image_id).zfill(5) + ".jpg" 
    filename = out_path + str(image_id).zfill(5) + ".txt"

    img = cv2.imread(img_name)
    if img is None:
        print(f"Failed to load image {img_name}")
        return
    
    img_h, img_w, img_c = img.shape

    f = open(filename, "w")

    if len(pred_boxes) == 0:
        print(f"Detection failed for image {image_id}; Msg uplink from save_predicted_boxes()")
        f.close()
        return

    for i in range(len(pred_boxes)):
        x_min = pred_boxes[i, 4]
        y_min = pred_boxes[i, 5]
        x_max = pred_boxes[i, 6]
        y_max = pred_boxes[i, 7]

        x_c = (x_min + x_max) / 2.0
        y_c = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min

        x_min = x_min * img_w
        y_min = y_min * img_h
        x_c = x_c * img_w
        y_c = y_c * img_h
        w = w * img_w
        h = h * img_h

        f.write(f"{cat_ids[i]} {x_min} {y_min} {w} {h}\n")
    
    f.close()


def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculate intersection over union for the predicted and ground truth boxes.

    Args:
    - boxes_preds (Tensor): Predicted bounding boxes
    - boxes_labels (Tensor): Ground truth bounding boxes

    Returns:
    - Tensor: IoU scores
    """
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def generate_mAP(pred_bboxes, true_boxes, iou_threshold, num_classes=4):
    """
    Calculate the mean average precision for object detection.

    Args:
    - pred_bboxes (list): List of predicted bounding boxes [train_idx, class_pred, prob_score, x1, y1, x2, y2]
    - true_boxes (list): List of ground truth boxes [train_idx, class_pred, x1, y1, x2, y2]
    - iou_threshold (float): IoU threshold to consider a detection valid
    - num_classes (int): Number of classes

    Returns:
    - float: Mean average precision
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [d for d in pred_bboxes if d[1] == c]
        ground_truths = [t for t in true_boxes if t[1] == c]

        amount_bboxes = Counter(gt[0] for gt in ground_truths)
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0


class Metric:
    def __init__(self):
        self.pred_boxex = []
        self.gt_boxex = []
        self.batch_id = 0

    def update(self,pred_boxes, pred_confidence, cat_ids, ann_confidence,
                   ann_box,
                   boxs_default, image_):

        image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)

        h, w, c = image.shape
        gt_boxes = get_actual_boxes(ann_box, boxs_default)
        gt_boxes1 = convert_to_real_scale(gt_boxes, w, h)

        # print(len(pred_boxes))
        if len(pred_boxes) == 0:
            return

        pred_boxes = convert_to_real_scale(pred_boxes, w, h)
        x0 = pred_boxes[:, 4]
        y0 = pred_boxes[:, 5]
        x1 = pred_boxes[:, 6]
        y1 = pred_boxes[:, 7]
        pred_box_for_ap = np.column_stack((torch.zeros(len(cat_ids))+self.batch_id, cat_ids, pred_confidence, x0, y0, x1, y1))

        x0 = []
        y0 = []
        x1 = []
        y1 = []
        gt_ids = []
        class_num = 3
        thickness = 2
        # draw ground truth
        for i in range(len(ann_confidence)):
            for j in range(class_num):
                if ann_confidence[i, j] == 1:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                    color = colors[j]
                    x0.append(gt_boxes1[i, 4])
                    y0.append(gt_boxes1[i, 5])
                    x1.append(gt_boxes1[i, 6])
                    y1.append(gt_boxes1[i, 7])
                    gt_ids.append(j)
        gt_box_for_ap = np.column_stack((torch.zeros(len(gt_ids))+self.batch_id, gt_ids, np.ones(len(gt_ids)), x0, y0, x1, y1))

        self.pred_boxex.append(pred_box_for_ap)
        self.gt_boxex.append(gt_box_for_ap)
        self.batch_id += 1

    def compute_mAP(self):
        print('comput mAP...')
        if len(self.pred_boxex) == 0:
            print('no box detected, mAP is 0; Msg uplink from compute_mAP()')
            return
        predbox = np.concatenate(self.pred_boxex, 0)
        gtbox = np.concatenate(self.gt_boxex, 0)
        print('mAP @0.5:')
        print(generate_mAP(predbox, gtbox, 0.5, 4).cpu().numpy())
        print('mAP @0.75:')
        print(generate_mAP(predbox, gtbox, 0.75, 4).cpu().numpy())
        print('mAP @0.95:')
        print(generate_mAP(predbox, gtbox, 0.95, 4).cpu().numpy())

    def compute_f1_score(self):
        print('comput f1 score...')
        if len(self.pred_boxex) == 0:
            print('no box detected, f1 score is 0; Msg uplink from compute_f1_score()')
            return
        predbox = np.concatenate(self.pred_boxex, 0)
        gtbox = np.concatenate(self.gt_boxex, 0)
        print('f1 score @0.5:')
        print(gen_f1_score(predbox, gtbox, 0.5, 4))
        print('f1 score @0.75:')
        print(gen_f1_score(predbox, gtbox, 0.75, 4))
        print('f1 score @0.95:')
        print(gen_f1_score(predbox, gtbox, 0.95, 4))

    def reset_states(self):
        self.pred_boxex = []
        self.gt_boxex = []
        self.batch_id = 0


def gen_f1_score(pred_bboxes, true_boxes, iou_threshold, num_classes):
    average_f1_scores = []  
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [d for d in pred_bboxes if d[1] == c]
        ground_truths = [t for t in true_boxes if t[1] == c]
        amount_bboxes = Counter(gt[0] for gt in ground_truths)
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)
        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        precision = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recall = TP_cumsum / (total_true_bboxes + epsilon)
        f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)
        if f1_scores.size() == 0:
            return 0
        max_f1_score = torch.max(f1_scores).item()
        average_f1_scores.append(max_f1_score)
    mean_f1_score = sum(average_f1_scores) / len(average_f1_scores) if average_f1_scores else 0

    return mean_f1_score
