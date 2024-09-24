# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:02:01 2024

@author: Administrator
"""
import numpy as np
import json



def compute_metrices(pred_file, val_ann_file, iou_range=(0.5, 0.5), step=0.05, confidence_threshold=0.0):
    with open(pred_file, 'r') as file:
        predictions = json.load(file)

    with open(val_ann_file, 'r') as file:
        ground_truths = json.load(file)
        annotations = ground_truths['annotations']

    image_id = 0
    aps = []
    precision_scores = []
    recall_scores = []

    while True:
        preds = [gt for gt in predictions if gt['image_id'] == image_id]
        preds = [gt for gt in preds if gt['category_id'] != 0]
        if not preds:  # Break the loop if no more predictions for next image_id
            break

        g_truths = [gt for gt in annotations if gt['image_id'] == image_id]
        gt_boxes, _, gt_labels = extract_coco_data(g_truths)
        preds_nms = apply_nms(preds, confidence_threshold)

        precision, recall, map_score = compute_map_over_iou_range(preds_nms, gt_boxes, gt_labels, iou_range=iou_range, step=step, confidence_threshold=confidence_threshold)

        # Avoid NaN by checking if lists are non-empty
        if precision and recall:
            precision_scores.append(np.mean(precision))
            recall_scores.append(np.mean(recall))
            aps.append(map_score)

        image_id += 1

    if not precision_scores or not recall_scores:
        return 0.0, 0.0, np.mean(aps) if aps else 0.0

    return np.mean(precision_scores), np.mean(recall_scores), np.mean(aps)




def calculate_iou(box1, box2):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes. """
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def extract_coco_data(data):
    """ Extract bounding boxes, scores, and labels from COCO-format data. """
    boxes = [item['bbox'] for item in data]
    scores = [item.get('score', 1.0) for item in data]  # Use 1.0 as default for ground truth
    labels = [item['category_id'] for item in data]
    return boxes, scores, labels

def apply_confidence_threshold(predictions, threshold):
    """ Filter predictions based on a confidence threshold. """
    return [pred for pred in predictions if pred['score'] >= threshold]

def calculate_precision_recall_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    tp = []
    fp = []
    all_scores = []

    matched_gt = set()  # To keep track of which ground truth boxes have been matched
  #  if len(pred_boxes)==0:
  #    return [0],[0],0
    for pred_box, score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
        best_iou = 0
        best_gt_index = -1

        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if i in matched_gt:  # Skip already matched ground truth boxes
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou and pred_label == gt_label:  # Ensure class IDs match
                best_iou = iou
                best_gt_index = i

        if best_iou >= iou_threshold and best_gt_index != -1:
            tp.append(1)  # True Positive
            fp.append(0)  # Not a False Positive
            matched_gt.add(best_gt_index)  # Mark this ground truth as matched
        else:
            tp.append(0)  # Not a True Positive
            fp.append(1)  # False Positive

        all_scores.append(score)


    # Convert to numpy arrays
    tp = np.array(tp)
    fp = np.array(fp)
    all_scores = np.array(all_scores)

    # Sort by scores descending
    indices = np.argsort(-all_scores)
    tp = tp[indices]
    fp = fp[indices]
    all_scores = all_scores[indices]
   # print(all_scores)

   # print(f'true positive :{tp}')
   # print(f'False positive :{fp}')
    # Calculate cumulative true positives and false positives
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Calculate recall and precision
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    precisions1 = np.insert(precisions, 0, 1)
    recalls1 = np.insert(recalls, 0, 0)
   # print(f'precision:  {precisions}')
   # print(f'recalls:  {recalls}')
    # Compute Average Precision (AP)
    ap = np.trapz(precisions1, recalls1)  # Area under the precision-recall curve

    return precisions, recalls, ap

def compute_map_over_iou_range(predictions, gt_boxes, gt_labels, iou_range=(0.5, 0.95), step=0.05, confidence_threshold=0.5):
    # Apply confidence thresholding
    filtered_predictions = apply_confidence_threshold(predictions, confidence_threshold)

    # Extract boxes, scores, and labels after thresholding
    pred_boxes, pred_scores, pred_labels = extract_coco_data(filtered_predictions)
    if iou_range[0]!=iou_range[1]:
      iou_thresholds = np.arange(iou_range[0], iou_range[1] + step, step)
    else:
      iou_thresholds=[iou_range[0]]
    #print(f'iou_thresholds : {iou_thresholds}')
    aps = []
    precisions=[]
    recalls=[]

    for iou_threshold in iou_thresholds:
        pre, rec, ap = calculate_precision_recall_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold)
       # print(f'AP: {ap: .2f} for threshold :{iou_threshold : .2f}')

        aps.append(ap)
        precisions.append(pre.mean())
        recalls.append(rec.mean())

       # print(ap)
       # print('\n')

    # Calculate the mean of the APs over all IoU thresholds
    map_score = np.mean(aps)

    return precisions,recalls,map_score


def is_box_inside(bboxA, bboxB):
    """
    Check if bounding box A is entirely within bounding box B.

    Args:
        bboxA (tuple): (xmin, ymin, xmax, ymax) for box A.
        bboxB (tuple): (xmin, ymin, xmax, ymax) for box B.

    Returns:
        bool: True if bboxA is inside bboxB, False otherwise.
    """
    # Unpack the bounding boxes
    xminA, yminA, xmaxA, ymaxA = bboxA[0],bboxA[1],bboxA[0]+bboxA[2],bboxA[1]+bboxA[3]
    xminB, yminB, xmaxB, ymaxB = bboxB[0],bboxB[1],bboxB[0]+bboxB[2],bboxB[1]+bboxB[3]

    # Check if all corners of bboxA are within bboxB
    return (xminA >= xminB and yminA >= yminB and
            xmaxA <= xmaxB and ymaxA <= ymaxB)

def apply_nms(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes
    and remove boxes that are entirely inside other boxes.

    Args:
        predictions (list): List of dictionaries where each dictionary contains:
            - 'bbox': (xmin, ymin, xmax, ymax) for the bounding box.
            - 'score': confidence score of the bounding box.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        list: List of dictionaries containing the filtered bounding boxes.
    """
    if len(predictions) == 0:
        return []

    # Convert predictions to numpy arrays for easier manipulation
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # Step 1: Remove boxes that are completely inside other boxes
    filtered_predictions = []
    while predictions:
        best_pred = predictions.pop(0)
        filtered_predictions.append(best_pred)

        # Filter out boxes inside the best_pred
        predictions = [pred for pred in predictions if not is_box_inside(pred['bbox'], best_pred['bbox'])]

    # Step 2: Apply standard IoU-based NMS on the filtered predictions
    nms_predictions = []
    while filtered_predictions:
        best_pred = filtered_predictions.pop(0)
        nms_predictions.append(best_pred)

        # Compare IoU with remaining boxes
        filtered_predictions = [pred for pred in filtered_predictions if calculate_iou(best_pred['bbox'], pred['bbox']) < iou_threshold]

    return nms_predictions
