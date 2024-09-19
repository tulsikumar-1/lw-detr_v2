import numpy as np
import json

def apply_confidence_threshold(predictions, threshold):
    """ Filter predictions based on a confidence threshold. """
    return [pred for pred in predictions if pred['score'] >= threshold]


def compute_metrices(pred_file, val_ann_file, iou_threshould, confidence_threshold=0.0):
    with open(pred_file, 'r') as file:
        predictions = json.load(file)

    with open(val_ann_file, 'r') as file:
        ground_truths = json.load(file)
        annotations = ground_truths['annotations']
    

    precisions, recalls, ap,_,_,_=calculate_precision_recall_ap(predictions, annotations, iou_threshould,confidence_threshold)
    
    
    return np.mean(precisions),np.mean(recalls), ap
        



#necessary functions here (extract_coco_data, apply_nms, calculate_iou)
def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[0] + box1[2], box2[0] + box2[2]), min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area, box2_area = box1[2] * box1[3], box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def extract_coco_data(data):
    """Extract bounding boxes, scores, and labels from COCO-format data."""
    boxes = [item['bbox'] for item in data]
    scores = [item.get('score', 1.0) for item in data]  # Default to 1.0 for ground truth
    labels = [item['category_id'] for item in data]
    return boxes, scores, labels

def apply_nms(predictions, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes."""
    if not predictions:
        return []

    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    filtered_predictions = []
    while predictions:
        best_pred = predictions.pop(0)
        filtered_predictions.append(best_pred)
        predictions = [pred for pred in predictions if calculate_iou(best_pred['bbox'], pred['bbox']) < iou_threshold]

    return filtered_predictions





def calculate_precision_recall_ap(predictions, annotations, iou_threshold=0.5,conf_threshold=0.5):
    tp, fp, all_scores = [], [], []
    class_tp, class_fp, class_scores, class_gt_boxes = {}, {}, {}, {}

    # Initialize dictionaries for classes
    for gt in annotations:
        class_id = gt['category_id']
        if class_id not in class_gt_boxes:
            class_gt_boxes[class_id] = 0
        class_gt_boxes[class_id] += 1

    total_gt_boxes = sum(class_gt_boxes.values())  # Total number of ground truth boxes

    for image_id in set([pred['image_id'] for pred in predictions] + [gt['image_id'] for gt in annotations]):
        preds = [pred for pred in predictions if pred['image_id'] == image_id and pred['category_id'] != 0]
        g_truths = [gt for gt in annotations if gt['image_id'] == image_id]

        if not preds and not g_truths:
            continue  # Skip if there are no predictions and no ground truths for this image

        gt_boxes, _, gt_labels = extract_coco_data(g_truths)
        preds=apply_confidence_threshold(preds,conf_threshold)
        preds_nms = apply_nms(preds, 0.2)
        preds_nms=preds
        pred_boxes, pred_scores, pred_labels = extract_coco_data(preds_nms)

        matched_gt = {cls: set() for cls in class_gt_boxes}

        for pred_box, score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
            if pred_label not in class_tp:
                class_tp[pred_label] = []
                class_fp[pred_label] = []
                class_scores[pred_label] = []

            best_iou, best_gt_index = 0, -1
            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label != pred_label:
                    continue
                if i in matched_gt[pred_label]:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_index = iou, i

            if best_iou >= iou_threshold and best_gt_index != -1:
                tp.append(1)
                fp.append(0)
                matched_gt[pred_label].add(best_gt_index)
                class_tp[pred_label].append(1)
                class_fp[pred_label].append(0)
                #if pred_label==1:
                  #print(preds_nms[0]['image_id'])
            else:
                tp.append(0)
                fp.append(1)
                class_tp[pred_label].append(0)
                class_fp[pred_label].append(1)

            class_scores[pred_label].append(score)  # Append score for the current class
            all_scores.append(score)

    tp, fp, all_scores = np.array(tp), np.array(fp), np.array(all_scores)
    #print(all_scores)
    # Sort by score (high to low)
    indices = np.argsort(-all_scores)
    tp, fp, all_scores = tp[indices], fp[indices], all_scores[indices]

    tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cumsum / total_gt_boxes if total_gt_boxes > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    precisions, recalls = np.insert(precisions, 0, 1), np.insert(recalls, 0, 0)
    precisions, recalls = np.append(precisions, 0), np.append(recalls, 1)
        # Compute the precision envelope
    precisions = np.flip(np.maximum.accumulate(np.flip(precisions)))

    x = np.linspace(0, 1, 101)  # 101-point interpolation (COCO)
    ap = np.trapz(np.interp(x, recalls, precisions), x)  # Integrate under the curve

    #ap = np.trapz(precisions, recalls)

    # Compute class-wise precision and recall
    class_precisions, class_recalls, class_aps = {}, {}, {}

    for cls in class_tp:
        tp_cls = np.array(class_tp[cls])
        fp_cls = np.array(class_fp[cls])
        scores_cls = np.array(class_scores.get(cls, []))  # Get scores for the current class

        gt_count = class_gt_boxes.get(cls, 0)

        if gt_count > 0 and len(scores_cls) > 0:
            # Sort the indices for the current class based on scores
            sorted_indices_cls = np.argsort(-scores_cls)
            tp_cls = tp_cls[sorted_indices_cls]
            fp_cls = fp_cls[sorted_indices_cls]

            tp_cumsum_cls, fp_cumsum_cls = np.cumsum(tp_cls), np.cumsum(fp_cls)
            recalls_cls = tp_cumsum_cls / gt_count
            precisions_cls = tp_cumsum_cls / (tp_cumsum_cls + fp_cumsum_cls)

            # Ensure the arrays have the correct length
            precisions_cls1, recalls_cls1 = np.insert(precisions_cls, 0, 1), np.insert(recalls_cls, 0, 0)
            precisions_cls1, recalls_cls1 = np.append(precisions_cls1, 0), np.append(recalls_cls1, 1)
            precisions_cls1 = np.flip(np.maximum.accumulate(np.flip(precisions_cls1)))

            ap_cls = np.trapz(np.interp(x, recalls_cls1, precisions_cls1), x)

            class_precisions[cls] = precisions_cls1
            class_recalls[cls] = recalls_cls1
            class_aps[cls] = ap_cls

    return precisions, recalls, ap, class_precisions, class_recalls, class_aps
