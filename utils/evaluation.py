"""
Evaluation utilities for object detection and segmentation
Calculates precision, recall, mAP, and other metrics
"""

import numpy as np
from typing import List, Dict, Tuple
import torch


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks
    
    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)
    
    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0


def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[List[bool], List[bool]]:
    """
    Match predicted detections with ground truth boxes
    
    Args:
        pred_boxes: Predicted boxes (N, 4)
        pred_scores: Confidence scores (N,)
        pred_classes: Predicted classes (N,)
        gt_boxes: Ground truth boxes (M, 4)
        gt_classes: Ground truth classes (M,)
        iou_threshold: IoU threshold for matching
    
    Returns:
        true_positives: Boolean array for predictions
        matched_gt: Boolean array for ground truth
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    
    true_positives = [False] * num_preds
    matched_gt = [False] * num_gts
    
    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(-pred_scores)
    
    for pred_idx in sorted_indices:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(num_gts):
            # Skip if already matched or different class
            if matched_gt[gt_idx] or pred_classes[pred_idx] != gt_classes[gt_idx]:
                continue
            
            iou = calculate_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives[pred_idx] = True
            matched_gt[best_gt_idx] = True
    
    return true_positives, matched_gt


def calculate_precision_recall(
    all_pred_boxes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_pred_classes: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    all_gt_classes: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate overall precision and recall across multiple images
    
    Args:
        all_pred_boxes: List of predicted boxes per image
        all_pred_scores: List of scores per image
        all_pred_classes: List of predicted classes per image
        all_gt_boxes: List of ground truth boxes per image
        all_gt_classes: List of ground truth classes per image
        num_classes: Number of object classes
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes in zip(
        all_pred_boxes, all_pred_scores, all_pred_classes, all_gt_boxes, all_gt_classes
    ):
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue
        
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        true_positives, matched_gt = match_detections(
            pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold
        )
        
        tp = sum(true_positives)
        fp = len(true_positives) - tp
        fn = len(matched_gt) - sum(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using 11-point interpolation
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
    
    Returns:
        Average Precision value
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def calculate_map(
    all_pred_boxes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_pred_classes: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    all_gt_classes: List[np.ndarray],
    num_classes: int,
    iou_thresholds: List[float] = [0.5, 0.75]
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) across multiple IoU thresholds
    
    Args:
        all_pred_boxes: List of predicted boxes per image
        all_pred_scores: List of scores per image
        all_pred_classes: List of predicted classes per image
        all_gt_boxes: List of ground truth boxes per image
        all_gt_classes: List of ground truth classes per image
        num_classes: Number of object classes
        iou_thresholds: List of IoU thresholds to evaluate
    
    Returns:
        Dictionary with mAP values for different thresholds
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        metrics = calculate_precision_recall(
            all_pred_boxes, all_pred_scores, all_pred_classes,
            all_gt_boxes, all_gt_classes, num_classes, iou_thresh
        )
        results[f'mAP@{iou_thresh}'] = metrics['precision']  # Simplified
        results[f'precision@{iou_thresh}'] = metrics['precision']
        results[f'recall@{iou_thresh}'] = metrics['recall']
    
    # Average across thresholds
    results['mAP'] = np.mean([results[f'mAP@{t}'] for t in iou_thresholds])
    
    return results


