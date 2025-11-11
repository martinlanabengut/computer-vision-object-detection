"""
Visualization utilities for object detection and segmentation
Creates annotated images and videos with detection results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import pandas as pd


# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for different classes
    
    Args:
        num_colors: Number of colors to generate
    
    Returns:
        List of BGR color tuples
    """
    np.random.seed(42)
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors


COLORS = get_colors(80)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    masks: Optional[np.ndarray] = None,
    class_names: List[str] = COCO_CLASSES,
    score_threshold: float = 0.5,
    show_masks: bool = True,
    show_boxes: bool = True,
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes, masks, and labels on image
    
    Args:
        image: Input image (H, W, 3)
        boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format
        scores: Confidence scores (N,)
        classes: Class indices (N,)
        masks: Binary masks (N, H, W) or None
        class_names: List of class names
        score_threshold: Minimum score to display
        show_masks: Whether to show segmentation masks
        show_boxes: Whether to show bounding boxes
        show_labels: Whether to show labels
    
    Returns:
        Annotated image
    """
    result = image.copy()
    
    # Filter by score threshold
    valid_indices = scores >= score_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    classes = classes[valid_indices]
    if masks is not None:
        masks = masks[valid_indices]
    
    # Create overlay for masks
    if show_masks and masks is not None:
        overlay = result.copy()
        for i, (mask, cls_idx) in enumerate(zip(masks, classes)):
            color = COLORS[int(cls_idx) % len(COLORS)]
            
            # Resize mask to image size if needed
            if mask.shape != result.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), 
                                (result.shape[1], result.shape[0]), 
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Apply colored mask
            overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        
        result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
    
    # Draw bounding boxes and labels
    for i, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[int(cls_idx) % len(COLORS)]
        
        # Draw bounding box
        if show_boxes:
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels:
            label = f"{class_names[int(cls_idx)]}: {score:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle
            cv2.rectangle(
                result,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                result, label, (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
    
    return result


def create_comparison_plot(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Approach 1",
    title2: str = "Approach 2",
    save_path: Optional[str] = None
) -> None:
    """
    Create side-by-side comparison of two detection results
    
    Args:
        image1: First annotated image
        image2: Second annotated image
        title1: Title for first image
        title2: Title for second image
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Convert BGR to RGB for matplotlib
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title1, fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2, fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(
    metrics_dict: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create bar plot comparing metrics between approaches
    
    Args:
        metrics_dict: Dictionary with metrics for each approach
        save_path: Path to save the plot
    """
    df = pd.DataFrame(metrics_dict).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df.columns))
    width = 0.35
    
    for i, (approach, values) in enumerate(df.iterrows()):
        offset = width * (i - 0.5)
        ax.bar(x + offset, values, width, label=approach)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_metrics_table(
    metrics_dict: dict,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create formatted table of metrics
    
    Args:
        metrics_dict: Dictionary with metrics for each approach
        save_path: Path to save the table as image
    
    Returns:
        DataFrame with formatted metrics
    """
    df = pd.DataFrame(metrics_dict).T
    
    # Format values as percentages where appropriate
    for col in ['precision', 'recall', 'f1_score', 'mAP']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
    
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix for classifications
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def add_text_overlay(
    image: np.ndarray,
    text: str,
    position: str = 'top',
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Add text overlay to image
    
    Args:
        image: Input image
        text: Text to overlay
        position: Position ('top' or 'bottom')
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
    
    Returns:
        Image with text overlay
    """
    result = image.copy()
    h, w = result.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Determine position
    if position == 'top':
        y = text_height + 20
        rect_y1, rect_y2 = 0, text_height + 30
    else:
        y = h - 10
        rect_y1, rect_y2 = h - text_height - 30, h
    
    # Draw background rectangle
    cv2.rectangle(result, (0, rect_y1), (w, rect_y2), bg_color, -1)
    
    # Draw text centered
    x = (w - text_width) // 2
    cv2.putText(result, text, (x, y), font, font_scale, text_color, thickness)
    
    return result


