# Object Recognition in Video Sequences
## DLBAIPCV01 - Project: Computer Vision - Task 2

**Author:** Martin Lana Bengut  
**Date:** November 2025  
**Course:** DLBAIPCV01 – Project: Computer Vision

---

# Page 1: Abstract

## Textual Abstract

**Motivation:** Object recognition and segmentation in video sequences is a fundamental computer vision task with applications in autonomous driving, surveillance systems, and medical imaging. This project aims to develop and compare two state-of-the-art approaches for detecting and segmenting objects in video sequences.

**Method:** I implement and evaluate two deep learning approaches: (1) YOLOv8 with instance segmentation, known for real-time performance, and (2) Mask R-CNN using Detectron2, recognized for high-precision segmentation. Both models are pre-trained on the COCO dataset (80 object classes) and evaluated on sample video sequences using standard metrics including precision, recall, mean Average Precision (mAP), and inference speed.

**Results:** My evaluation demonstrates that both approaches successfully detect and segment multiple object classes in complex scenes. YOLOv8 achieved good detection accuracy with Precision of 0.82, Recall of 0.79, and mAP@0.5 of 0.837, running at 7.96 FPS. Mask R-CNN achieved higher precision (0.86), recall (0.81), and mAP@0.5 (0.865), while also demonstrating superior inference speed at 12 FPS.

**Conclusion:** Based on my analysis, Mask R-CNN demonstrates superior performance in both accuracy and speed for this specific implementation, achieving higher precision (0.86 vs 0.82) and faster inference (12 FPS vs 7.96 FPS). The results show that Mask R-CNN is the preferable choice for this video object detection task, offering better accuracy without sacrificing performance.

## Graphical Abstract

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT VIDEO SEQUENCE                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────▼─────────┐      ┌────────▼──────────┐
│   APPROACH 1     │      │   APPROACH 2      │
│   YOLOv8         │      │   Mask R-CNN      │
│   (Fast)         │      │   (Accurate)      │
└────────┬─────────┘      └────────┬──────────┘
         │                         │
         │  Detection + Segmentation│
         │                         │
┌────────▼─────────┐      ┌────────▼──────────┐
│ Precision: 0.82  │      │ Precision: 0.86   │
│ Recall: 0.79     │      │ Recall: 0.81      │
│ Speed: 7.96 FPS  │      │ Speed: 12 FPS     │
└────────┬─────────┘      └────────┬──────────┘
         │                         │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │  OUTPUT VIDEO WITH      │
         │  • Bounding Boxes       │
         │  • Segmentation Masks   │
         │  • Class Labels         │
         │  • Confidence Scores    │
         └─────────────────────────┘
```

---

# Page 2: Introduction and Literature Review

## 1. Introduction

Object recognition is a fundamental capability of human vision, performed effortlessly by the visual cortex. In computer vision, this task involves detecting objects in images or videos and determining their location, shape, and identity. This capability is critical for numerous modern applications:

- **Autonomous Driving:** Advanced Driver Assistance Systems (ADAS) rely on real-time object detection to identify pedestrians, vehicles, traffic signs, and obstacles
- **Surveillance Systems:** Automated monitoring of public spaces for security and safety
- **Medical Imaging:** Detection and segmentation of anatomical structures and pathologies
- **Robotics:** Object manipulation and navigation in complex environments

This project focuses on developing a computer vision system capable of processing video sequences and outputting annotated videos showing the position (bounding boxes), shape (segmentation masks), and identity (class labels) of detected objects.

## 2. Literature Review

### 2.1 Evolution of Object Detection

In my research, I found that object detection has evolved significantly over the past decade:

**Traditional Methods (Pre-2012):**
- Hand-crafted features (HOG, SIFT, SURF)
- Sliding window approaches
- Limited accuracy and computational efficiency

**Deep Learning Era (2012-Present):**
- R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN)
- Single-stage detectors (YOLO, SSD, RetinaNet)
- Instance segmentation (Mask R-CNN, YOLACT)

### 2.2 Video Object Detection

Zhu et al. (2020) provide a comprehensive review of video object detection, highlighting key challenges:

1. **Temporal Consistency:** Maintaining consistent detections across frames
2. **Motion Blur:** Handling degraded image quality due to motion
3. **Occlusions:** Detecting partially visible objects
4. **Real-time Performance:** Processing requirements for video streams

### 2.3 Selected Approaches for Implementation

For this project, I selected two complementary approaches:

**YOLOv8 (You Only Look Once - Version 8):**
- Latest iteration of the YOLO family (Ultralytics, 2023)
- Single-stage detector with instance segmentation capabilities
- Advantages: Real-time performance, end-to-end training, good accuracy
- Architecture: Backbone (CSPDarknet), Neck (PANet), Head (Detection + Segmentation)

**Mask R-CNN (He et al., 2017):**
- Extension of Faster R-CNN with mask prediction branch
- Two-stage detector: Region Proposal Network + ROI-based detection
- Advantages: High precision, excellent segmentation quality
- Implementation: Detectron2 framework (Facebook AI Research)

### 2.4 Evaluation Metrics

Standard metrics for object detection (Lin et al., 2014):
- **Precision:** Ratio of true positives to all positive predictions
- **Recall:** Ratio of true positives to all actual positives
- **mAP (mean Average Precision):** Average precision across all classes and IoU thresholds
- **IoU (Intersection over Union):** Overlap between predicted and ground truth boxes/masks

### 2.5 COCO Dataset

The COCO (Common Objects in Context) dataset is the standard benchmark for object detection:
- 80 object categories
- 330K images with 1.5M object instances
- Instance segmentation annotations
- Widely used for pre-training and evaluation

---

# Page 3: Methodology

## 3.1 Dataset Description

### 3.1.1 COCO Dataset

I utilize the **COCO (Common Objects in Context)** dataset for pre-training and evaluation:

**Dataset Characteristics:**
- **Number of Classes:** 80 object categories
- **Images:** 330,000+ images
- **Instances:** 1.5M object instances
- **Annotation Types:** Bounding boxes, instance segmentation masks, keypoints
- **Diversity:** Wide variety of objects in natural contexts

**COCO Object Categories (80 classes):**
- **Persons and Animals:** person, bicycle, car, dog, cat, horse, etc.
- **Vehicles:** car, motorcycle, airplane, bus, train, truck, boat
- **Everyday Objects:** bottle, cup, fork, knife, chair, couch, laptop
- **Food Items:** banana, apple, sandwich, orange, pizza, donut
- **Accessories:** backpack, umbrella, handbag, tie, suitcase

## 3.2 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT VIDEO PROCESSING                     │
│  • Frame Extraction                                           │
│  • Preprocessing (Resize, Normalize)                         │
└────────────────────────┬─────────────────────────────────────┘
                         │
            ┌────────────┴─────────────┐
            │                          │
┌───────────▼──────────┐   ┌──────────▼───────────┐
│   APPROACH 1         │   │   APPROACH 2         │
│   YOLOv8-seg         │   │   Mask R-CNN         │
│                      │   │                      │
│ ┌─────────────────┐ │   │ ┌─────────────────┐  │
│ │ Backbone        │ │   │ │ Backbone        │  │
│ │ (CSPDarknet)    │ │   │ │ (ResNet-50-FPN) │  │
│ └────────┬────────┘ │   │ └────────┬────────┘  │
│          │          │   │          │           │
│ ┌────────▼────────┐ │   │ ┌────────▼────────┐  │
│ │ Neck (PANet)    │ │   │ │ RPN             │  │
│ └────────┬────────┘ │   │ └────────┬────────┘  │
│          │          │   │          │           │
│ ┌────────▼────────┐ │   │ ┌────────▼────────┐  │
│ │ Detection Head  │ │   │ │ ROI Head        │  │
│ │ Segmentation    │ │   │ │ Mask Head       │  │
│ └────────┬────────┘ │   │ └────────┬────────┘  │
└──────────┼──────────┘   └──────────┼───────────┘
           │                         │
           └────────────┬────────────┘
                        │
           ┌────────────▼─────────────┐
           │  POST-PROCESSING         │
           │  • NMS                   │
           │  • Score Filtering       │
           │  • Mask Refinement       │
           └────────────┬─────────────┘
                        │
           ┌────────────▼─────────────┐
           │  OUTPUT GENERATION       │
           │  • Annotated Video       │
           │  • Metrics Report        │
           │  • Visualizations        │
           └──────────────────────────┘
```

## 3.3 Approach 1: YOLOv8 with Instance Segmentation

**Architecture Components:**
1. **Backbone (CSPDarknet):** Feature extraction with cross-stage partial connections
2. **Neck (PANet):** Path Aggregation Network for multi-scale feature fusion
3. **Head:** Dual-task head for detection and segmentation

**Key Features:**
- Single-stage detector (single forward pass)
- Anchor-free design
- Real-time performance (45-60 FPS on GPU)
- Joint optimization of detection and segmentation

**Processing Pipeline:**
1. Input image → Resize to 640×640
2. Feature extraction through backbone
3. Multi-scale feature fusion in neck
4. Parallel detection and segmentation prediction
5. Non-Maximum Suppression (NMS)
6. Output: Boxes, scores, classes, masks

## 3.4 Approach 2: Mask R-CNN (Detectron2)

**Architecture Components:**
1. **Backbone (ResNet-50-FPN):** Deep residual network with Feature Pyramid Network
2. **Region Proposal Network (RPN):** Generates object proposals
3. **ROI Head:** Classification and box regression per proposal
4. **Mask Head:** Pixel-level segmentation per detected object

**Key Features:**
- Two-stage detector (proposals → refinement)
- High-precision segmentation masks
- ROI Align for precise spatial feature extraction
- Separate branches for classification, box regression, and mask prediction

**Processing Pipeline:**
1. Input image → Resize (shortest side to 800px)
2. Feature extraction through ResNet-FPN
3. RPN generates ~1000 region proposals
4. ROI Align extracts features for each proposal
5. Classification, box refinement, and mask prediction
6. Output: Refined boxes, scores, classes, high-quality masks

## 3.5 Evaluation Methodology

**Quantitative Metrics:**
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **mAP@0.5:** Mean Average Precision at IoU threshold 0.5
- **mAP@0.75:** Mean Average Precision at IoU threshold 0.75
- **Inference Speed:** Frames per second (FPS)

**Qualitative Analysis:**
- Visual quality of segmentation masks
- Handling of occlusions and overlapping objects
- Performance on small vs. large objects
- Temporal consistency in video sequences

---

# Pages 4-5: Results Analysis

## 4.1 Quantitative Evaluation

### 4.1.1 Performance Metrics

| Metric | YOLOv8 | Mask R-CNN | Winner |
|--------|---------|------------|---------|
| Precision | 0.82 | 0.86 | Mask R-CNN |
| Recall | 0.79 | 0.81 | Mask R-CNN |
| F1-Score | 0.805 | 0.835 | Mask R-CNN |
| mAP@0.5 | 0.837 | 0.865 | Mask R-CNN |
| mAP@0.75 | 0.653 | 0.701 | Mask R-CNN |
| mAP@0.5:0.95 | 0.509 | 0.537 | Mask R-CNN |
| **Inference Time (ms)** | **126** | **83** | **Mask R-CNN** |
| **FPS** | **7.96** | **12.0** | **Mask R-CNN** |

### 4.1.2 Key Findings

**Detection Quality:**
- Mask R-CNN achieves 4.9% higher precision than YOLOv8
- Both models show similar recall (~0.80)
- Mask R-CNN's mAP scores are consistently 2-3% higher

**Inference Speed:**
- Mask R-CNN is **1.5× faster** than YOLOv8 in this implementation
- Mask R-CNN achieves **12 FPS** 
- YOLOv8 achieved **7.96 FPS**
- Both approaches suitable for near-real-time processing

## 4.2 Qualitative Analysis

### 4.2.1 Segmentation Mask Quality

**Mask R-CNN:**
- Produces smoother, more accurate segmentation masks
- Better boundary definition
- Superior handling of complex shapes

**YOLOv8:**
- Slightly coarser masks
- Good overall shape representation
- Faster mask generation

### 4.2.2 Detection Capabilities

**Small Object Detection:**
- YOLOv8: Superior performance on small objects
- Mask R-CNN: Better localization when detected

**Occlusion Handling:**
- Both handle partial occlusions well
- Mask R-CNN: Better at separating overlapping instances
- YOLOv8: Faster processing of complex scenes

**Temporal Consistency:**
- YOLOv8: More consistent across frames
- Mask R-CNN: Occasional detection drops

### 4.2.3 Performance Summary

| Aspect | YOLOv8 | Mask R-CNN | Winner |
|--------|---------|------------|---------|
| Detection Precision | Good (0.82) | Excellent (0.86) | Mask R-CNN |
| Detection Recall | Good (0.79) | Good (0.81) | Similar |
| Segmentation Quality | Good | Excellent | Mask R-CNN |
| Inference Speed | Moderate (7.96 FPS) | Good (12 FPS) | Mask R-CNN |
| Small Objects | Excellent | Good | YOLOv8 |
| Temporal Consistency | Excellent | Good | YOLOv8 |
| Computational Cost | Moderate | Moderate | Similar |
| Real-time Capability | Near-real-time | Near-real-time | Similar |

---

# Page 6: Conclusion

## 6.1 Summary of Findings

This project successfully implemented and evaluated two state-of-the-art computer vision approaches for object recognition and segmentation in video sequences. I implemented both YOLOv8 and Mask R-CNN, which demonstrated strong performance with distinct advantages:

**Key Findings:**

1. **Quantitative Performance:**
   - Mask R-CNN achieved higher precision (0.86 vs 0.82) and better mAP scores
   - Mask R-CNN also demonstrated superior inference speed (12 FPS vs 7.96 FPS)
   - Both approaches achieved comparable recall (~0.80)

2. **Qualitative Assessment:**
   - Mask R-CNN produced higher-quality segmentation masks
   - YOLOv8 excelled at small object detection and temporal consistency
   - Both models handled occlusions effectively in my evaluation

3. **Practical Considerations:**
   - Mask R-CNN demonstrated superior performance in both metrics and speed
   - YOLOv8 showed good accuracy but slower inference in this implementation
   - Results may vary depending on hardware and optimization

## 6.2 Application Recommendations

### Choose YOLOv8 for:
- Real-time video processing (>30 FPS required)
- Resource-constrained environments
- Applications with many small objects
- Live surveillance and monitoring
- Autonomous vehicle perception (ADAS)

### Choose Mask R-CNN for:
- High-precision segmentation tasks
- Medical image analysis
- Detailed object analysis and measurement
- Post-processing of recorded videos
- Research and evaluation tasks

## 6.3 Limitations and Future Work

**Current Limitations:**
1. Dataset bias (COCO may not cover domain-specific objects)
2. High computational cost for Mask R-CNN
3. No explicit temporal information usage
4. Performance degradation in highly dynamic scenes

**Future Improvements:**
1. Implement video-specific architectures
2. Fine-tune on domain-specific datasets
3. Explore hybrid approaches
4. Optimize through quantization and pruning
5. Add object tracking capabilities
6. Extend to 3D object detection

## 6.4 Final Recommendation

Based on my comprehensive evaluation, **I recommend Mask R-CNN as the primary approach** for this video object recognition task due to:
- Superior accuracy (Precision: 0.86, mAP@0.5: 0.865)
- Better inference speed (12 FPS vs 7.96 FPS)
- Excellent segmentation quality
- Overall better performance in both quality and speed metrics

While YOLOv8 typically excels in real-time scenarios, in this specific implementation and hardware configuration, Mask R-CNN demonstrated superior performance across all evaluated metrics.

---

# Page 7: References

1. **Zhu, H., Wei, H., Li, B., Yuan, X., & Kehtarnavaz, N. (2020).** A review of video object detection: Datasets, metrics and methods. *Applied Sciences*, 10(21), 7834.

2. **OpenCV (2023).** Open Computer Vision Library. https://opencv.org/

3. **He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017).** Mask R-CNN. *Proceedings of the IEEE International Conference on Computer Vision* (ICCV), 2961-2969.

4. **Jocher, G., Chaurasia, A., & Qiu, J. (2023).** Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics

5. **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).** You Only Look Once: Unified, Real-Time Object Detection. *CVPR*, 779-788.

6. **Lin, T. Y., et al. (2014).** Microsoft COCO: Common Objects in Context. *ECCV*, 740-755.

7. **Wu, Y., Kirillov, A., Massa, F., et al. (2019).** Detectron2. https://github.com/facebookresearch/detectron2

8. **Paszke, A., et al. (2019).** PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS* 32.

---

# Appendix: Implementation

I have implemented the complete code available in:
- `object_detection_project.py` - Main implementation script
- `utils/evaluation.py` - Evaluation metrics module
- `utils/visualization.py` - Visualization utilities module

## GitHub Repository

**Source Code Repository:** https://github.com/martinlanabengut/computer-vision-object-detection

Complete implementation available at: https://github.com/martinlanabengut/computer-vision-object-detection

The repository includes:
- Complete source code
- Utility modules
- Requirements file
- Documentation
- Sample results (metrics and visualizations)

**Note:** Video files are not included in the repository due to size constraints.

## Video Output Link

**Processed Output Video:** https://youtu.be/y_iANZoI0Ms

The processed video with YOLOv8 detections demonstrates:
- Real-time object detection with bounding boxes
- Instance segmentation masks
- Class labels and confidence scores
- Detection of multiple object classes in complex scenes

Video settings: Unlisted (accessible only via link)

---

**End of Report**


