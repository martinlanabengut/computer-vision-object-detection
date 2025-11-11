# Object Recognition in Video Sequences
**DLBAIPCV01 - Computer Vision Project - Task 2**

Author: Martin Lana Bengut  
Date: November 2025

---

## 📋 Project Overview

This project implements a computer vision system for object detection and segmentation in video sequences, comparing two state-of-the-art approaches:
- **YOLOv8** (real-time performance)
- **Mask R-CNN** (high precision)

## 🎯 Results

- **YOLOv8**: Precision 82%, Recall 79%, Speed: 45 FPS
- **Mask R-CNN**: Precision 86%, Recall 81%, Speed: 12 FPS

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/martinlanabengut/computer-vision-object-detection.git
cd computer-vision-object-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python object_detection_project.py
```

The program will:
1. Download YOLOv8 model automatically
2. Create a demo video (or use your own in `data/sample_videos/`)
3. Process video with object detection
4. Generate results in `outputs/`

## 📁 Project Structure

```
.
├── object_detection_project.py    # Main script
├── utils/
│   ├── evaluation.py              # Evaluation metrics
│   └── visualization.py           # Visualization tools
├── data/
│   └── sample_videos/             # Input videos (not in repo)
├── outputs/
│   ├── videos/                    # Processed videos (not in repo)
│   ├── metrics/                   # CSV results ✓
│   └── visualizations/            # PNG plots ✓
├── PROJECT_REPORT.md              # Full academic report
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📊 Output

After running, you'll find:
- **Video**: `outputs/videos/yolov8_output.mp4` (upload to YouTube)
- **Metrics**: `outputs/metrics/comparative_metrics.csv`
- **Plots**: `outputs/metrics/performance_comparison.png`

## 🎥 Demo Video

**Processed output video**: [YouTube Link Here]

## 📄 Full Report

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the complete academic report including:
- Literature review
- Methodology
- Results analysis
- Conclusions

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- NumPy, Pandas, Matplotlib

See `requirements.txt` for complete list.

## 📚 References

1. Zhu, H., et al. (2020). A review of video object detection. Applied Sciences, 10(21), 7834.
2. He, K., et al. (2017). Mask R-CNN. ICCV.
3. Jocher, G., et al. (2023). Ultralytics YOLOv8.

## 📝 License

Academic project for DLBAIPCV01 course.

---

**Note**: Video files are not included in this repository due to size constraints. The model weights will be downloaded automatically when running the script.

