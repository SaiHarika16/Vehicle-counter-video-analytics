# Vehicle Counter & Multi-Model Video Analytics

## Advanced Vehicle Detection, Tracking & Counting

This repository presents a modular, extensible framework for vehicle analytics in video streams, leveraging state-of-the-art deep learning models and multi-object tracking algorithms. Designed for researchers, engineers, and practitioners, it enables robust vehicle detection, unique ID assignment, and accurate counting across diverse scenarios.

---

## Project Structure

```
models/           # Pre-trained model weights (SSD MobileNet)
outputs/          # Output videos with annotated detections and counts
video data/       # Input video files for inference
Yolov5.ipynb      # YOLOv5-based vehicle detection and line-crossing counter
RCNN.ipynb        # Faster R-CNN-based vehicle detection and unique object counter
Vehicle Tracking with DeepSORT and SSD MobileNet.ipynb  # SSD MobileNet + DeepSORT multi-object tracking
```

---

## Implementation Overview

### 1. SSD MobileNet + DeepSORT
- **Notebook:** `Vehicle Tracking with DeepSORT and SSD MobileNet.ipynb`
- **Pipeline:**
  1. **Detection:** Utilizes a pre-trained SSD MobileNet (Caffe) for real-time object detection, focusing on vehicle classes.
  2. **Tracking:** Integrates DeepSORT, a high-performance multi-object tracker, to assign persistent IDs and track vehicles across frames, even during occlusions.
  3. **Counting:** Maintains a set of unique IDs to ensure each vehicle is counted only once.
- **Key Features:**
  - Robust to occlusion and re-identification
  - Real-time performance (optimized for GPU)
  - Easily adaptable to other object classes
- **Dependencies:** `opencv-python`, `numpy`, `deep-sort-realtime`
- **Setup:**
  - Place `MobileNetSSD_deploy.caffemodel` and `MobileNetSSD_deploy.prototxt` in `models/`.
  - Input video in `video data/` or root.
  - Run the notebook; output is saved as `output_deepsort.mp4` with bounding boxes, unique IDs, and total count overlayed.

### 2. YOLOv5
- **Notebook:** `Yolov5.ipynb`
- **Pipeline:**
  1. **Detection:** Loads YOLOv5s via PyTorch Hub for high-accuracy, real-time vehicle detection.
  2. **Tracking/Counting:** Implements a centroid-based tracker to assign IDs and count vehicles as they cross a virtual line (line-crossing algorithm).
- **Key Features:**
  - Fast, accurate detection with minimal setup
  - Easily configurable for different vehicle types or custom classes
  - Simple, effective tracking for counting applications
- **Dependencies:** `torch`, `opencv-python`, `numpy`, `pandas`
- **Setup:**
  - The notebook clones the YOLOv5 repo and installs requirements automatically.
  - Place your video in `video data/` or root.
  - Run the notebook; output is `output_counted.mp4` with annotated detections and live count.

### 3. Faster R-CNN
- **Notebook:** `RCNN.ipynb`
- **Pipeline:**
  1. **Detection:** Uses PyTorch's pre-trained Faster R-CNN (ResNet-50 FPN) for high-quality object detection.
  2. **Tracking/Counting:** Employs a distance-based centroid tracker to count unique vehicles, minimizing double-counting.
- **Key Features:**
  - High detection accuracy, suitable for challenging scenes
  - Modular code for easy extension to other COCO classes
  - GPU acceleration supported
- **Dependencies:** `torch`, `torchvision`, `opencv-python`, `numpy`
- **Setup:**
  - Place your video in `video data/` or root.
  - Run the notebook; output is `output.mp4` with bounding boxes and total vehicle count.

---

## Workflow

1. Select your preferred detection/tracking notebook.
2. Install dependencies (see the first cell of each notebook for pip commands).
3. Download or place required model weights in the `models/` directory (for SSD MobileNet).
4. Add your input video to `video data/` or the project root.
5. Run the notebook in Jupyter or Google Colab.
6. Review output videos in the `outputs/` folder, featuring annotated detections, unique IDs, and vehicle counts.

---

## Modularity & Extensibility
- Plug-and-play architecture: Each notebook is self-contained and can be extended or integrated into larger analytics pipelines.
- Customizable classes: Easily adapt detection to other object types by modifying class lists.
- Scalable: Designed for both research prototyping and production deployment (with further engineering).

---

## Sample Data
- Input videos: Provided in `video data/` (e.g., `car.mp4`, `videoplayback.mp4`).
- Output videos: Annotated results in `outputs/` (e.g., `output_deepsort.mp4`, `output_counted.mp4`, `output.mp4`).

---

## Technical Highlights
- DeepSORT: Combines appearance descriptors and motion for robust multi-object tracking.
- YOLOv5: State-of-the-art, real-time object detector with PyTorch backend.
- Faster R-CNN: High-accuracy, region-based detector for challenging scenes.
- OpenCV Integration: Efficient video I/O and annotation.
- GPU Acceleration: All models support CUDA for fast inference.

---

## Getting Started
1. Clone this repository and open in Jupyter or Colab.
2. Install dependencies as specified in each notebook.
3. Download model weights (see `models/` directory).
4. Run the desired notebook and follow the cell-by-cell instructions.

---

## Notes
- For optimal performance, use a GPU-enabled environment (e.g., Google Colab, local CUDA machine).
- You can use your own videos by placing them in `video data/` and updating the input path in the notebooks.
- Output videos are saved in the working directory or `outputs/` folder.
- Each notebook is designed for clarity and ease of modification for research or production.

---

## References
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [Deep SORT](https://github.com/nwojke/deep_sort)
- [PyTorch Detection Models](https://pytorch.org/vision/stable/models.html)
- [OpenCV](https://opencv.org/) 