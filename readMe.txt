# Hybrid FER (Facial Emotion Recognition)

A real-time, low-resolution **Facial Emotion Recognition (FER)** application engineered for speed and accuracy. By combining the **LiExNet** architecture with **Mini-Xception** and utilizing **YOLOv8** for rapid face detection, this project provides a modular, end-to-end deep learning pipeline. It unifies the **FER2013**, **CK+**, and **AffectNet** datasets and accelerates data loading through disk-cache (`pickle`) optimization.

---

## Key Features

* **Hybrid Architecture:** Integrates **LiExNet** and **Mini-Xception** to maintain high classification performance on low-resolution facial inputs.
* **Real-Time Face Detection:** Uses **YOLOv8** (`yolov8n.pt`) for low-latency facial bounding box extraction.
* **Unified Multi-Dataset Pipeline:** Seamlessly merges **FER2013**, **CK+**, and **AffectNet** for improved generalization across diverse lighting, angles, and expressions.
* **Disk-Cache Optimization:** Preprocesses and caches dataset tensors using `pickle` to eliminate redundant I/O bottlenecks during training.
* **Standalone Compilation:** Ready for packaging into a standalone executable via **Nuitka** for easy deployment without requiring a pre-configured Python environment.

---

## 1. Installation

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
