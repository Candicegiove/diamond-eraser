# diamond-eraser
A magic eraser project
# 💎 Diamond Magic Eraser

A high-fidelity batch image processing tool designed to detect and remove translucent diamond-shaped overlays and watermarks using **Navier-Stokes based image inpainting**.

This application provides a seamless "vibe-coding" interface for researchers and creators to normalize image datasets by removing consistent distracting artifacts without compromising the underlying visual data.

---

## ✨ Key Features

* **Batch Processing:** Upload dozens of images and process them simultaneously.
* **Surgical Precision:** Optional "Surgical Mode" to target specific regions (e.g., bottom-right corners) to prevent false positives in the main image body.
* **Interactive Comparison:** A built-in "Before & After" slider to fine-tune detection sensitivity and healing smoothness.
* **Neural Healing:** Utilizes OpenCV's `INPAINT_NS` (Navier-Stokes) algorithm to "flow" surrounding textures into the erased area for a natural finish.
* **One-Click Export:** Download your entire cleaned batch as a single compressed `.zip` file.

---

## 🛠️ Technical Implementation

The "Magic" behind the eraser relies on a multi-stage computer vision pipeline:

1.  **Luminance Thresholding:** Identifying high-frequency white/translucent shapes based on grayscale intensity.
2.  **Morphological Dilation:** Expanding the detection mask by a kernel-based radius to capture anti-aliased edges.
3.  **Navier-Stokes Inpainting:** Solving fluid dynamics equations to propagate image intensity from the boundaries into the masked region.



---

## 🚀 Quick Start

### 1. Local Setup
Ensure you have Python 3.8+ installed, then clone this repo and install dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/diamond-eraser.git](https://github.com/YOUR_USERNAME/diamond-eraser.git)
cd diamond-eraser
pip install -r requirements.txt
