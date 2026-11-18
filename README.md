# ⚡ PixelLab – Image Processing Studio

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8.svg)](https://opencv.org/)
[![Pillow](https://img.shields.io/badge/library-Pillow-yellowgreen.svg)](https://python-pillow.org/)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()

> A modern, desktop-grade web application for digital image processing, computer vision enhancements, format conversions, compression analytics, and interactive visual analysis. Designed with a clean, dark-themed UI inspired by Linear and Raycast.

---

## 📖 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Technology Stack](#3-technology-stack)
4. [Architecture & Folder Structure](#4-architecture--folder-structure)
5. [Installation & Setup](#5-installation--setup)
6. [How to Run PixelLab](#6-how-to-run-pixellab)
7. [Running Automated Tests](#7-running-automated-tests)
8. [Image Processing Concepts Explained (Beginner Friendly)](#8-image-processing-concepts-explained)
9. [Module Reference Guide](#9-module-reference-guide)
10. [Troubleshooting & FAQs](#10-troubleshooting--faqs)

---

## 1. Project Overview

**PixelLab** is a full-featured image processing studio built entirely in Python. Unlike basic mockups or prototypes, **every single control in PixelLab performs real mathematical image manipulations** on pixel arrays using OpenCV, Pillow, and NumPy.

Whether you need to enhance architectural photos with CLAHE (Contrast Limited Adaptive Histogram Equalization), convert high-resolution transparent PNGs to web-optimized WEBP files, detect structural contours using multi-stage Canny filters, or inspect RGB color histograms, PixelLab provides an intuitive, high-performance studio interface.

---

## 2. Key Features

### 🖼️ 1. Image Ingestion & Validation
- Supports **JPG, JPEG, PNG, and WEBP**.
- Deep file integrity verification: detects corrupted, truncated, or invalid non-image byte streams before loading.
- Extracts metadata: dimensions, aspect ratio, color mode, channel count, megapixel resolution, and estimated in-memory footprint.
- Built-in sample images for instant 1-click testing (Nature Sunset, Geometric Architecture, Studio Colors).

### 🎨 2. Creative Image Filters
Every filter comes with interactive intensity or parameter sliders:
1. **Grayscale**: Luminosity-weighted desaturation ($Y = 0.299R + 0.587G + 0.114B$) with variable blend strength.
2. **Sepia**: Vintage silver-gelatin photographic color transform matrix.
3. **Black & White**: High-contrast binary binarization with adjustable cutoff threshold ($0 - 255$).
4. **Vintage Film**: Warm golden highlights, elevated faded shadows, and subtle radial vignette.
5. **Gaussian Blur**: Bell-curve low-pass convolution with odd kernel radius ($1 - 51\text{ px}$).
6. **Median Blur**: Non-linear impulse noise filter ($1 - 31\text{ px}$ aperture).
7. **Sharpen**: High-pass 2D spatial convolution kernel with intensity multiplier.
8. **Emboss**: Directional relief filter with $+128$ neutral gray offset for 3D illusion.
9. **Invert Colors**: Channel negation ($255 - X$) creating true photo negatives.
10. **Edge Detection Filter**: High-pass Laplacian contour highlight.

### 🎚️ 3. Tonal Adjustments
- **Brightness**: Linear scalar factor ($0.1\times - 2.5\times$).
- **Contrast**: Dynamic distance expansion from mean gray level ($0.1\times - 2.5\times$).
- **Saturation**: Color vibrancy scaling ($0.0\times = \text{grayscale}, 2.5\times = \text{vivid}$).
- **Sharpness**: Edge frequency amplification ($0.0\times - 3.0\times$).
- **Gamma Correction**: Power-law non-linear tone adjustment ($0.2 - 2.5$) with 256-bin lookup tables.

### 📐 4. Geometric Transformations
- **Resize**: Free resizing or proportional aspect ratio locking using Lanczos sinc interpolation.
- **Crop**: Precision pixel coordinate boundary clipping ($X_1, Y_1, X_2, Y_2$).
- **Fixed Rotations**: Instant 90° Clockwise, 180°, and 270° Clockwise.
- **Custom Rotation**: Smooth arbitrary angle rotation ($-180^\circ \text{ to } +180^\circ$) with optional canvas auto-expansion.
- **Flips**: Horizontal mirror and vertical flip.

### ✨ 5. Computer Vision Enhancement
- **Unsharp Masking (USM)**: Subtracts a Gaussian low-pass copy to amplify micro-textures without halo artifacts.
- **Bilateral Noise Reduction**: Edge-preserving filter combining spatial distance with color similarity to remove grain while keeping edges sharp.
- **Gaussian Smoothing**: Clean frequency smoothing.
- **Histogram Equalization**: Converts image to YCrCb space, equalizing only the Luminance (Y) channel to preserve natural color balances.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Divides image into local tiles ($8\times 8$), clips peak frequencies, and redistributes contrast without amplifying background noise.

### 🔍 6. Edge Detection
- **Canny Edge Detector**: Multi-stage algorithm featuring Gaussian pre-filtering, Sobel gradients, non-maximum suppression, and dual hysteresis thresholds ($T_1, T_2$).
- **Sobel X**: Computes horizontal partial derivative ($\partial I / \partial x$) detecting vertical edges.
- **Sobel Y**: Computes vertical partial derivative ($\partial I / \partial y$) detecting horizontal edges.
- **Sobel Combined**: True Euclidean gradient magnitude ($\sqrt{G_x^2 + G_y^2}$) detecting all orientations.
- **Laplacian**: Second-order spatial derivative ($\nabla^2 I$) highlighting zero-crossings.

### 📈 7. Image Analysis & Profiling
- **Overlaid RGB Histogram**: Dark-themed curves showing Red, Green, and Blue distribution.
- **Separate Channel Subplots**: 3 individual subplots for R, G, B channels.
- **Grayscale / Luminance Histogram**: Single-curve distribution with calculated mean luminance marker.
- **Statistical Dashboard**: Mean, standard deviation, min, max, and median per channel.

### 💾 8. Compression & Format Conversion
- Formats: **PNG, JPEG, and WEBP**.
- **Transparency Safety**: Transparent PNGs converted to JPEG are safely flattened onto a clean white background, avoiding ugly black artifacts.
- **Quality-Tuned Lossy Compression**: Adjust quality slider ($5 - 100$) for JPEG and WEBP.
- **Size Comparison**: Displays original size vs compressed size in human-readable units (KB/MB).
- **Accurate Reduction Analytics**: Calculates exact percentage saved. If high quality or lossless export produces a larger file, PixelLab accurately informs you of the size increase rather than making false claims.

### 🏷️ 9. Watermark Studio
- Custom watermark text overlay.
- Font size adjustment ($12 - 120\text{ px}$).
- Alpha opacity control ($0.05 - 1.0$).
- Standard presets (Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center) or Custom $X, Y$ coordinates.
- Color picker (Hex / RGB).

---

## 3. Technology Stack

| Layer | Library / Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Clean, readable, beginner-friendly core language |
| **Image Processing** | OpenCV (`cv2`) | Fast C++ backed computer vision algorithms (Canny, Sobel, CLAHE, Bilateral) |
| **Image Manipulation** | Pillow (`PIL`) | High-level image loading, color mode transformations, fonts, and geometry |
| **Numerical Array** | NumPy | High-performance multi-dimensional array operations and LUT transforms |
| **Data Visualization** | Matplotlib | Thread-safe, non-interactive Agg rendering of tonal histograms |
| **UI Framework** | Streamlit | Responsive reactive web studio layout |
| **Testing** | Pytest | Automated verification of processing pipelines and edge cases |

---

## 4. Architecture & Folder Structure

```
d:\imageProcessing\
│
├── app.py                      # Main Streamlit web application & UI navigation
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive documentation
│
├── modules/                    # Clean modular image processing algorithms
│   ├── __init__.py             # Package marker
│   ├── image_utils.py          # Validation, BGR <-> RGB conversions, metadata, bytes I/O
│   ├── filters.py              # 10 Creative visual filters
│   ├── adjustments.py          # Brightness, contrast, saturation, sharpness, gamma
│   ├── transformations.py      # Resize, crop, rotate, flip
│   ├── enhancement.py          # USM sharpening, bilateral denoising, CLAHE, histogram eq
│   ├── edge_detection.py       # Canny, Sobel X/Y/Combined, Laplacian
│   ├── analysis.py             # Matplotlib RGB & Grayscale histogram plots and stats
│   ├── compression.py          # PNG/JPEG/WEBP conversion and compression analytics
│   └── watermark.py            # Alpha-blended text watermarking
│
├── tests/                      # Automated unit test suite
│   ├── __init__.py             # Package marker
│   └── test_processing.py      # 25 automated pytest tests
│
└── assets/                     # Media assets & sample demo images
    ├── README.md
    └── sample_images/          # Built-in sample images for immediate testing
        ├── nature_sunset.jpg
        ├── geometric_shapes.png
        └── studio_colors.jpg
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.11 or higher installed on your machine.
- Verify your Python installation in your terminal:
  ```powershell
  python --version
  ```

### Step 1: Clone or Navigate to the Workspace
```powershell
cd d:\imageProcessing
```

### Step 2: (Recommended) Create and Activate a Virtual Environment
Using a virtual environment isolates dependencies:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
Run:
```powershell
pip install -r requirements.txt
```

Or install directly:
```powershell
pip install streamlit opencv-python pillow numpy matplotlib pytest
```

---

## 6. How to Run PixelLab

To launch PixelLab in your default web browser, run:

```powershell
streamlit run app.py
```

The application will start immediately, usually accessible at:
- **Local URL**: `http://localhost:8501`
- **Network URL**: `http://<your-local-ip>:8501`

---

## 7. Running Automated Tests

PixelLab includes a comprehensive test suite covering image validation, filter operations, arithmetic adjustments, geometry, edge detectors, format conversions, and edge cases.

To execute all tests with detailed verbose output:

```powershell
pytest -v tests/test_processing.py
```

Expected output:
```text
tests/test_processing.py::test_validate_image_file_valid PASSED          [  4%]
tests/test_processing.py::test_validate_image_file_invalid PASSED        [  8%]
tests/test_processing.py::test_format_bytes PASSED                       [ 12%]
tests/test_processing.py::test_calculate_aspect_ratio PASSED             [ 16%]
tests/test_processing.py::test_pil_cv2_roundtrip PASSED                  [ 20%]
tests/test_processing.py::test_grayscale_conversion PASSED               [ 24%]
tests/test_processing.py::test_sepia_filter PASSED                       [ 28%]
tests/test_processing.py::test_black_and_white PASSED                    [ 32%]
tests/test_processing.py::test_blur_and_sharpen_filters PASSED           [ 36%]
tests/test_processing.py::test_invert_colors PASSED                      [ 40%]
tests/test_processing.py::test_brightness_adjustment PASSED              [ 44%]
tests/test_processing.py::test_contrast_adjustment PASSED                [ 48%]
tests/test_processing.py::test_adjustments_pipeline PASSED               [ 52%]
tests/test_processing.py::test_resize_image PASSED                       [ 56%]
tests/test_processing.py::test_resize_invalid_dimensions PASSED          [ 60%]
tests/test_processing.py::test_crop_image PASSED                         [ 64%]
tests/test_processing.py::test_crop_invalid_bounds PASSED                [ 68%]
tests/test_processing.py::test_rotations PASSED                          [ 72%]
tests/test_processing.py::test_flips PASSED                              [ 76%]
tests/test_processing.py::test_canny_edge_detection PASSED               [ 80%]
tests/test_processing.py::test_sobel_and_laplacian PASSED                [ 84%]
tests/test_processing.py::test_enhancement_algorithms PASSED             [ 88%]
tests/test_processing.py::test_convert_format_and_transparency PASSED    [ 92%]
tests/test_processing.py::test_compression_reduction_calculation PASSED  [ 96%]
tests/test_processing.py::test_watermark PASSED                          [100%]

============================= 25 passed in 0.29s ==============================
```

---

## 8. Image Processing Concepts Explained

### 1. RGB vs. BGR Color Channels
- **Pillow** and **Streamlit** store images in **RGB** format: `[Red, Green, Blue]`.
- **OpenCV** by default stores images in **BGR** format: `[Blue, Green, Red]`.
- *Why does this matter?* If you pass a Pillow image directly into OpenCV without converting RGB to BGR, the red and blue channels swap, turning skies orange and skin tones blue. PixelLab handles this safely in `modules/image_utils.py` through `cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)`.

### 2. Spatial Convolution & Kernels
Filters like Sharpen, Emboss, and Blurs use a **kernel**—a small matrix of numbers (e.g., $3 \times 3$). The kernel slides across every pixel in the image:
```
[ -1, -1, -1 ]
[ -1,  9, -1 ]  <-- Sharpening Kernel
[ -1, -1, -1 ]
```
The central pixel is multiplied by $9$, and its neighbors are subtracted. This amplifies sharp differences between adjacent pixels.

### 3. Gamma Correction ($\gamma$)
Human eyes do not perceive light linearly; we are much more sensitive to differences in dark tones than in bright tones. Gamma correction applies a power curve:
$$\text{Output} = 255 \times \left( \frac{\text{Input}}{255} \right)^{\frac{1}{\gamma}}$$
- When $\gamma > 1.0$, shadow details brighten without overexposing highlights.
- When $\gamma < 1.0$, midtones darken smoothly.

### 4. Canny Edge Detection & Hysteresis
Canny uses two thresholds ($T_1$ and $T_2$):
- Any edge gradient $> T_2$ is definitely an edge.
- Any gradient $< T_1$ is definitely discarded.
- Gradients between $T_1$ and $T_2$ are retained **only if they connect** to a confirmed edge pixel. This removes random noise while preserving unbroken object borders.

### 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Global histogram equalization can blow out highlights and magnify background sensor grain. CLAHE divides the image into an $8 \times 8$ grid of localized tiles, equalizes contrast locally, and clips any histogram bin exceeding `clipLimit`.

---

## 9. Module Reference Guide

- **`app.py`**: Controls Streamlit layout, session states (`original_image`, `processed_image`), sidebars, image uploaders, and tabs.
- **`modules/image_utils.py`**: Converts between PIL images and OpenCV numpy arrays, handles transparency, validates file types, and computes aspect ratios.
- **`modules/filters.py`**: Pure functions taking a PIL Image and returning a filtered PIL Image (Grayscale, Sepia, Vintage, Blur, Sharpen, Emboss, Invert).
- **`modules/adjustments.py`**: Slider-controlled adjustments for brightness, contrast, saturation, sharpness, and LUT-accelerated gamma.
- **`modules/transformations.py`**: Lanczos resizing, pixel bounding-box cropping, fixed rotations (90°, 180°, 270°), arbitrary angle rotation, and flips.
- **`modules/enhancement.py`**: Unsharp masking, OpenCV bilateral filtering, Gaussian smoothing, and LAB-space CLAHE.
- **`modules/edge_detection.py`**: Canny, Sobel X, Sobel Y, Sobel Magnitude, and Laplacian operators.
- **`modules/analysis.py`**: Statistical profiling and Matplotlib dark-themed histogram generation.
- **`modules/compression.py`**: Lossy/lossless compression, RGBA to JPEG transparency compositing, and reduction percentage calculations.
- **`modules/watermark.py`**: Text overlay rendering with TrueType font resolution and alpha blending.
- **`tests/test_processing.py`**: 25 automated pytest tests verifying all image operations with synthetic test images.

---

## 10. Troubleshooting & FAQs

### Q: Why does my converted JPEG have a black background?
**A:** Standard JPEG does not support transparency. If a PNG with an alpha channel is converted directly without compositing, transparent pixels can appear black. PixelLab automatically composites transparent pixels onto a clean white background before JPEG encoding.

### Q: Can I apply multiple effects sequentially (e.g. Grayscale + Sharpen + Watermark)?
**A:** Yes! In the sidebar, click the **"➕ Stack"** button. This commits your current processed image as the new working base, allowing you to chain as many operations as you wish.

### Q: Can I run this offline?
**A:** Yes! PixelLab has zero external API dependencies, requires no cloud accounts, and runs 100% locally on your computer.

---

## 📄 License
MIT License. Built with ❤️ for computer vision enthusiasts, designers, and Python developers.
