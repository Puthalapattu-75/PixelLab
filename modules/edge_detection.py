"""
PixelLab - Edge Detection Module
Provides industry-standard edge detection operators:
- Canny Edge Detection (with hysteresis thresholds)
- Sobel X (vertical edge gradient)
- Sobel Y (horizontal edge gradient)
- Sobel Combined (gradient magnitude)
- Laplacian Edge Detection (second derivative)
"""

from typing import Tuple
from PIL import Image
import numpy as np
import cv2
from modules.image_utils import pil_to_cv2, cv2_to_pil, ensure_rgb


def detect_canny(
    image: Image.Image,
    threshold1: int = 100,
    threshold2: int = 200,
    aperture_size: int = 3,
) -> Image.Image:
    """
    Perform Canny Edge Detection.

    How it works:
    The Canny detector is a multi-stage edge detection algorithm:
    1. Smooths the image with a Gaussian filter to eliminate noise.
    2. Calculates intensity gradients using Sobel kernels.
    3. Performs Non-Maximum Suppression to thin out wide edges into single-pixel lines.
    4. Applies Hysteresis Thresholding:
       - Gradients > threshold2 are definitely edges.
       - Gradients < threshold1 are discarded.
       - Gradients between threshold1 and threshold2 are retained only if connected to strong edges.

    Args:
        image (Image.Image): Input PIL Image.
        threshold1 (int): Lower hysteresis threshold (0 to 500, default 100).
        threshold2 (int): Upper hysteresis threshold (0 to 500, default 200).
        aperture_size (int): Sobel aperture size (odd integer: 3, 5, or 7).

    Returns:
        Image.Image: Edge map as a clean RGB PIL Image.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    # Validate aperture size (must be 3, 5, or 7)
    aperture = 3
    if aperture_size in (3, 5, 7):
        aperture = aperture_size

    # Ensure threshold order
    t1 = min(int(threshold1), int(threshold2))
    t2 = max(int(threshold1), int(threshold2))

    edges = cv2.Canny(gray, t1, t2, apertureSize=aperture, L2gradient=True)

    # Convert 1-channel binary to 3-channel RGB for display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb, mode="RGB")


def detect_sobel_x(image: Image.Image, ksize: int = 3) -> Image.Image:
    """
    Compute horizontal gradient to detect vertical edges using Sobel X operator.

    How it works:
    Calculates the first derivative in the X direction:
    For ksize=3, the kernel is:
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
    Detects vertical intensity transitions (e.g. vertical columns, borders).

    Args:
        image (Image.Image): Input PIL Image.
        ksize (int): Kernel size (odd integer: 1, 3, 5, 7; default 3).

    Returns:
        Image.Image: Sobel X gradient map in RGB mode.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    k = min(7, k)

    # Compute 64-bit float gradient to prevent underflow from negative values
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    # Take absolute value and convert back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)

    abs_grad_rgb = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(abs_grad_rgb, mode="RGB")


def detect_sobel_y(image: Image.Image, ksize: int = 3) -> Image.Image:
    """
    Compute vertical gradient to detect horizontal edges using Sobel Y operator.

    How it works:
    Calculates the first derivative in the Y direction:
    For ksize=3, the kernel is:
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]]
    Detects horizontal intensity transitions (e.g. horizons, horizontal lines).

    Args:
        image (Image.Image): Input PIL Image.
        ksize (int): Kernel size (odd integer: 1, 3, 5, 7; default 3).

    Returns:
        Image.Image: Sobel Y gradient map in RGB mode.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    k = min(7, k)

    # Compute 64-bit float gradient
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    abs_grad_rgb = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(abs_grad_rgb, mode="RGB")


def detect_sobel_combined(image: Image.Image, ksize: int = 3) -> Image.Image:
    """
    Compute combined gradient magnitude (Sobel X + Sobel Y).

    How it works:
    Computes both horizontal (Gx) and vertical (Gy) gradients and calculates the
    true Euclidean gradient magnitude: Magnitude = sqrt(Gx^2 + Gy^2).
    This highlights edges in all orientations evenly.

    Args:
        image (Image.Image): Input PIL Image.
        ksize (int): Kernel size (odd integer: 1, 3, 5, 7; default 3).

    Returns:
        Image.Image: Combined gradient magnitude map in RGB mode.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    k = min(7, k)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)

    magnitude = cv2.magnitude(grad_x, grad_y)
    abs_magnitude = cv2.convertScaleAbs(magnitude)

    abs_rgb = cv2.cvtColor(abs_magnitude, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(abs_rgb, mode="RGB")


def detect_laplacian(image: Image.Image, ksize: int = 3) -> Image.Image:
    """
    Detect zero-crossings and sharp edges using the Laplacian second-derivative operator.

    How it works:
    The Laplacian operator computes the sum of second-order spatial derivatives:
    L(f) = (d^2 f / dx^2) + (d^2 f / dy^2).
    It detects rapid rate-of-change in gradient slopes, producing fine outlines around all edges.

    Args:
        image (Image.Image): Input PIL Image.
        ksize (int): Aperture size for computing second-derivative filters (odd integer: 1, 3, 5, 7; default 3).

    Returns:
        Image.Image: Laplacian edge map in RGB mode.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    k = min(7, k)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    laplacian_rgb = cv2.cvtColor(abs_laplacian, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(laplacian_rgb, mode="RGB")
