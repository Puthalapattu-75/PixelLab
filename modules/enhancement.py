"""
PixelLab - Image Enhancement Module
Contains computer vision algorithms for quality enhancement:
- Image Sharpening (Unsharp Masking)
- Noise Reduction (Bilateral Edge-Preserving Filter & Fast NLM)
- Gaussian Smoothing
- Global Histogram Equalization (Luminance channel)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
"""

from PIL import Image
import numpy as np
import cv2
from modules.image_utils import pil_to_cv2, cv2_to_pil, ensure_rgb


def enhance_sharpen(image: Image.Image, amount: float = 1.0, radius: int = 3) -> Image.Image:
    """
    Sharpen image using the industry-standard Unsharp Masking (USM) technique.

    How it works:
    1. A slightly blurred copy of the image is created using Gaussian blur.
    2. The blurred copy is subtracted from the original, isolating high-frequency edge details.
    3. The high-frequency details are multiplied by the 'amount' factor and added back.
    Formula: Sharpened = Original * (1 + amount) - Blurred * amount.

    Args:
        image (Image.Image): Input PIL Image.
        amount (float): Sharpening intensity (0.1 to 3.0, default 1.0).
        radius (int): Blur radius / kernel size (odd integer >= 1, default 3).

    Returns:
        Image.Image: Sharpened PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if amount <= 0:
        return rgb_img

    cv_bgr = pil_to_cv2(rgb_img)

    k = max(1, int(radius))
    if k % 2 == 0:
        k += 1

    # Create smooth Gaussian copy
    blurred = cv2.GaussianBlur(cv_bgr, (k, k), sigmaX=0)

    # Weighted addition to create unsharp mask
    # weight1 = 1.0 + amount, weight2 = -amount
    sharpened = cv2.addWeighted(cv_bgr, 1.0 + float(amount), blurred, -float(amount), 0)

    return cv2_to_pil(sharpened)


def enhance_noise_reduction(
    image: Image.Image,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> Image.Image:
    """
    Remove digital sensor noise while preserving sharp geometric edges using Bilateral Filtering.

    How it works:
    Standard smoothing blurs both noise and edges. Bilateral filtering considers
    both spatial distance AND radiometric color differences. Pixels with very different
    colors (across sharp edges) are not averaged together, keeping edges crisp while
    flattening noise in smooth regions.

    Args:
        image (Image.Image): Input PIL Image.
        diameter (int): Pixel neighborhood diameter (e.g. 5 to 15, default 9).
        sigma_color (float): Filter sigma in the color space (default 75.0).
        sigma_space (float): Filter sigma in the coordinate space (default 75.0).

    Returns:
        Image.Image: Denoised PIL Image.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)

    d = max(1, int(diameter))
    denoised_bgr = cv2.bilateralFilter(
        cv_bgr,
        d=d,
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )

    return cv2_to_pil(denoised_bgr)


def enhance_gaussian_smoothing(image: Image.Image, kernel_size: int = 5, sigma: float = 1.5) -> Image.Image:
    """
    Smooth image and attenuate high-frequency noise using Gaussian convolution.

    Args:
        image (Image.Image): Input PIL Image.
        kernel_size (int): Size of Gaussian kernel (odd integer >= 1).
        sigma (float): Gaussian kernel standard deviation.

    Returns:
        Image.Image: Smoothed PIL Image.
    """
    rgb_img = ensure_rgb(image)

    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1

    cv_bgr = pil_to_cv2(rgb_img)
    smoothed = cv2.GaussianBlur(cv_bgr, (k, k), sigmaX=float(sigma))
    return cv2_to_pil(smoothed)


def enhance_histogram_equalization(image: Image.Image) -> Image.Image:
    """
    Apply Global Histogram Equalization to expand dynamic contrast.

    How it works:
    For color images, equalizing R, G, and B independently corrupts color balance and hues.
    PixelLab converts the image to the YCrCb color space:
    - Y channel: Luminance (lightness intensity)
    - Cr, Cb channels: Chrominance (color information)
    We equalize only the Y (luminance) channel to expand contrast across dark and bright areas,
    then merge back and convert to RGB, preserving natural colors.

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Image.Image: Equalized PIL Image.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)

    # Convert BGR to YCrCb
    ycrcb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2YCrCb)

    # Split channels
    y, cr, cb = cv2.split(ycrcb)

    # Equalize histogram on Luminance channel only
    y_eq = cv2.equalizeHist(y)

    # Merge channels back
    ycrcb_eq = cv2.merge([y_eq, cr, cb])

    # Convert back to BGR
    bgr_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    return cv2_to_pil(bgr_eq)


def enhance_clahe(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    How it works:
    Standard histogram equalization operates globally, which often over-amplifies noise
    or washes out regions. CLAHE operates on localized rectangular tiles (e.g. 8x8).
    It clips histogram peaks above 'clip_limit' and redistributes them uniformly,
    preventing noise over-amplification, before using bilinear interpolation to remove boundaries.
    Applied exclusively on the L (Lightness) channel in the LAB color space.

    Args:
        image (Image.Image): Input PIL Image.
        clip_limit (float): Threshold for contrast limiting (1.0 to 10.0, default 2.0).
        tile_grid_size (int): Grid size for localized tiles (e.g. 4 to 16, default 8).

    Returns:
        Image.Image: CLAHE enhanced PIL Image.
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)

    # Convert BGR to LAB color space
    lab = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2LAB)

    # Split channels: L (Lightness), A (green-red), B (blue-yellow)
    l, a, b = cv2.split(lab)

    # Create CLAHE object
    grid = max(2, int(tile_grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(grid, grid),
    )

    # Apply CLAHE to L channel
    l_clahe = clahe.apply(l)

    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])

    # Convert back to BGR
    bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return cv2_to_pil(bgr_clahe)
