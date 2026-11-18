"""
PixelLab - Image Filters Module
Contains clean, beginner-friendly implementations of popular image filters
using OpenCV, Pillow, and NumPy.
"""

from typing import Tuple
from PIL import Image, ImageOps
import numpy as np
import cv2
from modules.image_utils import pil_to_cv2, cv2_to_pil, ensure_rgb


def apply_grayscale(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """
    Convert an image to grayscale with optional intensity blending.

    How it works:
    Each RGB pixel has Red, Green, and Blue values. In grayscale, the luminosity
    is computed as: Y = 0.299*R + 0.587*G + 0.114*B (reflecting human eye sensitivity).
    We can blend between the original image and the full grayscale image.

    Args:
        image (Image.Image): Input PIL Image.
        intensity (float): Blend intensity from 0.0 (original) to 1.0 (full grayscale).

    Returns:
        Image.Image: Grayscale or partially desaturated PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if intensity <= 0.0:
        return rgb_img

    # Convert to grayscale mode 'L', then back to 'RGB' for consistent display and blending
    gray_img = rgb_img.convert("L").convert("RGB")

    if intensity >= 1.0:
        return gray_img

    # Blend between original and grayscale based on intensity
    return Image.blend(rgb_img, gray_img, float(intensity))


def apply_sepia(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """
    Apply a warm, nostalgic sepia tone to the image.

    How it works:
    Sepia simulates the warm brownish tone of antique silver-gelatin photographs.
    We apply a linear color transformation matrix to the RGB channels:
    New R = 0.393*R + 0.769*G + 0.189*B
    New G = 0.349*R + 0.686*G + 0.168*B
    New B = 0.272*R + 0.534*G + 0.131*B

    Args:
        image (Image.Image): Input PIL Image.
        intensity (float): Blend intensity from 0.0 (original) to 1.0 (full sepia).

    Returns:
        Image.Image: Sepia-toned PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if intensity <= 0.0:
        return rgb_img

    img_array = np.array(rgb_img, dtype=np.float32)

    # 3x3 Sepia transformation matrix
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],  # Output Red
        [0.349, 0.686, 0.168],  # Output Green
        [0.272, 0.534, 0.131],  # Output Blue
    ])

    # Matrix multiplication: img_array shape is (H, W, 3)
    sepia_array = cv2.transform(img_array, sepia_matrix)
    # Clip pixel values to valid [0, 255] range
    sepia_array = np.clip(sepia_array, 0, 255).astype(np.uint8)
    sepia_pil = Image.fromarray(sepia_array, mode="RGB")

    if intensity >= 1.0:
        return sepia_pil

    return Image.blend(rgb_img, sepia_pil, float(intensity))


def apply_black_and_white(image: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Convert an image to high-contrast binary Black and White using a threshold.

    How it works:
    The image is first converted to grayscale. Then every pixel value is compared
    to the threshold: if pixel >= threshold, it becomes pure white (255);
    otherwise it becomes pure black (0).

    Args:
        image (Image.Image): Input PIL Image.
        threshold (int): Binarization cutoff threshold (0 to 255, default 128).

    Returns:
        Image.Image: Black and white PIL Image (in RGB mode for consistency).
    """
    rgb_img = ensure_rgb(image)
    cv_bgr = pil_to_cv2(rgb_img)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, bw = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    # Convert 1-channel back to RGB for uniform display
    bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(bw_rgb, mode="RGB")


def apply_vintage(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """
    Apply a vintage aesthetic filter with warm highlights, faded shadows,
    and a subtle vignette.

    How it works:
    1. Adjust color curves (boost red, lower blue for warm golden cast).
    2. Elevate black levels slightly for the classic film "faded" matte look.
    3. Add a soft vignette darkening around the edges.

    Args:
        image (Image.Image): Input PIL Image.
        intensity (float): Effect intensity from 0.0 to 1.0.

    Returns:
        Image.Image: Vintage film style PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if intensity <= 0.0:
        return rgb_img

    img_np = np.array(rgb_img, dtype=np.float32)
    h, w = img_np.shape[:2]

    # Warm color curve transformation
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]

    # Boost red tones slightly, boost green midtones, reduce blue (yellowish warmth)
    r_new = np.clip(r * 1.15 + 10, 0, 255)
    g_new = np.clip(g * 1.05 + 5, 0, 255)
    b_new = np.clip(b * 0.85 + 5, 0, 255)

    vintage_arr = np.stack([r_new, g_new, b_new], axis=2)

    # Generate soft radial vignette mask
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    # Normalized distance from center
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    norm_dist = dist_from_center / (max_dist + 1e-5)

    # Vignette strength curve
    vignette = 1.0 - (norm_dist ** 1.8) * 0.4
    vignette = np.clip(vignette, 0.4, 1.0)[:, :, np.newaxis]

    vintage_arr = np.clip(vintage_arr * vignette, 0, 255).astype(np.uint8)
    vintage_pil = Image.fromarray(vintage_arr, mode="RGB")

    if intensity >= 1.0:
        return vintage_pil

    return Image.blend(rgb_img, vintage_pil, float(intensity))


def apply_gaussian_blur(image: Image.Image, kernel_size: int = 5) -> Image.Image:
    """
    Smooth an image using Gaussian Blur.

    How it works:
    Convolves the image with a bell-shaped Gaussian kernel. Nearby pixels have
    more weight than distant pixels, effectively removing high-frequency noise
    while producing a naturally soft, cinematic blur.

    Args:
        image (Image.Image): Input PIL Image.
        kernel_size (int): Size of the Gaussian kernel (must be an odd integer >= 1).

    Returns:
        Image.Image: Gaussian blurred PIL Image.
    """
    rgb_img = ensure_rgb(image)

    # Ensure kernel size is an odd integer >= 1
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1

    if k == 1:
        return rgb_img

    cv_bgr = pil_to_cv2(rgb_img)
    blurred_bgr = cv2.GaussianBlur(cv_bgr, (k, k), sigmaX=0)
    return cv2_to_pil(blurred_bgr)


def apply_median_blur(image: Image.Image, kernel_size: int = 5) -> Image.Image:
    """
    Smooth an image using a Median Filter.

    How it works:
    Replaces each pixel with the median value of all pixels in its neighborhood.
    Unlike linear filters (like Gaussian blur), median blur is highly effective
    at removing "salt-and-pepper" impulse noise while preserving sharp boundaries.

    Args:
        image (Image.Image): Input PIL Image.
        kernel_size (int): Aperture linear size (must be odd integer >= 1).

    Returns:
        Image.Image: Median blurred PIL Image.
    """
    rgb_img = ensure_rgb(image)

    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1

    if k == 1:
        return rgb_img

    cv_bgr = pil_to_cv2(rgb_img)
    blurred_bgr = cv2.medianBlur(cv_bgr, k)
    return cv2_to_pil(blurred_bgr)


def apply_sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Enhance details and edge contrast using an unsharp convolution kernel.

    How it works:
    We convolve the image with a 3x3 sharpening kernel:
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]]
    The strength parameter allows smooth blending between the original image
    and the sharpened result.

    Args:
        image (Image.Image): Input PIL Image.
        strength (float): Sharpening factor (0.0 = original, 1.0 = standard, 2.0+ = strong).

    Returns:
        Image.Image: Sharpened PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if strength <= 0.0:
        return rgb_img

    cv_bgr = pil_to_cv2(rgb_img)

    # Standard 3x3 sharpening kernel
    center_val = 1.0 + 4.0 * strength
    side_val = -1.0 * strength
    kernel = np.array([
        [0, side_val, 0],
        [side_val, center_val, side_val],
        [0, side_val, 0]
    ], dtype=np.float32)

    sharpened_bgr = cv2.filter2D(cv_bgr, -1, kernel)
    return cv2_to_pil(sharpened_bgr)


def apply_emboss(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """
    Create a 3D embossed relief effect.

    How it works:
    Applies a directional differential kernel across diagonal neighbors:
    [[-2, -1,  0],
     [-1,  1,  1],
     [ 0,  1,  2]]
    and adds an offset of 128 (neutral gray). Pixels with a positive gradient
    become lighter, while negative gradients become darker, creating the illusion of depth.

    Args:
        image (Image.Image): Input PIL Image.
        intensity (float): Blend intensity from 0.0 to 1.0.

    Returns:
        Image.Image: Embossed PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if intensity <= 0.0:
        return rgb_img

    cv_bgr = pil_to_cv2(rgb_img)
    # Convert to grayscale first for classic clean emboss
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    kernel = np.array([
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2]
    ], dtype=np.float32)

    # Filter with 128 delta offset
    embossed = cv2.filter2D(gray, -1, kernel, delta=128)
    embossed_rgb = cv2.cvtColor(embossed, cv2.COLOR_GRAY2RGB)
    embossed_pil = Image.fromarray(embossed_rgb, mode="RGB")

    if intensity >= 1.0:
        return embossed_pil

    return Image.blend(rgb_img, embossed_pil, float(intensity))


def apply_invert_colors(image: Image.Image) -> Image.Image:
    """
    Invert image colors to create a photo negative effect.

    How it works:
    Each RGB pixel value x is subtracted from 255: x_new = 255 - x.
    Pure black (0) becomes pure white (255), and red (255, 0, 0) becomes cyan (0, 255, 255).

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Image.Image: Color-inverted PIL Image.
    """
    rgb_img = ensure_rgb(image)
    return ImageOps.invert(rgb_img)


def apply_edge_detection_filter(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """
    Edge filter for the filters collection that highlights structural outlines.

    How it works:
    Applies a Laplacian high-pass convolution kernel that computes the second spatial
    derivative of pixel intensities:
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]]

    Args:
        image (Image.Image): Input PIL Image.
        intensity (float): Effect intensity from 0.0 to 1.0.

    Returns:
        Image.Image: Edge-filtered PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if intensity <= 0.0:
        return rgb_img

    cv_bgr = pil_to_cv2(rgb_img)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    # 3x3 Edge highlight kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    edges = cv2.filter2D(gray, -1, kernel)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_pil = Image.fromarray(edges_rgb, mode="RGB")

    if intensity >= 1.0:
        return edges_pil

    return Image.blend(rgb_img, edges_pil, float(intensity))
