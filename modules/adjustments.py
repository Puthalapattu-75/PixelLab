"""
PixelLab - Image Adjustments Module
Provides functions for fine-grained image adjustments:
- Brightness
- Contrast
- Saturation
- Sharpness
- Gamma correction
"""

from PIL import Image, ImageEnhance
import numpy as np
import cv2
from modules.image_utils import ensure_rgb


def adjust_brightness(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """
    Adjust the brightness of an image.

    How it works:
    Multiplies every pixel value by a brightness factor.
    - factor = 0.0: produces a completely black image.
    - factor = 1.0: original image untouched.
    - factor > 1.0: increases image lightness.

    Args:
        image (Image.Image): Input PIL Image.
        factor (float): Brightness scaling factor (e.g. 0.0 to 2.5, default 1.0).

    Returns:
        Image.Image: Adjusted PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if factor == 1.0:
        return rgb_img

    enhancer = ImageEnhance.Brightness(rgb_img)
    return enhancer.enhance(float(factor))


def adjust_contrast(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """
    Adjust the contrast of an image.

    How it works:
    Expands or compresses the dynamic distance of pixel values from the mean gray level.
    - factor = 0.0: produces a flat, solid solid-gray image.
    - factor = 1.0: original image untouched.
    - factor > 1.0: highlights become brighter and shadows become deeper.

    Args:
        image (Image.Image): Input PIL Image.
        factor (float): Contrast factor (e.g. 0.0 to 2.5, default 1.0).

    Returns:
        Image.Image: Adjusted PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if factor == 1.0:
        return rgb_img

    enhancer = ImageEnhance.Contrast(rgb_img)
    return enhancer.enhance(float(factor))


def adjust_saturation(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """
    Adjust color saturation (vibrancy / color purity).

    How it works:
    Scales the distance of each color channel from the luminance (grayscale) value.
    - factor = 0.0: completely desaturated (pure black & white / grayscale).
    - factor = 1.0: original color saturation.
    - factor > 1.0: boosts color vibrancy.

    Args:
        image (Image.Image): Input PIL Image.
        factor (float): Saturation factor (e.g. 0.0 to 3.0, default 1.0).

    Returns:
        Image.Image: Adjusted PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if factor == 1.0:
        return rgb_img

    enhancer = ImageEnhance.Color(rgb_img)
    return enhancer.enhance(float(factor))


def adjust_sharpness(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """
    Adjust edge sharpness.

    How it works:
    Enhances or softens high-frequency spatial components.
    - factor = 0.0: softened / blurred image.
    - factor = 1.0: original image.
    - factor > 1.0: sharpens edges and micro-textures.

    Args:
        image (Image.Image): Input PIL Image.
        factor (float): Sharpness factor (e.g. 0.0 to 3.0, default 1.0).

    Returns:
        Image.Image: Adjusted PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if factor == 1.0:
        return rgb_img

    enhancer = ImageEnhance.Sharpness(rgb_img)
    return enhancer.enhance(float(factor))


def adjust_gamma(image: Image.Image, gamma: float = 1.0) -> Image.Image:
    """
    Apply nonlinear Gamma correction to adjust midtone luminance without clipping extremes.

    How it works:
    Human vision perceives light logarithmically, not linearly.
    Gamma correction applies a power-law transformation:
    Output = ((Input / 255) ^ (1 / gamma)) * 255.
    - gamma < 1.0: darkens midtones while keeping 0 and 255 fixed.
    - gamma = 1.0: linear / unchanged.
    - gamma > 1.0: brightens shadow details without blowing out whites.
    We precompute a 256-element Look-Up Table (LUT) for maximum speed.

    Args:
        image (Image.Image): Input PIL Image.
        gamma (float): Gamma value (e.g. 0.2 to 3.0, default 1.0).

    Returns:
        Image.Image: Gamma-corrected PIL Image.
    """
    rgb_img = ensure_rgb(image)
    if abs(gamma - 1.0) < 1e-4:
        return rgb_img

    gamma = max(0.05, float(gamma))
    inv_gamma = 1.0 / gamma

    # Pre-calculate 256-element lookup table (LUT)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)

    img_arr = np.array(rgb_img)
    # Apply LUT using OpenCV
    corrected_arr = cv2.LUT(img_arr, lut)
    return Image.fromarray(corrected_arr, mode="RGB")


def apply_adjustments_pipeline(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    gamma: float = 1.0,
) -> Image.Image:
    """
    Execute all tonal adjustments in a unified pipeline.

    Args:
        image (Image.Image): Input PIL Image.
        brightness (float): Brightness scaling factor (default 1.0).
        contrast (float): Contrast scaling factor (default 1.0).
        saturation (float): Saturation scaling factor (default 1.0).
        sharpness (float): Sharpness scaling factor (default 1.0).
        gamma (float): Gamma power value (default 1.0).

    Returns:
        Image.Image: Adjusted PIL Image.
    """
    result = image

    # 1. Brightness
    if brightness != 1.0:
        result = adjust_brightness(result, brightness)

    # 2. Contrast
    if contrast != 1.0:
        result = adjust_contrast(result, contrast)

    # 3. Saturation
    if saturation != 1.0:
        result = adjust_saturation(result, saturation)

    # 4. Sharpness
    if sharpness != 1.0:
        result = adjust_sharpness(result, sharpness)

    # 5. Gamma
    if abs(gamma - 1.0) >= 1e-4:
        result = adjust_gamma(result, gamma)

    return result
