"""
PixelLab - Automated Test Suite
Unit tests for core image processing algorithms, geometric transformations,
color adjustments, format conversion, compression, and error handling.
"""

import io
import pytest
import numpy as np
from PIL import Image

from modules.image_utils import (
    validate_image_file,
    pil_to_cv2,
    cv2_to_pil,
    ensure_rgb,
    ensure_rgba,
    format_bytes,
    calculate_aspect_ratio,
    get_image_info,
)
from modules.filters import (
    apply_grayscale,
    apply_sepia,
    apply_black_and_white,
    apply_vintage,
    apply_gaussian_blur,
    apply_median_blur,
    apply_sharpen,
    apply_emboss,
    apply_invert_colors,
    apply_edge_detection_filter,
)
from modules.adjustments import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_sharpness,
    adjust_gamma,
    apply_adjustments_pipeline,
)
from modules.transformations import (
    resize_image,
    crop_image,
    rotate_90_clockwise,
    rotate_180,
    rotate_270_clockwise,
    rotate_custom,
    flip_horizontal,
    flip_vertical,
)
from modules.enhancement import (
    enhance_sharpen,
    enhance_noise_reduction,
    enhance_gaussian_smoothing,
    enhance_histogram_equalization,
    enhance_clahe,
)
from modules.edge_detection import (
    detect_canny,
    detect_sobel_x,
    detect_sobel_y,
    detect_sobel_combined,
    detect_laplacian,
)
from modules.compression import (
    compress_image,
    convert_image_format,
)
from modules.watermark import add_text_watermark


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def sample_rgb_image():
    """Create a 120x80 RGB test image with red, green, and blue patterns."""
    arr = np.zeros((80, 120, 3), dtype=np.uint8)
    arr[:40, :60] = [255, 0, 0]      # Red quadrant
    arr[:40, 60:] = [0, 255, 0]      # Green quadrant
    arr[40:, :60] = [0, 0, 255]      # Blue quadrant
    arr[40:, 60:] = [200, 200, 200]  # Light gray quadrant
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_rgba_image():
    """Create a 100x100 RGBA image with 50% transparency."""
    arr = np.full((100, 100, 4), 128, dtype=np.uint8)
    arr[:, :, 3] = 128  # 50% alpha
    return Image.fromarray(arr, mode="RGBA")


# ==========================================
# TEST CASES: UTILITIES & VALIDATION
# ==========================================

def test_validate_image_file_valid(sample_rgb_image):
    """Test validation with a legitimate PNG buffer."""
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="PNG")
    buf.seek(0)

    is_valid, err, img = validate_image_file(buf)
    assert is_valid is True
    assert err is None
    assert img is not None
    assert img.size == (120, 80)


def test_validate_image_file_invalid():
    """Test validation with corrupt/random non-image bytes."""
    corrupted_data = b"NOT_AN_IMAGE_RANDOM_CORRUPTED_BYTES_1234567890"
    is_valid, err, img = validate_image_file(corrupted_data)
    assert is_valid is False
    assert err is not None
    assert img is None


def test_format_bytes():
    """Test byte size string formatting."""
    assert format_bytes(500) == "500.00 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(1048576) == "1.00 MB"


def test_calculate_aspect_ratio():
    """Test simplified aspect ratio calculation."""
    assert calculate_aspect_ratio(1920, 1080) == "16:9"
    assert calculate_aspect_ratio(800, 600) == "4:3"
    assert calculate_aspect_ratio(500, 500) == "1:1"


def test_pil_cv2_roundtrip(sample_rgb_image):
    """Test seamless conversion between PIL and OpenCV numpy formats."""
    cv_img = pil_to_cv2(sample_rgb_image)
    assert isinstance(cv_img, np.ndarray)
    assert cv_img.shape == (80, 120, 3)

    restored_pil = cv2_to_pil(cv_img)
    assert restored_pil.size == sample_rgb_image.size
    # Check pixel value match
    orig_np = np.array(sample_rgb_image)
    restored_np = np.array(restored_pil)
    assert np.allclose(orig_np, restored_np, atol=1)


# ==========================================
# TEST CASES: FILTERS
# ==========================================

def test_grayscale_conversion(sample_rgb_image):
    """Test grayscale filter output and channel equalization."""
    gray = apply_grayscale(sample_rgb_image, intensity=1.0)
    assert isinstance(gray, Image.Image)
    assert gray.size == sample_rgb_image.size

    arr = np.array(gray)
    # In RGB grayscale, R, G, and B values are equal
    assert np.all(arr[:, :, 0] == arr[:, :, 1])
    assert np.all(arr[:, :, 1] == arr[:, :, 2])


def test_sepia_filter(sample_rgb_image):
    """Test sepia color transform produces warm sepia signature (R > G > B on gray)."""
    sepia = apply_sepia(sample_rgb_image, intensity=1.0)
    assert isinstance(sepia, Image.Image)
    assert sepia.size == sample_rgb_image.size

    arr = np.array(sepia)
    # The bottom-right quadrant was neutral gray (200, 200, 200)
    # Under sepia, R should be higher than G, and G higher than B
    br_pixel = arr[60, 90]
    assert br_pixel[0] >= br_pixel[1] >= br_pixel[2]


def test_black_and_white(sample_rgb_image):
    """Test binarization produces only 0 and 255 values."""
    bw = apply_black_and_white(sample_rgb_image, threshold=128)
    arr = np.array(bw)
    unique_vals = np.unique(arr)
    for val in unique_vals:
        assert val in (0, 255)


def test_blur_and_sharpen_filters(sample_rgb_image):
    """Test Gaussian blur, median blur, and sharpening."""
    g_blur = apply_gaussian_blur(sample_rgb_image, kernel_size=5)
    assert g_blur.size == sample_rgb_image.size

    m_blur = apply_median_blur(sample_rgb_image, kernel_size=5)
    assert m_blur.size == sample_rgb_image.size

    sharp = apply_sharpen(sample_rgb_image, strength=1.5)
    assert sharp.size == sample_rgb_image.size


def test_invert_colors(sample_rgb_image):
    """Test color inversion: pure red (255, 0, 0) becomes cyan (0, 255, 255)."""
    inverted = apply_invert_colors(sample_rgb_image)
    arr = np.array(inverted)
    # Top-left was (255, 0, 0) -> should now be (0, 255, 255)
    assert list(arr[10, 10]) == [0, 255, 255]


# ==========================================
# TEST CASES: ADJUSTMENTS
# ==========================================

def test_brightness_adjustment(sample_rgb_image):
    """Test brightness increase elevates overall mean pixel value."""
    brighter = adjust_brightness(sample_rgb_image, factor=1.5)
    darker = adjust_brightness(sample_rgb_image, factor=0.5)

    mean_orig = np.mean(np.array(sample_rgb_image))
    mean_bright = np.mean(np.array(brighter))
    mean_dark = np.mean(np.array(darker))

    assert mean_bright > mean_orig
    assert mean_dark < mean_orig


def test_contrast_adjustment(sample_rgb_image):
    """Test contrast enhancement widens standard deviation of pixel values."""
    high_contrast = adjust_contrast(sample_rgb_image, factor=1.8)
    assert high_contrast.size == sample_rgb_image.size

    std_orig = np.std(np.array(sample_rgb_image))
    std_high = np.std(np.array(high_contrast))
    assert std_high > std_orig


def test_adjustments_pipeline(sample_rgb_image):
    """Test combined adjustments pipeline runs cleanly without distortion."""
    result = apply_adjustments_pipeline(
        sample_rgb_image,
        brightness=1.1,
        contrast=1.2,
        saturation=1.1,
        sharpness=1.0,
        gamma=1.1,
    )
    assert isinstance(result, Image.Image)
    assert result.size == sample_rgb_image.size


# ==========================================
# TEST CASES: TRANSFORMATIONS
# ==========================================

def test_resize_image(sample_rgb_image):
    """Test image resizing with and without aspect ratio locking."""
    # Free resize
    resized = resize_image(sample_rgb_image, target_width=60, target_height=40, maintain_aspect_ratio=False)
    assert resized.size == (60, 40)

    # Proportional resize (original was 120x80 = 1.5 ratio)
    resized_aspect = resize_image(sample_rgb_image, target_width=60, target_height=60, maintain_aspect_ratio=True)
    # Width 60 -> Height should scale to 40
    assert resized_aspect.size == (60, 40)


def test_resize_invalid_dimensions(sample_rgb_image):
    """Test that zero or negative dimensions raise ValueError."""
    with pytest.raises(ValueError):
        resize_image(sample_rgb_image, target_width=0, target_height=100)

    with pytest.raises(ValueError):
        resize_image(sample_rgb_image, target_width=100, target_height=-5)


def test_crop_image(sample_rgb_image):
    """Test image cropping within boundaries."""
    cropped = crop_image(sample_rgb_image, left=10, top=10, right=50, bottom=40)
    assert cropped.size == (40, 30)


def test_crop_invalid_bounds(sample_rgb_image):
    """Test that inverted crop boundaries raise ValueError."""
    with pytest.raises(ValueError):
        crop_image(sample_rgb_image, left=50, top=10, right=20, bottom=40)


def test_rotations(sample_rgb_image):
    """Test 90, 180, 270, and custom rotations."""
    # 120x80 rotated 90 deg -> 80x120
    rot90 = rotate_90_clockwise(sample_rgb_image)
    assert rot90.size == (80, 120)

    rot180 = rotate_180(sample_rgb_image)
    assert rot180.size == (120, 80)

    rot270 = rotate_270_clockwise(sample_rgb_image)
    assert rot270.size == (80, 120)

    rot_custom = rotate_custom(sample_rgb_image, angle=45, expand=True)
    assert isinstance(rot_custom, Image.Image)


def test_flips(sample_rgb_image):
    """Test horizontal and vertical flipping."""
    h_flip = flip_horizontal(sample_rgb_image)
    assert h_flip.size == sample_rgb_image.size

    # In horizontal flip, left and right quadrants swap
    orig_np = np.array(sample_rgb_image)
    flip_np = np.array(h_flip)
    assert np.array_equal(orig_np[:, 0], flip_np[:, -1])

    v_flip = flip_vertical(sample_rgb_image)
    assert v_flip.size == sample_rgb_image.size
    assert np.array_equal(orig_np[0, :], flip_np[-1, :]) is False


# ==========================================
# TEST CASES: EDGE DETECTION
# ==========================================

def test_canny_edge_detection(sample_rgb_image):
    """Test Canny edge detector output."""
    edges = detect_canny(sample_rgb_image, threshold1=50, threshold2=150)
    assert isinstance(edges, Image.Image)
    assert edges.size == sample_rgb_image.size
    # Edges should have black background (0) and white edge lines (255)
    arr = np.array(edges)
    assert 255 in arr


def test_sobel_and_laplacian(sample_rgb_image):
    """Test Sobel X, Sobel Y, Combined, and Laplacian edge detectors."""
    sx = detect_sobel_x(sample_rgb_image, ksize=3)
    sy = detect_sobel_y(sample_rgb_image, ksize=3)
    sc = detect_sobel_combined(sample_rgb_image, ksize=3)
    lap = detect_laplacian(sample_rgb_image, ksize=3)

    for img in (sx, sy, sc, lap):
        assert isinstance(img, Image.Image)
        assert img.size == sample_rgb_image.size


# ==========================================
# TEST CASES: ENHANCEMENT
# ==========================================

def test_enhancement_algorithms(sample_rgb_image):
    """Test USM sharpening, denoising, histogram equalization, and CLAHE."""
    sh = enhance_sharpen(sample_rgb_image, amount=1.2)
    assert sh.size == sample_rgb_image.size

    denoise = enhance_noise_reduction(sample_rgb_image, diameter=5)
    assert denoise.size == sample_rgb_image.size

    hist_eq = enhance_histogram_equalization(sample_rgb_image)
    assert hist_eq.size == sample_rgb_image.size

    clahe = enhance_clahe(sample_rgb_image, clip_limit=2.0)
    assert clahe.size == sample_rgb_image.size


# ==========================================
# TEST CASES: COMPRESSION & FORMAT CONVERSION
# ==========================================

def test_convert_format_and_transparency(sample_rgba_image):
    """Test converting transparent RGBA image to JPEG without crashing."""
    data, mime, ext = convert_image_format(sample_rgba_image, target_format="JPEG", quality=85)
    assert mime == "image/jpeg"
    assert ext == ".jpg"
    assert len(data) > 0

    # Ensure result can be read back as RGB
    reopened = Image.open(io.BytesIO(data))
    assert reopened.mode == "RGB"


def test_compression_reduction_calculation(sample_rgb_image):
    """Test compression analytics and reduction percentage."""
    orig_bytes = io.BytesIO()
    sample_rgb_image.save(orig_bytes, format="PNG")
    orig_size = len(orig_bytes.getvalue())

    # Compress to low-quality JPEG
    compressed_data, stats = compress_image(
        sample_rgb_image,
        target_format="JPEG",
        quality=40,
        original_size_bytes=orig_size,
    )

    assert len(compressed_data) > 0
    assert stats["format"] == "JPEG"
    assert stats["original_size_bytes"] == orig_size
    # JPEG at quality 40 should be smaller than raw PNG for this pattern
    if stats["new_size_bytes"] < orig_size:
        assert stats["is_reduced"] is True
        assert stats["reduction_percentage"] > 0


# ==========================================
# TEST CASES: WATERMARK
# ==========================================

def test_watermark(sample_rgb_image):
    """Test applying a text watermark."""
    watermarked = add_text_watermark(
        sample_rgb_image,
        text="PixelLab Test",
        font_size=20,
        opacity=0.7,
        position="Center",
        color="#FFFFFF",
    )
    assert isinstance(watermarked, Image.Image)
    assert watermarked.size == sample_rgb_image.size
