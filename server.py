'''
@author Kyle R. Covington
@author Yellow Bird Creek Intelligent Systems
'''

import numpy as np
import io
import cv2
import re
import xml.etree.ElementTree as ET
import cairosvg
import potrace  # pip install potracer cairosvg opencv-python pillow numpy

from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from PIL import UnidentifiedImageError, Image

# #####
# Image processing functions
# #####

def inkscape_style_preprocess(
        image: Image.Image,
        blur_ksize: int = 1,
        otsu: bool = False,
        sat_thresh: int = 40,   # Saturation cutoff (0-255). Values above this are COLOR fills.
        val_thresh: int = 20,   # Brightness cutoff (0-255). Values below this are DARK.
    ) -> np.ndarray:
    """
    Replicates Inkscape's C++ bitmap pre-processing:
    Grayscale -> Gaussian Anti-Aliasing Blur -> Otsu Auto-Thresholding -> Inversion
    """
    rgb = np.array(image.convert('RGB'))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(gray.shape)

    if otsu:
        # 1. Mild Gaussian blur eliminates pixelation before Potrace sees it
        if blur_ksize > 1:
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # 2. Otsu's binarization automatically finds the ideal threshold cutoff
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary > 0  # Returns boolean bitmap for Potrace

    else:
        if blur_ksize > 1:
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            rgb = cv2.GaussianBlur(rgb, (blur_ksize, blur_ksize), 0.8)
        # 2. Convert to HSV space
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        _, S, V = cv2.split(hsv)

        # 3. Outline definition: Low Value (Dark) AND Low Saturation (Achromatic/Gray/Black)
        # Dark color fills (Navy, Dark Red) have low Value BUT high Saturation (S > sat_thresh)
        outline_bitmap = (V <= val_thresh) & (S <= sat_thresh)

        return outline_bitmap  # Returns boolean mask for Potrace

def trace_unified_silhouette(boolean_bitmap: np.ndarray, turdsize: int = 5, opttolerance: float = 0.4) -> str:
    """
    Runs Potrace over the entire image at once (Inkscape Single-Scan style)
    to produce continuous, unified Bezier paths.
    """
    bmp = potrace.Bitmap(boolean_bitmap)
    path = bmp.trace(
        turdsize=turdsize,                  # Filters noise speckles
        turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY,
        alphamax=1.0,                       # Maximum corner smoothing
        opticurve=1,                        # Enables Bezier curve optimizer
        opttolerance=opttolerance          # Smooths jittery nodes into long curves
    )

    d_parts = []
    for curve in path:
        start = curve.start_point
        d_parts.append(f"M {start.x:.2f},{start.y:.2f}")
        for segment in curve:
            end = segment.end_point
            if segment.is_corner:
                c = segment.c
                d_parts.append(f"L {c.x:.2f},{c.y:.2f} L {end.x:.2f},{end.y:.2f}")
            else:
                c1, c2 = segment.c1, segment.c2
                d_parts.append(f"C {c1.x:.2f},{c1.y:.2f} {c2.x:.2f},{c2.y:.2f} {end.x:.2f},{end.y:.2f}")
        d_parts.append("Z")

    return " ".join(d_parts)

def sample_border_ring_color(
    subpath_d: str,
    orig_rgb: np.ndarray,
    width: int,
    height: int,
    stroke_px: int = 4,
    blur_ksize: int = 7
) -> tuple[int, str]:
    """
    Samples color along a ring immediately inside the path boundary
    to handle layered/nested objects properly.
    This tries to guess the correct color of the object.
    """
    # Render interior mask
    fill_svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="black"/>
        <path d="{subpath_d}" fill="white" stroke="none" />
    </svg>'''
    fill_bytes = cairosvg.svg2png(bytestring=fill_svg.encode('utf-8'))
    fill_img = cv2.imdecode(np.frombuffer(fill_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    _, interior_mask = cv2.threshold(fill_img, 127, 255, cv2.THRESH_BINARY)

    area = np.count_nonzero(interior_mask)
    if area == 0:
        return 0, "#000000"

    # Render border stroke mask
    stroke_svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="black"/>
        <path d="{subpath_d}" fill="none" stroke="white" stroke-width="{stroke_px}" stroke-linejoin="round" />
    </svg>'''
    stroke_bytes = cairosvg.svg2png(bytestring=stroke_svg.encode('utf-8'))
    stroke_img = cv2.imdecode(np.frombuffer(stroke_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    _, stroke_mask = cv2.threshold(stroke_img, 127, 255, cv2.THRESH_BINARY)

    # Blur border stroke and intersect with interior
    blurred_stroke = cv2.GaussianBlur(stroke_mask, (blur_ksize, blur_ksize), 0)
    _, expanded_border = cv2.threshold(blurred_stroke, 20, 255, cv2.THRESH_BINARY)

    # Interior border ring
    ring_mask = cv2.bitwise_and(expanded_border, interior_mask)
    sample_mask = ring_mask if np.any(ring_mask) else interior_mask

    # Sample mean color
    ring_pixels = orig_rgb[sample_mask == 255]
    mean_rgb = np.mean(ring_pixels[:, :3], axis=0).astype(int)
    return area, f"#{mean_rgb[0]:02x}{mean_rgb[1]:02x}{mean_rgb[2]:02x}"


def convert_png_to_colored_svg(
    img_pil: Image.Image,
    mask_pil: Image.Image
) -> str:
    """
    Pure Python execution matching Inkscape's exact Potrace pipeline + post-processing.
    """
    width, height = img_pil.size
    orig_rgb = np.array(img_pil)

    # Inkscape-style pre-processing; this section generates a bitmap of the
    # dark/black outlines that objects should be colored in. Remember that this
    # tool does not separate by colors, it only works on cartoon-like drawings
    # at this time. Shades or gradients are not kept, the color is averaged.
    boolean_bitmap = np.array(mask_pil) > 1

    # Trace unified silhouette; this does create a single stroke object that
    # needs to be split
    unified_d = trace_unified_silhouette(boolean_bitmap, turdsize=5, opttolerance=0.4)

    # Split compound path into standalone subpaths
    subpaths = [s.strip() for s in re.findall(r'[Mm][^Mm]+', unified_d) if s.strip()]

    # Sample ring colors and measure areas; take each sub-path (which are
    # real paths at this point) in turn and tries to guess the correct color by
    # doing a blur of the bounding path and intersecting that with the eroded
    # internal area of the path. This effectively selects just the pixels that
    # are internal to the stroke, avoiding internal objects (hopefully)
    path_data = []
    for subpath in subpaths:
        area, color = sample_border_ring_color(subpath, orig_rgb, width, height)
        if area > 10:
            path_data.append({'d': subpath, 'color': color, 'area': area})

    # Sort by surface area (Backgrounds rendered first, inner shapes on top)
    # This (hopefully) sorts the objects so that the larger objects are on the
    # bottom. Since this comes from a bmp, there should be no 'hidden' objects
    # like we might have in a real svg.
    path_data.sort(key=lambda x: x['area'], reverse=True)

    # Export clean SVG
    svg_root = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': '100%',
        'height': '100%',
        'viewBox': f'0 0 {width} {height}'
    })

    for i, item in enumerate(path_data):
        ET.SubElement(svg_root, 'path', {
            'id': f'path_{i}',
            'd': item['d'],
            'fill': item['color'],
            'stroke': '#000000',
            'stroke-width': '1.0',
            'stroke-linejoin': 'round',
            'stroke-linecap': 'round'
        })

    # Return ET as a string
    return ET.tostring(svg_root, encoding='unicode')


app = FastAPI()

@app.get("/")
async def read_root():
    with open("index.html", "r") as f:
        return Response(content=f.read(), media_type="text/html")

@app.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...),
        blur_ksize: int = 1,
        otsu: bool = False,
        sat_thresh: int = 40,   # Saturation cutoff (0-255). Values above this are COLOR fills.
        val_thresh: int = 20,   # Brightness cutoff (0-255). Values below this are DARK.
        ):
    try:
        print(file)
        image_data = await file.read()

        image: Image.Image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image.convert('RGB')

        binary_mask =  inkscape_style_preprocess(
                image,
                blur_ksize = blur_ksize,
                otsu = otsu,
                sat_thresh = sat_thresh,
                val_thresh = val_thresh)

        mask_u8 = binary_mask.astype(np.uint8) * 255
        grayscale_mask = Image.fromarray(mask_u8, mode='L')

        png_buffer = io.BytesIO()
        grayscale_mask.save(png_buffer, format='PNG')

        return Response(content=png_buffer.getvalue(),
                        media_type='image/png')
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e)) from e




@app.post("/process")
async def process_image(file: UploadFile = File(...), mask: UploadFile =
                        File(...)):
    try:
        image_data = await file.read()
        mask_data = await mask.read()

        image: Image.Image = Image.open(io.BytesIO(image_data))
        maskI: Image.Image = Image.open(io.BytesIO(mask_data))
        if image.mode != 'RGB':
            image.convert('RGB')

        svg_text: str = convert_png_to_colored_svg(image, maskI)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file format. Please upload a PNG or JPEG."
        )
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )

    return {"svg": svg_text}

def preprocess_for_vectorization(
    pil_image: Image.Image,
    k_colors: int = 8,
    bilateral_d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    force_black_white: bool = True,
    black_thresh: int = 30,  # L-channel threshold for dark pixels (0-255)
    white_thresh: int = 225,  # L-channel threshold for bright pixels (0-255)
) -> Image.Image:
    """Pre-processes a PIL Image for StarVector by smoothing textures, flattening

    colors with CIELAB K-Means, and explicitly locking Pure Black and Pure White
    into the color palette.
    """
    # 1. Handle transparency and convert PIL Image to RGB NumPy array
    has_alpha = pil_image.mode == "RGBA"
    if has_alpha:
        alpha_channel = pil_image.split()[3]
        img_rgb = pil_image.convert("RGB")
    else:
        img_rgb = pil_image.convert("RGB")

    img_np = np.array(img_rgb)

    # 2. Bilateral Filter for edge-preserving spatial smoothing
    smoothed = cv2.bilateralFilter(
        img_np, d=bilateral_d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )

    # 3. Convert RGB to CIELAB space
    lab_img = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
    h, w, c = lab_img.shape
    pixels = lab_img.reshape((-1, 3)).astype(np.float32)

    if force_black_white and k_colors > 2:
        # Define exact Pure Black and Pure White in OpenCV CIELAB space
        # (L: 0..255, a: 128=neutral, b: 128=neutral)
        black_center = np.array([0.0, 128.0, 128.0], dtype=np.float32)
        white_center = np.array([255.0, 128.0, 128.0], dtype=np.float32)

        # Filter out extreme light and dark pixels from the K-Means training set
        l_channel = pixels[:, 0]
        intermediate_mask = (l_channel >= black_thresh) & (
            l_channel <= white_thresh
        )
        intermediate_pixels = pixels[intermediate_mask]

        num_mid_colors = k_colors - 2

        if len(intermediate_pixels) >= num_mid_colors and num_mid_colors > 0:
            # Cluster remaining intermediate colors
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                0.2,
            )
            _, _, mid_centers = cv2.kmeans(
                intermediate_pixels,
                num_mid_colors,
                None,
                criteria,
                10,
                cv2.KMEANS_PP_CENTERS,
            )
            # Combine forced Black & White with learned intermediate colors
            palette = np.vstack(
                [black_center, white_center, mid_centers]
            ).astype(np.float32)
        else:
            palette = np.vstack([black_center, white_center]).astype(
                np.float32
            )
    else:
        # Fallback to standard K-Means
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.2,
        )
        _, _, palette = cv2.kmeans(
            pixels, k_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

    # 4. Snap all image pixels to their nearest centroid in the locked palette
    dists = np.sum((pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)

    # 5. Reshape and convert quantized image back to RGB space
    quantized_lab = palette[labels].reshape((h, w, c)).astype(np.uint8)
    quantized_rgb = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2RGB)

    # 6. Reconstruct PIL Image (restoring alpha channel if present)
    output_image = Image.fromarray(quantized_rgb)
    if has_alpha:
        output_image.putalpha(alpha_channel)

    output_image.save('/tmp/snapimage.png')

    return output_image



