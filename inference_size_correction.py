"""
InsertAnything-SizeCorrection Full Inference Pipeline (Standalone) — FIXED (Scheme B)

Usage:
    python run_size_correction_fixed.py --image_path examples/soccer_and_tennis.png

[NEW]
- --crop_ratio: controls zoom-in crop size (default 3.0)
- --mask_dilate: controls simple mask dilation strength (default matches old behavior)

[IMPORTANT CHANGE]
- Align inference IO format to training / original InsertAnything:
  image     = [ref | target]
  mask      = [0   | target_mask]
  depthcond = [0   | depth]
  output    = crop right half
"""

import gc
import io
import os
import re
import sys
import shutil
import copy
import math
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw
import json
import base64
import requests
import argparse
from omegaconf import OmegaConf

# Diffusers & Transformers
from diffusers import (
    FluxFillPipeline,
    FluxPriorReduxPipeline,
    FluxControlNetModel,
    FluxKontextPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

# Project Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add Depth Anything V2 path
depth_anything_v2_path = os.environ.get("DEPTH_ANYTHING_V2_PATH", "")
if depth_anything_v2_path and depth_anything_v2_path not in sys.path:
    sys.path.insert(0, depth_anything_v2_path)

try:
    from src.models.pipeline_tools import encode_images, prepare_text_input, Flux_fill_encode_masks_images
    from src.models.image_project import image_output
    from src.models.transformer import tranformer_forward
    from src.models.size_correction import (
        mask_controlnet_residuals_for_diptych,
        HF_LATENT_DIM,
        HF_INJECT_CKPT_NAME,
    )
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ Error: Could not import src.models.*: {e}")
    sys.exit(1)

_LOCAL_MODEL_CACHE = {}


# ==============================================================================
# HF map helpers (inline; avoid importing src.data.base which pulls extra deps like bezier)
# ==============================================================================
def high_frequency_map_rgb_numpy(
    image_uint8: np.ndarray,
    radius_frac: float = 0.15,
) -> np.ndarray:
    """
    HiFi-Inpaint-style high-frequency map (matches src/data/base.py):
    - DFT high-pass per channel
    - magnitude
    - per-image max normalization
    image_uint8: H,W,3 uint8
    Returns: H,W,3 float32 in [0, 1]
    """
    x = image_uint8.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    out = np.zeros_like(x, dtype=np.float32)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    r = max(1, int(radius_frac * min(h, w) / 2))
    hp_mask = (dist > float(r)).astype(np.float32)
    for c in range(3):
        F = np.fft.fft2(x[:, :, c])
        Fshift = np.fft.fftshift(F)
        Fh = Fshift * hp_mask
        F2 = np.fft.ifftshift(Fh)
        out[:, :, c] = np.abs(np.fft.ifft2(F2))
    mx = float(out.max()) + 1e-8
    out = out / mx
    return out


def ref_hf_interior_weight_rgb(masked_ref_uint8: np.ndarray, erode_iters=None) -> np.ndarray:
    """
    Weights in [0,1] (H,W) to keep ref HF mostly inside the visible object,
    downweighting boundaries and synthetic-occlusion rims (matches src/data/base.py).
    """
    h, w = masked_ref_uint8.shape[:2]
    gray = np.mean(masked_ref_uint8.astype(np.float32), axis=2)
    fg = (gray < 250.0).astype(np.uint8)
    n_fg = int(fg.sum())
    if n_fg < 16:
        return np.ones((h, w), dtype=np.float32)

    if erode_iters is None:
        erode_iters = max(2, min(6, min(h, w) // 128))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner = cv2.erode(fg, k, iterations=int(erode_iters))
    n_in = int(inner.sum())
    min_keep = max(16, int(n_fg * 0.03))
    if n_in < min_keep:
        inner = cv2.erode(fg, k, iterations=max(1, int(erode_iters) // 2))
        n_in = int(inner.sum())
    if n_in < 8:
        inner = fg
    return inner.astype(np.float32)


def _cache_get(key, create_fn):
    if key not in _LOCAL_MODEL_CACHE:
        _LOCAL_MODEL_CACHE[key] = create_fn()
    return _LOCAL_MODEL_CACHE[key]

# ==============================================================================
# 0. Debug helpers
# ==============================================================================
def draw_bbox(img: Image.Image, bbox_yxyx, out_path: str, color=(255, 0, 0), width=4):
    """Draw bbox [y1,x1,y2,x2] on image for debugging."""
    im = img.copy()
    d = ImageDraw.Draw(im)
    y1, x1, y2, x2 = bbox_yxyx
    d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    im.save(out_path)


def letterbox_pil_to_canvas(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale uniformly to fit inside target_w x target_h, pad with white (for ref image swap)."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (max(1, target_w), max(1, target_h)), (255, 255, 255))
    scale = min(target_w / float(w), target_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def scale_bbox_yxyx(b, sx, sy):
    """Scale bbox in pixel coords from (W1,H1) space to (W2,H2) space."""
    y1, x1, y2, x2 = b
    return [
        int(round(y1 * sy)),
        int(round(x1 * sx)),
        int(round(y2 * sy)),
        int(round(x2 * sx)),
    ]


def bbox_raw_to_pixels(raw_bbox, width, height):
    """Convert Gemini bbox to pixel yxyx; supports [0,1], [0,1000], or pixel-like values."""
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox[0], list):
        raw_bbox = raw_bbox[0]
    raw = [float(x) for x in raw_bbox]
    is_normalized = (max(raw) <= 1.0)
    scale_y = float(height) if is_normalized else float(height) / 1000.0
    scale_x = float(width) if is_normalized else float(width) / 1000.0
    y1 = int(raw[0] * scale_y)
    x1 = int(raw[1] * scale_x)
    y2 = int(raw[2] * scale_y)
    x2 = int(raw[3] * scale_x)
    return [y1, x1, y2, x2]


def bbox_iou_yxyx(a, b):
    """IoU for two boxes in [y1,x1,y2,x2]."""
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    ih, iw = max(0, iy2 - iy1), max(0, ix2 - ix1)
    inter = ih * iw
    area_a = max(1, (ay2 - ay1) * (ax2 - ax1))
    area_b = max(1, (by2 - by1) * (bx2 - bx1))
    union = area_a + area_b - inter
    return float(inter) / float(max(1, union))


def bbox_intersection_over_min_area(a, b):
    """Intersection divided by the smaller box area."""
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    ih, iw = max(0, iy2 - iy1), max(0, ix2 - ix1)
    inter = ih * iw
    area_a = max(1, (ay2 - ay1) * (ax2 - ax1))
    area_b = max(1, (by2 - by1) * (bx2 - bx1))
    return float(inter) / float(max(1, min(area_a, area_b)))


def _layout_name_key(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _layout_update_object_bbox(layout_doc: dict, object_name: str, bbox_yxyx) -> bool:
    """Patch one object's bbox_yxyx in a layout JSON dict (multi-round layout_final tracking)."""
    if not isinstance(layout_doc, dict) or bbox_yxyx is None or len(bbox_yxyx) != 4:
        return False
    bb_i = [int(x) for x in bbox_yxyx]
    key = _layout_name_key(object_name)
    for o in layout_doc.get("objects") or []:
        if _layout_name_key(str(o.get("name", ""))) == key:
            o["bbox_yxyx"] = bb_i
            return True
    return False


# ==============================================================================
# 1. Core Functions
# ==============================================================================
def encode_depth_for_controlnet(pipeline, depth_images):
    """Encode depth images into latent space for ControlNet conditioning.

    NOTE:
    - diffusers.image_processor.preprocess expects:
        * PIL / numpy: uint8 [0,255] ok
        * torch tensor: float [0,1] (NOT [-1,1])
    This wrapper makes it robust to accidental [-1,1] tensors.
    """
    # ✅ If someone upstream passes torch tensor in [-1,1], map it back to [0,1]
    if torch.is_tensor(depth_images):
        # depth_images: (B,C,H,W) or (C,H,W)
        if depth_images.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            depth_images = depth_images.float()
        # heuristic: if any negative => likely [-1,1]
        if depth_images.min().item() < 0:
            depth_images = (depth_images + 1.0) * 0.5
        depth_images = depth_images.clamp(0.0, 1.0)

    depth_images = pipeline.image_processor.preprocess(depth_images)
    depth_images = depth_images.to(pipeline.device).to(pipeline.dtype)

    depth_latents = pipeline.vae.encode(depth_images).latent_dist.sample()
    depth_latents = (depth_latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor

    depth_latents_packed = pipeline._pack_latents(depth_latents, *depth_latents.shape)
    depth_ids = pipeline._prepare_latent_image_ids(
        depth_latents.shape[0],
        depth_latents.shape[2],
        depth_latents.shape[3],
        pipeline.device,
        pipeline.dtype,
    )

    # Safety: some configs halve spatial
    if depth_latents_packed.shape[1] != depth_ids.shape[0]:
        depth_ids = pipeline._prepare_latent_image_ids(
            depth_latents.shape[0],
            depth_latents.shape[2] // 2,
            depth_latents.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return depth_latents_packed, depth_ids


def build_hf_diptych_pil_from_ref(
    ref_rgb_uint8: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    hf_hp_radius: float = 0.15,
):
    """
    Match ``src/data/base.py`` AnyInsertionDataset: ref HF | black right, for VAE+pack -> hf_latent_inject.
    ref_rgb_uint8: H,W,3 uint8 (same spatial size as left canvas half).
    """
    hf_ref = high_frequency_map_rgb_numpy(ref_rgb_uint8, radius_frac=float(hf_hp_radius))
    _hf_w = ref_hf_interior_weight_rgb(ref_rgb_uint8)[..., None]
    hf_ref = hf_ref * _hf_w
    hf_ref = hf_ref / (float(hf_ref.max()) + 1e-8)
    black_right = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    hf_left_rgb = (hf_ref * 255.0).clip(0, 255).astype(np.uint8)
    hf_diptych_u8 = np.concatenate([hf_left_rgb, black_right], axis=1)
    return Image.fromarray(hf_diptych_u8, mode="RGB")


# ==============================================================================
# 2. Utility Functions (Zoom & Geometry) — FIXED
# ==============================================================================
def get_new_bbox_from_anchor_unclamped(bbox, scale_factor, anchor_point):
    """
    Calculates new bbox coordinates based on an anchor point and scale factor.
    IMPORTANT: returns UNCLAMPED bbox, can go out of image bounds.
    bbox: [ymin, xmin, ymax, xmax]
    Pure edge-anchor: the contact edge stays fixed.
    """
    ymin, xmin, ymax, xmax = [float(v) for v in bbox]
    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)

    new_w = max(1, int(round(w * float(scale_factor))))
    new_h = max(1, int(round(h * float(scale_factor))))

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    if anchor_point == "BOTTOM_CENTER":
        new_ymax = int(round(ymax))
        new_ymin = new_ymax - new_h
        new_xmin = int(round(cx - new_w / 2.0))
        new_xmax = new_xmin + new_w
    elif anchor_point == "TOP_CENTER":
        new_ymin = int(round(ymin))
        new_ymax = new_ymin + new_h
        new_xmin = int(round(cx - new_w / 2.0))
        new_xmax = new_xmin + new_w
    elif anchor_point == "LEFT_CENTER":
        new_xmin = int(round(xmin))
        new_xmax = new_xmin + new_w
        new_ymin = int(round(cy - new_h / 2.0))
        new_ymax = new_ymin + new_h
    elif anchor_point == "RIGHT_CENTER":
        new_xmax = int(round(xmax))
        new_xmin = new_xmax - new_w
        new_ymin = int(round(cy - new_h / 2.0))
        new_ymax = new_ymin + new_h
    else:  # CENTER
        new_xmin = int(round(cx - new_w / 2.0))
        new_xmax = new_xmin + new_w
        new_ymin = int(round(cy - new_h / 2.0))
        new_ymax = new_ymin + new_h

    return [int(new_ymin), int(new_xmin), int(new_ymax), int(new_xmax)]

def get_bbox_from_mask(mask):
    """Get bounding box from binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, mask.shape[0], 0, mask.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def select_best_component_in_bbox(mask_uint8, bbox_yxyx):
    """
    Keep one SAM2 connected component that best matches target bbox.
    This avoids oversized/shifted masks that cause anchor-position drift.
    """
    y1, x1, y2, x2 = [int(v) for v in bbox_yxyx]
    H, W = mask_uint8.shape[:2]
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(H, y2), min(W, x2)
    if y2 <= y1 or x2 <= x1:
        return mask_uint8

    bin_mask = (mask_uint8 > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if n_labels <= 1:
        return bin_mask

    bbox_area = max(1, (y2 - y1) * (x2 - x1))
    cy = 0.5 * (y1 + y2)
    cx = 0.5 * (x1 + x2)
    diag = max(1.0, np.hypot((y2 - y1), (x2 - x1)))

    best_label = 0
    best_score = -1e9
    for lab in range(1, n_labels):
        comp = (labels == lab)
        area = int(comp.sum())
        if area < 16:
            continue
        inter = int(comp[y1:y2, x1:x2].sum())
        overlap = inter / float(bbox_area)
        comp_cx, comp_cy = centroids[lab][0], centroids[lab][1]
        center_dist = np.hypot(comp_cy - cy, comp_cx - cx) / diag
        # favor overlap and area, penalize far-away components
        score = 2.0 * overlap + 0.15 * np.log1p(area) - 1.2 * center_dist
        if score > best_score:
            best_score = score
            best_label = lab

    if best_label == 0:
        return bin_mask
    return (labels == best_label).astype(np.uint8)

def clip_scale_to_image_boundary(orig_box_px, scale_factor, anchor_point, img_w, img_h):
    """
    If the scaled bbox would overflow the image, reduce scale_factor so the
    scaled box just fits within [0, img_w) x [0, img_h).  Returns the
    (possibly reduced) scale_factor.  Never increases scale_factor.
    """
    if scale_factor <= 1.0:
        return scale_factor  # shrinking never overflows
    ymin, xmin, ymax, xmax = [float(v) for v in orig_box_px]
    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)

    if anchor_point == "BOTTOM_CENTER":
        fixed_cx = (xmin + xmax) / 2.0
        max_scale_y = ymax / h if h > 0 else scale_factor
        max_scale_x = 2.0 * min(fixed_cx, img_w - fixed_cx) / w if w > 0 else scale_factor
    elif anchor_point == "TOP_CENTER":
        fixed_cx = (xmin + xmax) / 2.0
        max_scale_y = (img_h - ymin) / h if h > 0 else scale_factor
        max_scale_x = 2.0 * min(fixed_cx, img_w - fixed_cx) / w if w > 0 else scale_factor
    elif anchor_point == "LEFT_CENTER":
        fixed_cy = (ymin + ymax) / 2.0
        max_scale_x = (img_w - xmin) / w if w > 0 else scale_factor
        max_scale_y = 2.0 * min(fixed_cy, img_h - fixed_cy) / h if h > 0 else scale_factor
    elif anchor_point == "RIGHT_CENTER":
        fixed_cy = (ymin + ymax) / 2.0
        max_scale_x = xmax / w if w > 0 else scale_factor
        max_scale_y = 2.0 * min(fixed_cy, img_h - fixed_cy) / h if h > 0 else scale_factor
    else:  # CENTER
        fixed_cx = (xmin + xmax) / 2.0
        fixed_cy = (ymin + ymax) / 2.0
        max_scale_x = 2.0 * min(fixed_cx, img_w - fixed_cx) / w if w > 0 else scale_factor
        max_scale_y = 2.0 * min(fixed_cy, img_h - fixed_cy) / h if h > 0 else scale_factor

    max_scale = min(max_scale_x, max_scale_y)
    max_scale = max(1.0, max_scale)
    clipped = min(scale_factor, max_scale)
    if clipped < scale_factor - 1e-6:
        print(
            f"   [overflow-clip] scale {scale_factor:.3f} → {clipped:.3f} "
            f"(bbox fits image {img_w}×{img_h})",
            flush=True,
        )
    return clipped

def get_padded_square_crop_coords(
    bbox,
    ratio=3.0,
    image_size=None,               # (W, H) or None
    bias_y=0.18,                   # 向下偏移比例（相对 crop_size）
    bias_x=0.0,                    # 向右偏移比例（相对 crop_size）
    prefer_down=True,
    prefer_right=False,
):
    """
    Bifrost-style square crop:
    - crop_size = ratio * max(h,w)
    - center is biased (e.g., downward) to keep more context near target
    - if image_size is provided, shift crop fully inside image to avoid padding/white border

    bbox: [y1, x1, y2, x2] in the SAME coord system as the image
    returns: crop_coords [ny1, nx1, ny2, nx2], (cy, cx)
    """
    y1, x1, y2, x2 = bbox
    h, w = max(1, y2 - y1), max(1, x2 - x1)

    crop_size = int(round(max(h, w) * float(ratio)))

    # ✅ NEW：限制最大不超过图像尺寸（避免 padding / 白边）
    if image_size is not None:
        W, H = image_size
        max_size = min(W, H)
        crop_size = min(crop_size, max_size)
    crop_size = max(1, crop_size)
    half = crop_size // 2

    # ----- 1) base center on bbox center -----
    cy = (y1 + y2) * 0.5
    cx = (x1 + x2) * 0.5

    # ----- 2) apply bias (keep more bg below / ahead) -----
    # bias is in "crop_size units"
    cy = cy + (bias_y * crop_size)
    cx = cx + (bias_x * crop_size)

    # initial crop (float -> int)
    ny1 = int(round(cy - half))
    nx1 = int(round(cx - half))
    ny2 = ny1 + crop_size
    nx2 = nx1 + crop_size

    # ----- 3) shift into bounds if image_size is given -----
    if image_size is not None:
        W, H = image_size

        dy_top = 0 - ny1
        dy_bot = ny2 - H
        dx_left = 0 - nx1
        dx_right = nx2 - W

        # y shift: prefer pushing upward only when needed, otherwise keep downward context
        if dy_top > 0:
            # crop goes above top -> shift down
            ny1 += dy_top
            ny2 += dy_top
        if dy_bot > 0:
            # crop goes below bottom -> shift up
            # but if prefer_down is True, shift only as much as needed
            ny1 -= dy_bot
            ny2 -= dy_bot

        # x shift
        if dx_left > 0:
            nx1 += dx_left
            nx2 += dx_left
        if dx_right > 0:
            nx1 -= dx_right
            nx2 -= dx_right

        # final clamp safety
        ny1 = max(0, min(H - crop_size, ny1))
        nx1 = max(0, min(W - crop_size, nx1))
        ny2 = ny1 + crop_size
        nx2 = nx1 + crop_size
    
    if image_size is not None:
        W, H = image_size

        ny1 = max(0, ny1)
        nx1 = max(0, nx1)
        ny2 = min(H, ny2)
        nx2 = min(W, nx2)

        # 再修正保证正方形尺寸一致
        crop_h = ny2 - ny1
        crop_w = nx2 - nx1
        crop_size = min(crop_h, crop_w)

        ny2 = ny1 + crop_size
        nx2 = nx1 + crop_size

    return [ny1, nx1, ny2, nx2], (int(round(cy)), int(round(cx)))


def shift_crop_to_bounds(crop_yx, img_w, img_h):
    """
    Shift (translate) crop box to stay inside image as much as possible,
    WITHOUT changing crop size.
    crop_yx: [y1,x1,y2,x2]
    """
    y1, x1, y2, x2 = crop_yx
    ch = y2 - y1
    cw = x2 - x1

    # If crop is smaller than image, shift it fully inside.
    # If crop is larger than image, center it (still will need padding).
    if cw <= img_w:
        if x1 < 0:
            x1 = 0
        if x1 + cw > img_w:
            x1 = img_w - cw
    else:
        x1 = (img_w - cw) // 2

    if ch <= img_h:
        if y1 < 0:
            y1 = 0
        if y1 + ch > img_h:
            y1 = img_h - ch
    else:
        y1 = (img_h - ch) // 2

    x1 = int(x1); y1 = int(y1)
    return [y1, x1, y1 + ch, x1 + cw]

def smart_crop_with_padding(image, crop_coords, pad_value=(0, 0, 0)):
    """
    Crops image using crop_coords. Pads out-of-bound areas with pad_value.
    Ensures the result is exactly the requested size.
    crop_coords: [y1,x1,y2,x2] (can be OOB)
    """
    ny1, nx1, ny2, nx2 = crop_coords
    w_target = nx2 - nx1
    h_target = ny2 - ny1

    if image.mode == "RGB":
        canvas = Image.new("RGB", (w_target, h_target), pad_value)
    elif image.mode == "L":
        canvas = Image.new("L", (w_target, h_target), 0)
    else:
        canvas = Image.new(image.mode, (w_target, h_target))

    img_w, img_h = image.size

    ix1 = max(0, nx1)
    iy1 = max(0, ny1)
    ix2 = min(img_w, nx2)
    iy2 = min(img_h, ny2)

    if ix2 > ix1 and iy2 > iy1:
        crop = image.crop((ix1, iy1, ix2, iy2))
        ox = ix1 - nx1
        oy = iy1 - ny1
        canvas.paste(crop, (ox, oy))

    return canvas

def _clean_binary_mask(mask_uint8, open_ksz=3, close_ksz=3):
    """Morphological open+close to remove isolated pixels and fill tiny holes."""
    k_open = np.ones((open_ksz, open_ksz), np.uint8)
    k_close = np.ones((close_ksz, close_ksz), np.uint8)
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, k_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)
    return cleaned


def _fill_binary_holes(mask_uint8: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask (0/255) via flood-fill from the border."""
    if mask_uint8.ndim != 2:
        raise ValueError(f"mask_uint8 must be 2D, got shape={mask_uint8.shape}")
    h, w = mask_uint8.shape
    if h == 0 or w == 0:
        return mask_uint8
    m = (mask_uint8 > 0).astype(np.uint8) * 255
    inv = 255 - m
    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=(0, 0), newVal=0)
    holes = ff  # remaining 255 regions were enclosed holes
    filled = np.clip(m + holes, 0, 255).astype(np.uint8)
    return filled


def _largest_connected_component(mask_uint8: np.ndarray) -> np.ndarray:
    """Keep largest 8-connected foreground component (255 = fg)."""
    m = (mask_uint8 > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (m * 255).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8) * 255


def regularize_inpaint_mask(
    mask_uint8: np.ndarray,
    mode: str,
    close_ksz: int = 25,
    close_iter: int = 2,
) -> np.ndarray:
    """
    Simplify mask shape before inpainting so occlusion notches on the silhouette
    do not strongly bias the model (e.g. fill concave bites on a product).

    Modes:
      none: no change
      close: morphological close (fills small gaps / smooths boundary)
      convex_hull: convex hull of largest component (fills concavities; strong)
      close_convex_hull: close then convex hull (recommended for partial occlusion)
      fill_holes: flood-fill interior holes only
      fill_holes_close: fill_holes then close
    """
    mode = (mode or "none").strip().lower()
    if mode in ("none", "0", "off", ""):
        return mask_uint8

    m = _largest_connected_component(mask_uint8)

    def _close(u8: np.ndarray) -> np.ndarray:
        k = max(3, int(close_ksz) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, ker, iterations=max(1, int(close_iter)))

    def _hull(u8: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return u8
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 8:
            return u8
        hull = cv2.convexHull(c)
        out = np.zeros_like(u8)
        cv2.fillPoly(out, [hull], 255)
        return out

    if mode == "close":
        m = _close(m)
    elif mode == "convex_hull":
        m = _hull(m)
    elif mode == "close_convex_hull":
        m = _hull(_close(m))
    elif mode == "fill_holes":
        m = _fill_binary_holes(m)
    elif mode == "fill_holes_close":
        m = _close(_fill_binary_holes(m))
    else:
        m = m

    # Always fill enclosed interior holes (donut SAM / ring masks). Idempotent on solid masks.
    m = _fill_binary_holes(m)
    return m


def build_mask_for_ref_depth_fusion(clean_ref_pil: Image.Image) -> np.ndarray:
    """
    Foreground mask on clean_ref (catalog / letterboxed ref), same size as ref_crop.
    Used when --reference_image_path replaces scene SAM ref: scene SAM mask no longer
    matches ref depth geometry; must align mask with ref_depth for adaptive_depth_fusion.
    """
    arr = np.array(clean_ref_pil.convert("RGB"))
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = np.mean(arr.astype(np.float32), axis=2)
    fg = (gray < 250.0).astype(np.uint8) * 255
    fg = _clean_binary_mask(fg)
    bin_mask = (fg > 127).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if n_labels <= 1:
        return fg
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8) * 255


def estimate_ref_depth_safe(depth_estimator, ref_image: Image.Image) -> np.ndarray:
    """
    DepthAnythingV2 lower-bound resizing can explode memory on ultra-thin refs
    (rulers/pencils). Estimate on a square letterbox canvas, then crop back to
    the original ref geometry used by mask/depth fusion.
    """
    if not isinstance(ref_image, Image.Image):
        return depth_estimator.estimate_depth(ref_image)
    ref_w, ref_h = ref_image.size
    min_side = max(1, min(ref_w, ref_h))
    aspect = max(ref_w, ref_h) / float(min_side)
    if aspect <= 4.0 and min_side >= 64:
        return depth_estimator.estimate_depth(ref_image)

    side = max(ref_w, ref_h)
    pad_x = (side - ref_w) // 2
    pad_y = (side - ref_h) // 2
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(ref_image.convert("RGB"), (pad_x, pad_y))
    print(
        f"   [Depth] ref is thin/small ({ref_w}x{ref_h}, aspect={aspect:.1f}); "
        f"estimate on {side}x{side} letterbox and crop back.",
        flush=True,
    )
    depth_canvas = depth_estimator.estimate_depth(canvas)
    return depth_canvas[pad_y:pad_y + ref_h, pad_x:pad_x + ref_w]


def _gemini_upscale_ref(ref_pil, target_long_side=338, max_retries=2):
    """Use Gemini image generation to upscale a small ref_clean image.

    Maintains aspect ratio; scales so the longest side == target_long_side.
    Returns upscaled PIL image or None on failure.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return None

    w, h = ref_pil.size
    if max(w, h) >= target_long_side:
        return ref_pil

    scale = target_long_side / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    bicubic_up = ref_pil.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    bicubic_up.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "Upscale this product photo to higher resolution. "
        "Sharpen edges and recover fine details (text, bezels, buttons, textures). "
        "Do NOT change the object's shape, color, proportions, pose, or background. "
        "Do NOT add, remove, or modify any content. Keep the white background pure white. "
        "Output ONLY the enhanced image at the same dimensions."
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3.1-flash-image-preview:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": b64_img}},
            ]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=90,
            )
            if resp.status_code != 200:
                print(f"   [Gemini upscale] attempt {attempt+1} API error {resp.status_code}", flush=True)
                continue
            result = resp.json()
            parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for p in parts:
                data = p.get("inlineData", {}).get("data")
                if data:
                    img_bytes = base64.b64decode(data)
                    out = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if out.size != (new_w, new_h):
                        out = out.resize((new_w, new_h), Image.LANCZOS)
                    return out
            print(f"   [Gemini upscale] attempt {attempt+1} no image in response", flush=True)
        except Exception as e:
            print(f"   [Gemini upscale] attempt {attempt+1} failed: {e}", flush=True)
    return None


def place_ref_mask_at_target(ref_mask, target_bbox_yxyx, canvas_shape):
    """Place the reference mask at the target location, returning a full-image mask."""
    H, W = canvas_shape[:2]
    y1, x1, y2, x2 = target_bbox_yxyx
    tgt_h, tgt_w = max(1, y2 - y1), max(1, x2 - x1)

    scaled_mask = cv2.resize(ref_mask.astype(np.uint8), (tgt_w, tgt_h),
                              interpolation=cv2.INTER_NEAREST)
    scaled_mask = (scaled_mask > 0).astype(np.uint8) * 255
    scaled_mask = _clean_binary_mask(scaled_mask)

    canvas = np.zeros((H, W), dtype=np.uint8)
    py1, px1 = max(0, y1), max(0, x1)
    py2, px2 = min(H, y2), min(W, x2)
    sy1, sx1 = py1 - y1, px1 - x1
    sh = py2 - py1
    sw = px2 - px1
    if sh > 0 and sw > 0:
        canvas[py1:py2, px1:px2] = scaled_mask[sy1:sy1 + sh, sx1:sx1 + sw]
    return canvas


def adaptive_depth_fusion(
    back_depth, ref_depth, ref_mask, target_bbox_yxyx,
    target_depth_range=None, coverage_threshold=0.5, ref_depth_bias=0.0,
):
    """
    Adaptive depth fusion with quality-aware fallback.

    Key fix: inpainting mask is ALWAYS derived from reference shape (not depth comparison).
    Depth fusion only affects depth conditioning, never the mask.

    Returns: (fused_depth, inpaint_mask_uint8, use_depth_control_flag)

    Strategy:
      1. Compute inpainting mask from reference shape (independent of depth)
      2. Try occlusion-aware fusion with dynamic margin
      3. Quality gate: if coverage < threshold, fall back to simple paste or no depth
    """
    H, W = back_depth.shape[:2]
    y1, x1, y2, x2 = target_bbox_yxyx
    tgt_h, tgt_w = max(1, y2 - y1), max(1, x2 - x1)

    norm_back = np.zeros_like(back_depth, dtype=np.float32)
    cv2.normalize(back_depth, norm_back, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    norm_ref = np.zeros_like(ref_depth, dtype=np.float32)
    if target_depth_range:
        d_min, d_max = target_depth_range
        cv2.normalize(ref_depth, norm_ref, alpha=d_min, beta=d_max,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        cv2.normalize(ref_depth, norm_ref, alpha=180, beta=240,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    scaled_ref = cv2.resize(norm_ref, (tgt_w, tgt_h))
    if abs(float(ref_depth_bias)) > 1e-6:
        scaled_ref = np.clip(scaled_ref + float(ref_depth_bias), 0.0, 255.0)
    scaled_mask = cv2.resize(ref_mask.astype(np.float32), (tgt_w, tgt_h))
    ref_valid_raw = (scaled_mask > 0.5).astype(np.uint8)
    ref_valid_raw = _clean_binary_mask(ref_valid_raw * 255) > 127
    ref_valid = ref_valid_raw

    py1, px1 = max(0, y1), max(0, x1)
    py2, px2 = min(H, y2), min(W, x2)
    sy1, sx1 = py1 - y1, px1 - x1
    sh = py2 - py1
    sw = px2 - px1
    if sh <= 0 or sw <= 0:
        inpaint_mask = place_ref_mask_at_target(ref_mask, target_bbox_yxyx, back_depth.shape)
        return norm_back, inpaint_mask, False, 0.0, "disabled_oob"

    patch_ref = scaled_ref[sy1:sy1 + sh, sx1:sx1 + sw]
    patch_mask = ref_valid[sy1:sy1 + sh, sx1:sx1 + sw]
    patch_bg = norm_back[py1:py2, px1:px2].copy()

    total_ref_pixels = int(np.sum(patch_mask))
    if total_ref_pixels == 0:
        inpaint_mask = place_ref_mask_at_target(ref_mask, target_bbox_yxyx, back_depth.shape)
        return norm_back, inpaint_mask, False, 0.0, "disabled_empty_ref"

    # Soft alpha for depth blending: smooth transition at mask border instead
    # of hard paste that creates depth discontinuities / jagged edges.
    depth_feather_px = max(2, int(min(tgt_h, tgt_w) * 0.04))
    mask_f = patch_mask.astype(np.uint8)
    dist_inside = cv2.distanceTransform(mask_f, cv2.DIST_L2, 3)
    soft_alpha = np.clip(dist_inside / max(1.0, float(depth_feather_px)), 0.0, 1.0)
    soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=max(1.0, depth_feather_px * 0.4))
    soft_alpha = np.clip(soft_alpha, 0.0, 1.0)

    # Strategy B: Occlusion-aware with dynamic margin
    ref_range = float(np.ptp(patch_ref[patch_mask])) if total_ref_pixels > 1 else 30.0
    margin = max(15.0, ref_range * 0.3)
    is_not_occluded = (patch_ref >= (patch_bg - margin))
    occlusion_valid = np.logical_and(patch_mask, is_not_occluded)
    occlusion_pixels = int(np.sum(occlusion_valid))
    coverage = occlusion_pixels / max(1, total_ref_pixels)

    inpaint_mask = place_ref_mask_at_target(ref_mask, target_bbox_yxyx, back_depth.shape)

    def _soft_paste(base, patch_vals, alpha_map):
        out = base.copy()
        region = out[py1:py2, px1:px2].astype(np.float32)
        ref_f = patch_vals.astype(np.float32)
        blended = region * (1.0 - alpha_map) + ref_f * alpha_map
        out[py1:py2, px1:px2] = blended.astype(base.dtype)
        return out

    if coverage >= coverage_threshold:
        occ_alpha = soft_alpha.copy()
        occ_alpha[~occlusion_valid] = 0.0
        fused = _soft_paste(norm_back, patch_ref, occ_alpha)
        print(f"    Depth fusion: occlusion-aware (coverage={coverage:.2f}, feather={depth_feather_px}px)")
        return fused, inpaint_mask, True, float(coverage), "occlusion"
    elif coverage >= 0.2:
        fused = _soft_paste(norm_back, patch_ref, soft_alpha)
        print(f"    Depth fusion: simple paste (coverage={coverage:.2f} < {coverage_threshold}, feather={depth_feather_px}px)")
        return fused, inpaint_mask, True, float(coverage), "simple"
    else:
        print(f"    Depth fusion: DISABLED (coverage={coverage:.2f})")
        return norm_back, inpaint_mask, False, float(coverage), "disabled_low_coverage"

# ==============================================================================
# 3. Tool Classes
# ==============================================================================
def _load_feedback_pair_for_task(feedback_json_path: str, task_id: str, pair_index: int = 0):
    """Return (entry, pair) from gemini_scores JSON, or (None, None) / (entry, None) if missing."""
    fp = str(feedback_json_path or "").strip()
    if not fp or not os.path.isfile(fp):
        return None, None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None
    entry = None
    for it in data if isinstance(data, list) else []:
        if str(it.get("task_id", "")) == str(task_id):
            entry = it
            break
    if not entry:
        return None, None
    pairs = entry.get("pairs", []) or []
    if not pairs or int(pair_index) < 0 or int(pair_index) >= len(pairs):
        return entry, None
    return entry, pairs[int(pair_index)]


def _load_feedback_entry_for_task(feedback_json_path: str, task_id: str):
    fp = str(feedback_json_path or "").strip()
    if not fp or not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    for it in data if isinstance(data, list) else []:
        if str(it.get("task_id", "")) == str(task_id):
            return it
    return None


def _unique_eval_names_from_feedback_entry(entry) -> list:
    seen = set()
    out = []
    for p in (entry or {}).get("pairs", []) or []:
        for key in ("object_a", "object_b"):
            n = str((p.get(key) or {}).get("name", "")).strip()
            if not n:
                continue
            k = n.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(n)
    return out


def _feedback_scene_object_names(entry) -> list:
    names = _unique_eval_names_from_feedback_entry(entry)
    if names:
        return names
    sp = (entry or {}).get("scene_prefilter") or {}
    raw = sp.get("prefilter_primary_labels") or []
    out = []
    seen = set()
    for x in raw if isinstance(raw, list) else []:
        n = str(x).strip()
        if not n:
            continue
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out


def _load_scene_prefilter_from_feedback_json(feedback_json_path: str, task_id: str, image_path: str = ""):
    entry = _load_feedback_entry_for_task(feedback_json_path, task_id)
    sp = (entry or {}).get("scene_prefilter")
    if not isinstance(sp, dict):
        return None
    out = copy.deepcopy(sp)
    out["prefilter_source"] = "feedback_json"
    out["feedback_json_path_used"] = os.path.abspath(str(feedback_json_path))
    out["task_id"] = str(task_id)
    if image_path:
        out["prefilter_image_path"] = os.path.abspath(str(image_path))
    return out


def _missing_scene_prefilter_record(feedback_json_path: str, task_id: str, image_path: str = ""):
    out = {
        "prefilter_enabled": True,
        "skip_size_correction": False,
        "api_failed": True,
        "prefilter_source": "feedback_json",
        "prefilter_missing_from_score_json": True,
        "feedback_json_path_used": os.path.abspath(str(feedback_json_path or "")),
        "task_id": str(task_id),
    }
    if image_path:
        out["prefilter_image_path"] = os.path.abspath(str(image_path))
    return out


def _should_apply_bbox_patch_from_meta(meta: dict) -> bool:
    """Patch layout_final only for rounds that actually wrote an edited image."""
    if not isinstance(meta, dict):
        return False
    if (meta.get("rescore") or {}).get("did_rollback") is True:
        return False
    fb = meta.get("_feedback_blob") or meta.get("feedback_blob") or {}
    if isinstance(fb, dict) and fb.get("skip_correction") is True:
        return False
    return str(meta.get("status") or "") == "ok"


def _apply_feedback_canonical_names_to_detected(
    analyzer,
    detected_list,
    image_path,
    probe_w,
    probe_h,
    feedback_json_path,
    task_id,
    pair_index,
    use_feedback_planner: int,
):
    """Ground all distinct names from all pairs, then rename each detection to best IoU label."""
    if int(use_feedback_planner) != 1:
        return
    fp = str(feedback_json_path or "").strip()
    if not fp or not os.path.isfile(fp) or not detected_list:
        return
    entry = _load_feedback_entry_for_task(feedback_json_path, task_id)
    if not entry:
        return
    eval_names = _unique_eval_names_from_feedback_entry(entry)
    if not eval_names:
        return
    grounded = []
    for n in eval_names:
        raw = analyzer.detect_target_bbox(image_path, n)
        if raw is None:
            continue
        px = bbox_raw_to_pixels(raw, probe_w, probe_h)
        if px is None:
            continue
        grounded.append((n, px))
    if not grounded:
        print(
            "[feedback] Could not ground any JSON object names; keeping scene detection names.",
            flush=True,
        )
        return
    iou_thresh = 0.12
    for d in detected_list:
        best_n = None
        best_iou = 0.0
        for n, px in grounded:
            iou = bbox_iou_yxyx(d["bbox"], px)
            if iou > best_iou:
                best_iou = iou
                best_n = n
        if best_iou < iou_thresh or best_n is None:
            continue
        old = d["name"]
        if old != best_n:
            d["name"] = best_n
            print(
                f"   [feedback] label '{old}' -> '{best_n}' "
                f"(IoU={best_iou:.3f} vs JSON grounding for '{best_n}')",
                flush=True,
            )
    print(
        f"[feedback] JSON labels grounded for IoU map (task_id={task_id}): {eval_names}",
        flush=True,
    )


# Global clamp (single-step linear scale). Must envelope evaluate_genscale_gemini tier bounds.
# Aligned with scripts/evaluate_genscale_gemini.py::_TIER_DIRECT on rubric object_b.
_TIER_DIRECT_B = {
    1: (1.6, 3.0),
    2: (1.2, 1.6),
    4: (0.625, 0.84),
    5: (0.33, 0.625),
}
_SCALE_HARD_MAX = 3.0
_SCALE_HARD_MIN = 1.0 / _SCALE_HARD_MAX  # ≈ 0.333; allows tier-5 shrink to 0.33


def _tier_gate_lo_hi_for_target(size_score: int, target_is_object_b: bool):
    """
    Return (lo, hi) for linear scale_factor on the edit target.
    ``target_is_object_b`` matches feedback code's ``match_inverted`` (True => target is object_b).
    """
    if size_score not in _TIER_DIRECT_B:
        return None, None
    lo_b, hi_b = _TIER_DIRECT_B[size_score]
    if target_is_object_b:
        return float(lo_b), float(hi_b)
    inv_lo = 1.0 / float(hi_b)
    inv_hi = 1.0 / float(lo_b)
    if inv_lo > inv_hi:
        inv_lo, inv_hi = inv_hi, inv_lo
    return float(inv_lo), float(inv_hi)


def _build_rescore_prompt(object_a: str, len_a_cm: float, object_b: str, len_b_cm: float,
                          scenario: str = "") -> str:
    """
    Build the exact same Gemini judge prompt used in evaluate_genscale_task1_gemini.py,
    so post-correction re-scores are comparable to the original evaluation JSON.
    """
    scenario_block = ""
    if (scenario or "").strip() == "S2_Extreme_Contrast":
        scenario_block = (
            "### SCENARIO CONTEXT (same depth plane):\n"
            "This benchmark row is a coplanar layout: objects are meant on the same ground plane "
            "with little depth separation. Use the stated typical lengths (cm) as the primary "
            "reference for whether Object B's scale is proportionate to Object A. "
            "Do not justify a wrong relative size with 'farther from the camera' unless the image "
            "clearly shows large depth separation. Occlusion, foreshortening, and flexible-object "
            "pose still apply.\n\n"
        )
    return (
        "You are an expert physical spatial reasoning engine and a professional photography adjudicator. "
        "Your task is to evaluate the physical size accuracy of objects in the provided generated image.\n"
        "Your scoring should reflect typical human judgement: be tolerant to plausible variation and "
        "only use extreme scores when the error is unmistakably huge.\n\n"
        "### GROUND TRUTH REFERENCE:\n"
        f"- Object A (Anchor): {object_a}, Typical physical longest edge: {len_a_cm:.2f} cm.\n"
        f"- Object B (Target): {object_b}, Typical physical longest edge: {len_b_cm:.2f} cm.\n\n"
        f"{scenario_block}"
        "### EVALUATION PROTOCOL:\n"
        "Please analyze the image step-by-step to avoid common computer vision pitfalls:\n"
        "1. Detection: Are BOTH Object A and Object B clearly identifiable in the image?\n"
        "2. 3D Spatial Layout (Depth): Where are they placed relative to the camera? "
        "Remember the rule of perspective: objects further away appear smaller.\n"
        "3. Orientation & Foreshortening: Are the objects tilted, rolled up, folded, or "
        "pointing toward the camera? In such cases the visible 2D length can be much shorter "
        "even when the real-world scale is correct.\n"
        "4. Configuration realism: For flexible objects (e.g., towels, blankets, cloth), it is "
        "normal to appear folded/rumpled. Do NOT penalize a correct-scale towel as 'undersized' "
        "just because it is folded.\n"
        "5. Occlusion: Is a significant part of either object hidden behind something else?\n\n"
        "### FINAL JUDGMENT:\n"
        f"Assume Object A is its real-world physical size in 3D space. Accounting for depth, "
        f"perspective, occlusion, and realistic configuration (folding/rolling/foreshortening), "
        f"how accurate is the size of Object B compared to its stated typical length of "
        f"{len_b_cm:.2f} cm?\n"
        "Use these human-aligned tolerance guidelines:\n"
        "- Score 3 (Proportionate): looks basically right; within ~20% of expected scale OR "
        "plausibly explained by pose/configuration.\n"
        "- Score 2/4 (Slightly under/over): clearly off, but still believable in a real photo; "
        "roughly 20%–60% off.\n"
        "- Score 1/5 (Severely under/over): only if the error is comically obvious (toy-sized or "
        "giant-prop sized), typically >~100% off (more than ~2× too big or <~0.5× too small).\n"
        "Reserve 1 and 5 only for extreme mismatches; use 2 or 4 when the error is clear but not "
        "absurd.\n"
        "Select exactly one category from the 1-5 scale below:\n"
        "1: Severely Undersized\n"
        "2: Slightly Undersized\n"
        "3: Proportionate\n"
        "4: Slightly Oversized\n"
        "5: Severely Oversized\n\n"
        "### OUTPUT FORMAT:\n"
        "You MUST output your response in valid JSON format. Do not include markdown code blocks.\n"
        "CRITICAL: Keep BOTH reasoning fields extremely short (<= 25 words each). "
        "Do NOT use ellipses (...).\n"
        "{\n"
        '  "reasoning_detection": "...",\n'
        '  "reasoning_depth_and_perspective": "...",\n'
        '  "both_objects_present": true,\n'
        '  "size_score": 3\n'
        "}\n"
    )


def _gemini_rescore_pair(
    image_path: str,
    object_a: str,
    len_a_cm: float,
    object_b: str,
    len_b_cm: float,
    api_key: str,
    model_version: str = "gemini-3.1-pro-preview",
    scenario: str = "",
    timeout: int = 60,
) -> int | None:
    """
    Call the same Gemini judge used in evaluate_genscale_task1_gemini.py and return size_score,
    or None if the call fails. Used for post-correction re-scoring.
    """
    import re as _re
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        prompt = _build_rescore_prompt(object_a, len_a_cm, object_b, len_b_cm, scenario)
        extra = (
            "\n\nCRITICAL: Output MUST be minified JSON in a SINGLE LINE. "
            "The JSON MUST include all required keys and end with a closing brace '}'."
        )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_version}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt + extra},
                    {"inlineData": {"mimeType": "image/png", "data": b64}},
                ]
            }],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
        }
        resp = requests.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"   [rescore] HTTP {resp.status_code}", flush=True)
            return None
        data = resp.json()
        text = "".join(
            p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
            if isinstance(p, dict)
        ).strip()
        # strip markdown fences
        if text.startswith("```"):
            text = _re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = _re.sub(r"\n?```$", "", text).strip()
        obj = json.loads(text)
        score = obj.get("size_score")
        if isinstance(score, int) and 1 <= score <= 5:
            return score
        return None
    except Exception as e:
        print(f"   [rescore] exception: {e}", flush=True)
        return None


class GeminiBBoxAnalyzer:
    def __init__(
        self,
        api_key=None,
        model_version="gemini-3.1-pro-preview",
        max_rounds=3,
        feedback_json_path: str = "",
        use_feedback_planner: int = 1,
        feedback_lookup_task_id: str = "",
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        self.model_version = model_version
        self.max_rounds = max_rounds
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_version}:generateContent?key={self.api_key}"
        self.feedback_json_path = str(feedback_json_path or "").strip()
        self.use_feedback_planner = int(use_feedback_planner)
        self.feedback_lookup_task_id = str(feedback_lookup_task_id or "").strip() or None

    def _call_gemini(self, prompt, encoded_image, mime_type: str = "image/png"):
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": encoded_image}}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"}
        }
        try:
            response = requests.post(self.url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            if response.status_code != 200:
                return None
            result = response.json()
            if "candidates" not in result or not result["candidates"]:
                return None
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            import re
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                return None
            parsed = json.loads(json_match.group(0))
            normalized = {}
            for k, v in parsed.items():
                k_lower = k.lower()
                if "original" in k_lower and "box" in k_lower:
                    normalized["original_bbox"] = v
                elif "new" in k_lower and "box" in k_lower:
                    normalized["new_bbox"] = v
                else:
                    normalized[k] = v
            return normalized
        except Exception:
            return None

    def identify_scaling_targets(self, image_path):
        print(f"\n[Gemini] Identifying scaling anomalies in scene...")
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = """
        Analyze this image. Identify the TWO distinct objects involved in the scene.

        Task:
        1. "target_object": ALWAYS select the SMALLER tangible object (e.g., tennis ball, rabbit, bottle). This is the object we will resize.
        2. "reference_object": Select the larger, dominant object (e.g., soccer ball, elephant, table).
        3. Both objects must be countable entities with clear boundaries. NEVER output background/stuff classes such as ground/floor/wall/sky/road.

        Output JSON: { "target_object": "name", "reference_object": "name", "reasoning": "why" }
        """
        res = self._call_gemini(prompt, encoded_string)
        if res and "target_object" in res:
            tgt = str(res.get("target_object", "")).strip()
            ref = str(res.get("reference_object", "context")).strip()
            banned_tokens = {"background", "ground", "floor", "wall", "road", "sky", "terrain", "pavement", "sidewalk", "grass", "sand", "water"}
            def _is_bad_name(name: str) -> bool:
                n = name.lower()
                return (not n) or any(tok in n for tok in banned_tokens)
            if _is_bad_name(tgt) or _is_bad_name(ref):
                print(f"   -> Detection produced non-object label(s): target='{tgt}', ref='{ref}'.", flush=True)
                return None, None
            print(f"   -> Detected: Target='{tgt}' (Ref: {ref})")
            return tgt, ref

        print("   -> Detection failed.")
        return None, None

    def get_adjustment_plan(self, image_path, target_object, reference_object):
        print(f"\n[Gemini] Analyzing: '{target_object}' vs '{reference_object}'...")
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        task_id_local = self.feedback_lookup_task_id or os.path.splitext(
            os.path.basename(image_path)
        )[0]

        def _extract_feedback_hint(task_id: str):
            """
            Return (hint, feedback_blob, skip_entire_correction).
            skip_entire_correction True when name-matched pair has size_score==3 (already correct).
            """
            if int(self.use_feedback_planner) != 1:
                return "unknown", None, False
            fp = str(self.feedback_json_path or "").strip()
            if not fp or not os.path.isfile(fp):
                return "unknown", None, False
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = None
                for it in data if isinstance(data, list) else []:
                    if str(it.get("task_id", "")) == str(task_id):
                        entry = it
                        break
                if not entry:
                    return "unknown", None, False
                pairs = entry.get("pairs", []) or []
                best_pair = None
                match_inverted = False
                name_matched = False

                tgt_l = str(target_object).lower().strip()
                ref_l = str(reference_object).lower().strip()

                def _names_match(n1, n2):
                    """Fuzzy match: exact, or one is a substring of the other."""
                    return n1 == n2 or n1 in n2 or n2 in n1

                # Pass 1: exact match (both target and reference).
                for p in pairs:
                    a = str((p.get("object_a") or {}).get("name", "")).lower().strip()
                    b = str((p.get("object_b") or {}).get("name", "")).lower().strip()
                    if _names_match(a, tgt_l) and _names_match(b, ref_l):
                        best_pair = p
                        match_inverted = False
                        name_matched = True
                        break
                    if _names_match(b, tgt_l) and _names_match(a, ref_l):
                        best_pair = p
                        match_inverted = True
                        name_matched = True
                        break

                # Pass 2: target-only match (reference may differ — still tells us
                # which side of the score the target is on).
                if best_pair is None:
                    for p in pairs:
                        a = str((p.get("object_a") or {}).get("name", "")).lower().strip()
                        b = str((p.get("object_b") or {}).get("name", "")).lower().strip()
                        if _names_match(a, tgt_l):
                            best_pair = p
                            match_inverted = False
                            break
                        if _names_match(b, tgt_l):
                            best_pair = p
                            match_inverted = True
                            break

                # Pass 3: hard fallback — use pairs[0] but infer inversion from
                # which side of the pair is closer to target_object name.
                if best_pair is None and pairs:
                    best_pair = pairs[0]
                    a0 = str((best_pair.get("object_a") or {}).get("name", "")).lower().strip()
                    b0 = str((best_pair.get("object_b") or {}).get("name", "")).lower().strip()
                    # If target is more similar to object_b than object_a, no inversion needed.
                    # Simple heuristic: check substring containment.
                    tgt_in_b = (tgt_l in b0 or b0 in tgt_l)
                    tgt_in_a = (tgt_l in a0 or a0 in tgt_l)
                    if tgt_in_b and not tgt_in_a:
                        match_inverted = True   # target is on the b-side, score directly applies
                    else:
                        match_inverted = False  # target is on the a-side, invert as before

                score = best_pair.get("size_score", None)
                hint_b = "unknown"
                try:
                    s = int(score)
                except Exception:
                    s = None

                # Name-matched pair with score 3: eval says relative scale is already correct — skip pipeline.
                if name_matched and s == 3:
                    fb_blob_skip = {
                        "task_id": str(entry.get("task_id", task_id)),
                        "size_score": best_pair.get("size_score"),
                        "both_objects_present": best_pair.get("both_objects_present"),
                        "object_a": best_pair.get("object_a"),
                        "object_b": best_pair.get("object_b"),
                        "reasoning_detection": best_pair.get("reasoning_detection"),
                        "reasoning_depth_and_perspective": best_pair.get("reasoning_depth_and_perspective"),
                        "size_score_definition": "object_b judged relative to object_a: 1 too small, 2 slightly small, 3 correct, 4 slightly large, 5 too large",
                        "skip_correction": True,
                        "skip_reason": "feedback_size_score_3_name_matched",
                        "pair_match_inverted_ab": bool(match_inverted),
                    }
                    return "unknown", fb_blob_skip, True

                tier_lo = None
                tier_hi = None
                if s == 1:
                    hint_b = "enlarge"
                elif s == 2:
                    hint_b = "enlarge"
                elif s == 3:
                    hint_b = "unknown"
                elif s == 4:
                    hint_b = "shrink"
                elif s == 5:
                    hint_b = "shrink"
                if s in _TIER_DIRECT_B:
                    tier_lo, tier_hi = _tier_gate_lo_hi_for_target(s, match_inverted)

                hint = hint_b
                if not match_inverted:
                    if hint == "shrink":
                        hint = "enlarge"
                    elif hint == "enlarge":
                        hint = "shrink"

                if s is None:
                    fb_text = " ".join(
                        [
                            str(best_pair.get("reasoning_detection", "")),
                            str(best_pair.get("reasoning_depth_and_perspective", "")),
                        ]
                    ).lower()
                    hint = "unknown"
                    if any(k in fb_text for k in ("too large", "unnaturally large", "oversized", "disproportionately large")):
                        hint = "shrink"
                    if any(k in fb_text for k in ("too small", "unnaturally small", "tiny", "undersized", "disproportionately small")):
                        hint = "enlarge"
                    if not match_inverted:
                        if hint == "shrink":
                            hint = "enlarge"
                        elif hint == "enlarge":
                            hint = "shrink"
                        tier_lo = None
                        tier_hi = None

                fb_blob = {
                    "task_id": str(entry.get("task_id", task_id)),
                    "size_score": best_pair.get("size_score"),
                    "both_objects_present": best_pair.get("both_objects_present"),
                    "object_a": best_pair.get("object_a"),
                    "object_b": best_pair.get("object_b"),
                    "reasoning_detection": best_pair.get("reasoning_detection"),
                    "reasoning_depth_and_perspective": best_pair.get("reasoning_depth_and_perspective"),
                    "size_score_definition": "object_b judged relative to object_a: 1 too small, 2 slightly small, 3 correct, 4 slightly large, 5 too large",
                    "size_score_tier_gate": {
                        "lo": tier_lo,
                        "hi": tier_hi,
                        "note": "Clamp scale_factor to [lo,hi] on edit target; aligned with evaluate_genscale_gemini._TIER_DIRECT.",
                    },
                    "direction_hint": hint,
                    "pair_match_inverted_ab": bool(match_inverted),
                    "name_matched_pair": bool(name_matched),
                    "assumption": "direction_hint computed from size_score; if target!=object_b then invert direction because score is about b relative to a",
                }
                return hint, fb_blob, False
            except Exception:
                return "unknown", None, False

        fb_hint, fb_blob, skip_feedback_3 = _extract_feedback_hint(task_id_local)
        if skip_feedback_3:
            print(
                "  [feedback] size_score=3 (name-matched pair): skip Gemini planning & correction; keep original.",
                flush=True,
            )
            return {
                "_skip_correction": True,
                "_skip_reason": "feedback_size_score_3",
                "_feedback_blob": fb_blob,
            }

        print("  Step 1: Initial Detection...", end="", flush=True)
        prompt_detect = (
            "Ground the object using the EXACT evaluator label below (do not rename it). "
            "The instance may match common synonyms in the scene; localize the object for that benchmark label. "
            f"Label for the object to resize: '{target_object}'. "
            f"Output TIGHT bbox [ymin, xmin, ymax, xmax] around that object only. "
            f'JSON: {{ "bbox": [...] }}'
        )
        res_detect = self._call_gemini(prompt_detect, encoded_string)
        if not res_detect:
            return None

        raw_bbox = res_detect.get("bbox") or res_detect.get("original_bbox")
        if raw_bbox and isinstance(raw_bbox[0], list):
            raw_bbox = raw_bbox[0]
        current_bbox = raw_bbox
        print(f" Found: {current_bbox}")

        print("  Step 2: Refinement...", end="", flush=True)
        prompt_refine = (
            f"Tighten bbox {current_bbox} to fully enclose the object with evaluator label '{target_object}'. "
            f'Fix missing parts or excess margin. JSON: {{ "bbox": [...] }}'
        )
        res_refine = self._call_gemini(prompt_refine, encoded_string)
        if res_refine:
            raw_refined = res_refine.get("bbox") or current_bbox
            if isinstance(raw_refined[0], list):
                raw_refined = raw_refined[0]
            current_bbox = raw_refined
        print(f" Refined: {current_bbox}")

        fb_rule = ""
        if fb_hint == "shrink":
            fb_rule = (
                "Feedback constraint: the target object appears TOO LARGE relative to the reference. "
                "You MUST shrink the target (scale_factor < 1.0)."
            )
        elif fb_hint == "enlarge":
            fb_rule = (
                "Feedback constraint: the target object appears TOO SMALL relative to the reference. "
                "You MUST enlarge the target (scale_factor > 1.0)."
            )
        if fb_blob is not None and fb_hint in ("shrink", "enlarge"):
            oa = (fb_blob.get("object_a") or {}).get("name")
            ob = (fb_blob.get("object_b") or {}).get("name")
            inv = fb_blob.get("pair_match_inverted_ab")
            sc = fb_blob.get("size_score")
            rd = str(fb_blob.get("reasoning_depth_and_perspective", "") or "").strip()
            if len(rd) > 220:
                rd = rd[:220] + "..."
            print(
                f"  [feedback] USED task_id={task_id_local} size_score={sc} "
                f"pair=({oa} -> {ob}) inverted_ab={inv} => hint={fb_hint}",
                flush=True,
            )
            if rd:
                print(f"  [feedback] depth_reason: {rd}", flush=True)
        elif int(self.use_feedback_planner) == 1 and str(self.feedback_json_path or "").strip():
            print(
                f"  [feedback] NOT used (hint={fb_hint}) for task_id={task_id_local}. "
                "Check object name mismatch / missing keywords in reasoning.",
                flush=True,
            )

        # Detect reference bbox to provide stronger physical-scale constraints.
        ref_bbox = None
        try:
            prompt_ref_detect = (
                f"Task: Detect '{reference_object}'. Output TIGHT bbox [ymin, xmin, ymax, xmax]. "
                f'JSON: {{ "bbox": [...] }}'
            )
            res_ref_detect = self._call_gemini(prompt_ref_detect, encoded_string)
            if res_ref_detect:
                raw_ref_box = res_ref_detect.get("bbox") or res_ref_detect.get("original_bbox")
                if raw_ref_box and isinstance(raw_ref_box[0], list):
                    raw_ref_box = raw_ref_box[0]
                if raw_ref_box:
                    ref_bbox = [float(x) for x in raw_ref_box]
        except Exception:
            ref_bbox = None

        print("  Step 3: Physics & Anchor...", end="", flush=True)
        prompt_physics = f"""
You are a perceptual size realism analyst.

Goal: adjust ONLY the target object's size so the scene looks physically plausible.
Target (to resize): '{target_object}'
Reference (keep unchanged): '{reference_object}'

Known geometry:
- Target bbox [ymin,xmin,ymax,xmax]: {current_bbox}
- Reference bbox [ymin,xmin,ymax,xmax]: {ref_bbox if ref_bbox is not None else "unknown"}

CRITICAL reasoning requirements:
1) Use REAL-WORLD typical size priors (e.g., typical lengths in cm) for BOTH objects when available.
2) Explicitly account for perspective / depth:
   - Nearby objects appear larger; far objects appear smaller.
   - If the target is closer than the reference, a large pixel bbox may still be plausible.
   - If the target is farther than the reference, a large pixel bbox is MORE suspicious.
3) Estimate ratios using a LINEAR characteristic length (e.g., sqrt(area) of bbox, or max(w,h)),
   not area ratio. Always report ratio_type="linear".
4) Do NOT change reference scale. Only choose (scale_factor, anchor_point) for target.

Think step-by-step and be conservative:
Step A: Estimate CURRENT apparent linear ratio(target/reference) from the image
        using bbox geometry and any depth ordering cues (foreground/background, occlusions, contact with surfaces).
Step B: Estimate DESIRED plausible real-world linear ratio(target/reference)
        using real-world priors (sizes) and adjusting for the estimated depth ordering.
Step C: Compute scale_factor ~= desired_ratio / current_ratio.
Step D: Choose anchor_point to preserve physical contact (e.g., on ground/table: BOTTOM_CENTER;
        hanging/ceiling: TOP_CENTER; attached to side: LEFT/RIGHT_CENTER; floating: CENTER).

{fb_rule if fb_rule else ""}
{("Evaluator feedback JSON snippet (includes typical_len_cm priors): " + json.dumps(fb_blob, ensure_ascii=False)) if fb_blob else ""}

Return ONLY valid JSON:
{{
  "need_correction": true,
  "current_ratio": 0.30,
  "desired_ratio": 0.10,
  "ratio_type": "linear",
  "scale_factor": 0.8,
  "anchor_point": "BOTTOM_CENTER",
  "confidence": 0.0,
  "reason": "Explain current_ratio (image+depth) and desired_ratio (real-world priors) briefly."
}}
"""
        res_physics = self._call_gemini(prompt_physics, encoded_string)
        if not res_physics:
            return None

        current_ratio = float(res_physics.get("current_ratio", 0.0) or 0.0)
        desired_ratio = float(res_physics.get("desired_ratio", 0.0) or 0.0)
        ratio_type = str(res_physics.get("ratio_type", "linear") or "linear").strip().lower()
        fallback_scale = float(res_physics.get("scale_factor", 1.0) or 1.0)
        anchor_point = res_physics.get("anchor_point", "BOTTOM_CENTER")
        need_correction = bool(res_physics.get("need_correction", True))
        confidence = float(res_physics.get("confidence", 0.0) or 0.0)

        # Modal-style robust scale execution:
        # prefer ratio-derived scale; fallback to direct scale if ratios are missing/noisy.
        scale_factor = fallback_scale
        if current_ratio > 1e-8 and desired_ratio > 1e-8:
            ratio = desired_ratio / current_ratio
            scale_factor = math.sqrt(max(1e-8, ratio)) if ratio_type == "area" else ratio
        if not math.isfinite(scale_factor):
            scale_factor = fallback_scale

        # If Gemini says no correction and predicted scale is near identity, keep 1.0.
        if (not need_correction) and abs(scale_factor - 1.0) < 0.12:
            scale_factor = 1.0

        # Direction gate: enforce feedback hint to avoid flip-flopping.
        if fb_hint == "shrink":
            scale_factor = min(float(scale_factor), 0.999)
        elif fb_hint == "enlarge":
            scale_factor = max(float(scale_factor), 1.001)
        # Five-tier gate: clamp to [lo, hi] (aligned with evaluate_genscale_gemini).
        if fb_blob is not None:
            gate = fb_blob.get("size_score_tier_gate") or {}
            glo, ghi = gate.get("lo"), gate.get("hi")
            try:
                lo_f = float(glo) if glo is not None else None
                hi_f = float(ghi) if ghi is not None else None
            except (TypeError, ValueError):
                lo_f, hi_f = None, None
            if lo_f is not None and hi_f is not None:
                if lo_f > hi_f:
                    lo_f, hi_f = hi_f, lo_f
                scale_factor = max(lo_f, min(hi_f, float(scale_factor)))
            else:
                gmode = gate.get("mode")
                gscale = gate.get("scale")
                try:
                    gscale_f = float(gscale) if gscale is not None else None
                except (TypeError, ValueError):
                    gscale_f = None
                if gmode == "min" and gscale_f is not None:
                    scale_factor = max(float(scale_factor), gscale_f)
                elif gmode == "max" and gscale_f is not None:
                    scale_factor = min(float(scale_factor), gscale_f)

        print(
            " Done. "
            f"(anchor={anchor_point}, scale={scale_factor:.3f}, "
            f"cur_ratio={current_ratio:.4f}, des_ratio={desired_ratio:.4f}, conf={confidence:.2f})"
        )

        return {
            "original_bbox": current_bbox,
            "scale_factor": float(scale_factor),
            "anchor_point": anchor_point,
            "_feedback_hint": fb_hint,
            "_feedback_blob": fb_blob,
        }

    def detect_target_bbox(self, image_path, target_object):
        """Detect + refine bbox only (no scale planning)."""
        print(f"\n[Gemini] Detecting bbox only for '{target_object}'...", flush=True)
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        prompt_detect = (
            "Ground the object using the EXACT evaluator label below. "
            "The instance may match synonyms in the image; localize the object for that label. "
            f"Label: '{target_object}'. Output TIGHT bbox [ymin, xmin, ymax, xmax]. "
            f'JSON: {{ "bbox": [...] }}'
        )
        res_detect = self._call_gemini(prompt_detect, encoded_string)
        if not res_detect:
            return None

        raw_bbox = res_detect.get("bbox") or res_detect.get("original_bbox")
        if raw_bbox and isinstance(raw_bbox[0], list):
            raw_bbox = raw_bbox[0]
        current_bbox = raw_bbox

        prompt_refine = (
            f"Tighten bbox {current_bbox} for evaluator label '{target_object}'. "
            f'Fix missing parts or excess margin. JSON: {{ "bbox": [...] }}'
        )
        res_refine = self._call_gemini(prompt_refine, encoded_string)
        if res_refine:
            raw_refined = res_refine.get("bbox") or current_bbox
            if isinstance(raw_refined[0], list):
                raw_refined = raw_refined[0]
            current_bbox = raw_refined
        print(f"  -> Bbox: {current_bbox}", flush=True)
        return current_bbox

    def identify_scene_objects(self, image_path, max_objects=4):
        """Return a list of distinct object names for multi-round correction."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        max_objects = max(2, min(4, int(max_objects)))
        prompt = f"""
        Analyze this image and list 2 to {int(max_objects)} distinct, countable, segmentable OBJECTS.

        STRICT RULES:
        - Include only tangible objects with clear boundaries (e.g., mouse, wrench, surfboard, television).
        - DO NOT include background/stuff/scene regions, materials, terrain, or surfaces:
          examples: ground, concrete ground, floor, wall, sky, water, road, grass, sand, pavement, background.
        - DO NOT output scene descriptors or broad categories.
        - Prefer independent objects that can be resized individually.
        - Do NOT invent objects. If only 3 clear objects exist, return 3.

        Output JSON only:
        {{
          "objects": ["obj1", "obj2", ...]
        }}
        """
        res = self._call_gemini(prompt, encoded_string)
        if not res:
            return []
        objs = res.get("objects", [])
        if not isinstance(objs, list):
            return []
        banned_terms = {
            "background", "scene", "environment", "context",
            "ground", "concrete ground", "floor", "wall", "ceiling",
            "road", "street", "pavement", "sidewalk",
            "grass", "sand", "soil", "dirt", "terrain",
            "sky", "cloud", "water", "sea", "river", "ocean",
            "mountain", "hill", "forest",
        }
        cleaned = []
        for o in objs:
            name = str(o).strip()
            if not name:
                continue
            name_l = name.lower()
            # Hard filter: remove background/"stuff" terms even if Gemini returns them.
            if name_l in banned_terms:
                continue
            if any(
                t in name_l
                for t in (
                    "background", "ground", "floor", "wall", "road", "sky",
                    "terrain", "pavement", "sidewalk", "grass", "sand", "water",
                )
            ):
                continue
            if name.lower() in {x.lower() for x in cleaned}:
                continue
            cleaned.append(name)
        return cleaned[:max_objects]

    def screen_multi_object_scene_for_size_edit(self, image_path: str, primary_object_names: list):
        """
        One Gemini VLM call before multi-round edits. Flags scenes where size correction is
        unreliable — aligned with cues aggregated in scripts/eval/task1/evaluate_genscale_task1_gemini.py
        (duplicate_objects, extra_unnamed_objects), plus severe generation artifacts / unclear objects.

        Returns None on API/parse failure (caller should proceed, not skip).
        """
        names = [str(x).strip() for x in (primary_object_names or []) if str(x).strip()]
        names_csv = ", ".join(names[:10]) if names else "(unspecified)"
        ext = os.path.splitext(str(image_path or ""))[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        try:
            with open(image_path, "rb") as f:
                enc = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

        prompt = f"""You audit a synthetic image BEFORE an automated object size-correction pipeline.

Prominent labels from our detector (reference only): {names_csv}.

The list may include evaluator synonyms for the same physical object (e.g. short vs parenthesized names).
Treat those as ONE intended label set — do not count them as separate extra clutter.

From the image alone, output ONE JSON object with:
- "duplicate_objects" (bool): two+ clearly separate instances of the SAME category so it is ambiguous which to edit (e.g. two identical eggs, two rulers).
- "extra_unnamed_objects" (bool): major extra props/clutter/repeated shapes beyond the intended label set so relative-scale reasoning is unreliable (not mere synonyms in the list above).
- "severe_generation_artifacts" (bool): obvious AI flaws (fused objects, melted geometry, incoherent boundaries, extra limbs) that would break inpainting.
- "objects_individually_clear" (bool): each listed label could be matched to a distinct instance with usable boundaries; false if blur/heavy overlap/crop blocks that.

Set "skip_size_correction" (bool) true if ANY of: duplicate_objects, extra_unnamed_objects, severe_generation_artifacts, OR objects_individually_clear is false.
Add "brief_reason" (string, <= 35 words, English).

JSON only, no markdown."""

        res = self._call_gemini(prompt, enc, mime_type=mime)
        if not res or not isinstance(res, dict):
            return None

        def _as_bool(v, default: bool = False) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes", "y")
            return default

        dup = _as_bool(res.get("duplicate_objects"), False)
        extra = _as_bool(res.get("extra_unnamed_objects"), False)
        art = _as_bool(res.get("severe_generation_artifacts"), False)
        clear = _as_bool(res.get("objects_individually_clear"), True)
        derived_skip = bool(dup or extra or art or (not clear))
        skip = derived_skip or _as_bool(res.get("skip_size_correction"), False)

        return {
            "duplicate_objects": dup,
            "extra_unnamed_objects": extra,
            "severe_generation_artifacts": art,
            "objects_individually_clear": clear,
            "skip_size_correction": skip,
            "brief_reason": str(res.get("brief_reason", "") or res.get("notes", "") or "")[:500],
        }


class SAM2Segmentor:
    def __init__(self, device_id=0):
        self.device = f"cuda:{device_id}"
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        checkpoint = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2_hiera_large.pt")
        sam2_model = build_sam2("sam2_hiera_l", checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def segment_with_box(self, image, bbox_xyxy):
        """Segment object using a bounding box prompt (robust to curved objects).
        bbox_xyxy: [x1, y1, x2, y2] in pixel coords on the full image.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox_xyxy),
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)

        x1, y1, x2, y2 = bbox_xyxy
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        mask_in_box = best_mask[y1:y2, x1:x2]
        coverage = np.sum(mask_in_box) / bbox_area

        if 0.05 < coverage < 0.98:
            return best_mask

        print("    Box prompt coverage poor, trying multi-point fallback ...")
        return self._fallback_multipoint(image, bbox_xyxy)

    def _fallback_multipoint(self, image, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        points = []
        for fy in [0.25, 0.5, 0.75]:
            for fx in [0.25, 0.5, 0.75]:
                points.append([x1 + int(w * fx), y1 + int(h * fy)])
        points = np.array(points)
        labels = np.ones(len(points), dtype=np.int32)

        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)].astype(np.uint8)
        mask_area = np.sum(best_mask)
        bbox_area = max(1, w * h)
        if mask_area / bbox_area < 0.05:
            best_mask = np.any(masks, axis=0).astype(np.uint8)
        return best_mask

    def segment_object(self, image, point_coords=None):
        """Legacy center-point interface (kept for backward compatibility)."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([point_coords]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        return masks[np.argmax(scores)].astype(np.uint8)

    def unload(self):
        del self.predictor
        gc.collect()
        torch.cuda.empty_cache()

    def to_device(self, device):
        self.device = str(device)
        self.predictor.model.to(self.device)
        return self

class InstructionEditingRemover:
    """
    Instruction-based remover (FLUX.1-Kontext only).
    """
    def __init__(
        self,
        device_id=1,
        kontext_path=os.environ.get("FLUX_KONTEXT_PATH", "black-forest-labs/FLUX.1-Kontext-dev"),
    ):
        self.device = f"cuda:{device_id}"
        self.pipeline = FluxKontextPipeline.from_pretrained(
            kontext_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.backend = "flux_kontext"

    def remove_object(
        self,
        image_pil,
        target_object,
        num_steps=28,
        guidance_scale=4.0,
        seed=42,
        prompt_override=None,
    ):
        prompt = prompt_override or (
            f"Remove the {target_object} completely from this image. "
            f"Also remove physical effects caused by the {target_object} "
            "(reflection, mirror highlight, cast shadow, contact shadow, glow). "
            "Fill missing regions with a natural photorealistic background. "
            "Do not modify any other object in the image. "
            "Preserve all unrelated objects, geometry, perspective, lighting, and texture."
        )

        generator = torch.Generator(self.device).manual_seed(seed)

        call_kwargs = dict(
            image=image_pil,
            prompt=prompt,
            num_inference_steps=num_steps,
            generator=generator,
        )
        call_kwargs["guidance_scale"] = guidance_scale
        call_kwargs["negative_prompt"] = (
            "artifacts, blurry, distorted geometry, duplicated object, unnatural texture"
        )

        try:
            out = self.pipeline(**call_kwargs).images[0]
        except TypeError:
            # Some custom pipelines do not support all kwargs
            reduced_kwargs = dict(
                image=image_pil,
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=generator,
            )
            out = self.pipeline(**reduced_kwargs).images[0]

        if out.mode != "RGB":
            out = out.convert("RGB")
        if out.size != image_pil.size:
            print(f"⚠️ Editing output size changed {out.size} -> {image_pil.size}, resizing back.")
            out = out.resize(image_pil.size, Image.LANCZOS)
        return out

    def unload(self):
        del self.pipeline
        gc.collect()
        torch.cuda.empty_cache()

    def to_device(self, device):
        self.device = str(device)
        self.pipeline.to(self.device)
        return self

class DepthAnythingV2Estimator:
    def __init__(self, device_id=1):
        self.device = f"cuda:{device_id}"
        from depth_anything_v2.dpt import DepthAnythingV2
        self.model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        self.model.load_state_dict(
            torch.load(os.environ.get("DEPTH_ANYTHING_V2_CHECKPOINT", "checkpoints/depth_anything_v2_vitl.pth"),
                       map_location="cpu")
        )
        self.model = self.model.to(self.device).eval()

    def estimate_depth(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        with torch.no_grad():
            depth = self.model.infer_image(image)
        depth_norm = np.zeros(depth.shape, dtype=np.float32)
        cv2.normalize(depth, depth_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return depth_norm.astype(np.uint8)

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def to_device(self, device):
        self.device = str(device)
        self.model = self.model.to(self.device).eval()
        return self

class InsertAnythingInferencer:
    def __init__(
        self,
        weights_dir,
        flux_fill_path,
        flux_redux_path,
        device="cuda",
        dtype=torch.bfloat16,
        use_hf_inject=None,
        hf_hp_radius: float = 0.15,
    ):
        self.device = device
        self.dtype = dtype
        self.hf_hp_radius = float(hf_hp_radius)
        self.hf_latent_inject = None
        print(f"🚀 Loading InsertAnything Models on {device}...")
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(
            flux_fill_path, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        _orig_preprocess = self.flux_fill_pipe.image_processor.preprocess

        def _safe_preprocess(images, *args, **kwargs):
            if torch.is_tensor(images):
                # If tensor looks like [-1,1], map to [0,1]
                if images.min().item() < 0:
                    images = (images + 1.0) * 0.5
                images = images.clamp(0.0, 1.0)
            return _orig_preprocess(images, *args, **kwargs)

        self.flux_fill_pipe.image_processor.preprocess = _safe_preprocess
        self.flux_redux = FluxPriorReduxPipeline.from_pretrained(
            flux_redux_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self.controlnet = FluxControlNetModel.from_pretrained(
            os.path.join(weights_dir, "controlnet"), torch_dtype=dtype, low_cpu_mem_usage=True
        )

        self.transformer = self.flux_fill_pipe.transformer
        lora_config = LoraConfig(
            r=256, lora_alpha=256, init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"]
        )
        self.transformer.add_adapter(lora_config)

        state_dict = load_file(os.path.join(weights_dir, "pytorch_lora_weights.safetensors"))
        converted = {k.replace("transformer.", "base_model.model."): v for k, v in state_dict.items()}
        set_peft_model_state_dict(self.transformer, converted)
        print("✅ Models Loaded.")

        # Optional HF latent inject (same as training: VAE(packed) @ hf_diptych -> Linear -> x_embedder residual)
        _hf_ckpt = os.path.join(weights_dir, HF_INJECT_CKPT_NAME)
        if use_hf_inject is None:
            use_hf_inject = os.path.isfile(_hf_ckpt)
        if use_hf_inject:
            odim = int(self.transformer.x_embedder.out_features)
            self.hf_latent_inject = nn.Linear(HF_LATENT_DIM, odim, bias=False).to(device).to(dtype)
            nn.init.zeros_(self.hf_latent_inject.weight)
            if os.path.isfile(_hf_ckpt):
                self.hf_latent_inject.load_state_dict(torch.load(_hf_ckpt, map_location=device))
                print(f"✅ Loaded {HF_INJECT_CKPT_NAME} (HF inject)", flush=True)
            else:
                print(
                    f"⚠️ HF inject requested but missing {_hf_ckpt} — skipping HF branch",
                    flush=True,
                )
                self.hf_latent_inject = None
        else:
            print("ℹ️ HF inject off (use --use_hf_inject 1 or place hf_latent_inject.pt in weights_dir).", flush=True)

    def to_device(self, device):
        self.device = str(device)
        self.flux_fill_pipe.to(self.device)
        self.transformer.to(self.device)
        self.controlnet.to(self.device)
        if self.hf_latent_inject is not None:
            self.hf_latent_inject.to(self.device)
        return self

    @torch.no_grad()
    def generate(
        self,
        ref_image,
        background,
        target_mask,
        depth_map,
        num_steps=30,
        guidance_scale=30.0,
        controlnet_scale=0.5,
        seed=42,
        controlnet_end=0.6,
        use_depth_control=True,
        hf_hp_radius=None,
    ):
        """
        IMPORTANT: Align to training diptych format:
          image     = [ref | target]
          mask      = [0   | target_mask]
          depthcond = [0   | depth]
        Return ONLY right half (target) like original InsertAnything inference.
        """
        generator = torch.Generator(self.device).manual_seed(seed)

        def _prepare_ref_to_canvas(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
            """
            Preserve reference aspect ratio:
            1) tight-crop foreground on white bg
            2) pad to square
            3) resize to target canvas size
            """
            arr = np.array(img.convert("RGB"))
            fg = np.any(arr < 250, axis=2)  # ref_clean uses white background
            if np.any(fg):
                ys, xs = np.where(fg)
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                crop = img.crop((x1, y1, x2, y2))
            else:
                crop = img

            cw, ch = crop.size
            side = max(cw, ch)
            square = Image.new("RGB", (side, side), (255, 255, 255))
            ox = (side - cw) // 2
            oy = (side - ch) // 2
            square.paste(crop, (ox, oy))
            return square.resize((out_w, out_h), Image.LANCZOS)

        # ------------------------------
        # build diptych inputs
        # ------------------------------
        W, H = background.size
        ref_resized = _prepare_ref_to_canvas(ref_image, W, H)
        bg_resized = background.resize((W, H), Image.LANCZOS)

        # image diptych: [ref | target]
        diptych = Image.new("RGB", (2 * W, H), (255, 255, 255))
        diptych.paste(ref_resized, (0, 0))
        diptych.paste(bg_resized, (W, 0))

        # mask diptych: [0 | target_mask]  (1-channel)
        m = target_mask.convert("L").resize((W, H), Image.NEAREST)
        mask_dip = Image.new("L", (2 * W, H), 0)
        mask_dip.paste(m, (W, 0))

        # depth diptych: [0 | depth]
        d = depth_map.convert("RGB").resize((W, H), Image.NEAREST)
        depth_dip = Image.new("RGB", (2 * W, H), (0, 0, 0))
        depth_dip.paste(d, (W, 0))

        width, height = diptych.size  # (2W, H)

        # ============================================================
        # [FIX - MUST] match training/callback preprocessing exactly
        #   - image/depth: image_processor.preprocess -> [-1, 1]
        #   - mask: mask_processor.preprocess -> [0, 1], 1-channel
        # ============================================================
        bg_tensor = self.flux_fill_pipe.image_processor.preprocess(diptych)
        bg_tensor = bg_tensor.to(self.device).to(self.dtype)  # (1,3,H,2W)

        # depth_tensor = self.flux_fill_pipe.image_processor.preprocess(depth_dip)
        # depth_tensor = depth_tensor.to(self.device).to(self.dtype)  # (1,3,H,2W)

        mask_tensor = self.flux_fill_pipe.mask_processor.preprocess(
            mask_dip, height=height, width=width
        )
        mask_tensor = mask_tensor.to(self.device).to(self.dtype)  # (1,1,H,2W)

        # Reference embeddings: move redux to GPU, encode, then offload back to CPU
        self.flux_redux.to(self.device)
        prompt_embeds, pooled_prompt_embeds = image_output(self.flux_redux, ref_resized, self.device)
        self.flux_redux.to("cpu")
        torch.cuda.empty_cache()
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_fill_pipe,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=512
        )

        # controlnet depth latents (your helper already uses image_processor.preprocess)
        if use_depth_control:
            depth_latents, depth_ids = encode_depth_for_controlnet(self.flux_fill_pipe, depth_dip)
            self.controlnet.to(self.device)
        else:
            depth_latents, depth_ids = None, None

        # HF diptych -> VAE pack -> hf_latent_inject (matches training ``step()`` when use_hf_inject)
        r_hf = self.hf_hp_radius if hf_hp_radius is None else float(hf_hp_radius)
        image_cond_residual = None
        if self.hf_latent_inject is not None:
            ref_np = np.array(ref_resized.convert("RGB"))
            hf_dip_pil = build_hf_diptych_pil_from_ref(ref_np, W, H, hf_hp_radius=r_hf)
            hf_latents, _ = encode_depth_for_controlnet(self.flux_fill_pipe, hf_dip_pil)
            image_cond_residual = self.hf_latent_inject(hf_latents.to(dtype=self.dtype))

        # Masked BG encode (IMPORTANT: keep mask 1-channel but broadcast to 3-channel for multiply)
        mask_binary = (mask_tensor > 0.5).float()              # (1,1,H,2W)
        mask_binary_3 = mask_binary.repeat(1, 3, 1, 1)         # (1,3,H,2W)

        masked_bg = bg_tensor * (1 - mask_binary_3)
        src_latents, mask_latents = Flux_fill_encode_masks_images(self.flux_fill_pipe, masked_bg, mask_binary_3)
        condition_latents = torch.cat((src_latents, mask_latents), dim=-1)

        latent_h, latent_w = height // 8, width // 8
        latents = torch.randn((1, 16, latent_h, latent_w), device=self.device, dtype=self.dtype, generator=generator)
        latents = self.flux_fill_pipe._pack_latents(latents, 1, 16, latent_h, latent_w)
        img_ids = self.flux_fill_pipe._prepare_latent_image_ids(
            1, latent_h // 2, latent_w // 2, self.device, self.dtype
        )

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.flux_fill_pipe.scheduler.config)
        scheduler.set_timesteps(num_steps, device=self.device, mu=height * width / (1024 * 1024))

        for i, t in enumerate(scheduler.timesteps):
            t_vec = torch.tensor([t / 1000], device=self.device, dtype=self.dtype)
            guidance_vec = torch.tensor([guidance_scale], device=self.device, dtype=self.dtype)
            current_scale = controlnet_scale if i < int(num_steps * controlnet_end) else 0.0

            if use_depth_control and current_scale > 0:
                block_samples, single_block_samples = self.controlnet(
                    hidden_states=latents,
                    controlnet_cond=depth_latents,
                    conditioning_scale=current_scale,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=t_vec,
                    img_ids=img_ids,
                    txt_ids=text_ids,
                    guidance=guidance_vec,
                    return_dict=False
                )
                block_samples, single_block_samples = mask_controlnet_residuals_for_diptych(
                    block_samples, single_block_samples, img_ids, self.dtype,
                )
            else:
                block_samples, single_block_samples = None, None


            model_input = torch.cat((latents, condition_latents), dim=2)
            noise_pred = tranformer_forward(
                self.transformer,
                model_config=self.transformer.config,
                hidden_states=model_input,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t_vec,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=guidance_vec,
                controlnet_block_samples=block_samples,
                controlnet_single_block_samples=single_block_samples,
                image_cond_residual=image_cond_residual,
                return_dict=False
            )[0]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if use_depth_control:
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        decoded = self.flux_fill_pipe._unpack_latents(latents, height, width, self.flux_fill_pipe.vae_scale_factor)
        decoded = (decoded / self.flux_fill_pipe.vae.config.scaling_factor) + self.flux_fill_pipe.vae.config.shift_factor
        image_tensor = self.flux_fill_pipe.vae.decode(decoded, return_dict=False)[0]
        out = self.flux_fill_pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

        # crop right half only
        out = out.crop((W, 0, 2 * W, H))
        return out


# ==============================================================================
# 5. Main Workflow — FIXED (Scheme B)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str,
                        default="")
    parser.add_argument("--target_object", type=str, default=None)
    parser.add_argument("--ref_object", type=str, default=None)
    parser.add_argument(
        "--feedback_json_path",
        type=str,
        default="",
        help="Optional path to gemini_scores_*.json to inject evaluation feedback into planner (direction constraint).",
    )
    parser.add_argument(
        "--use_feedback_planner",
        type=int,
        default=1,
        help="1: use feedback_json_path to constrain shrink/enlarge; 0: disable (default=1)",
    )
    parser.add_argument(
        "--feedback_lookup_task_id",
        type=str,
        default="",
        help="Optional stem for feedback JSON lookup (e.g. T1_0010). Use when image_path is not "
        "the original file (e.g. multi-round intermediate final_result.png). Default: basename of image_path.",
    )
    parser.add_argument(
        "--feedback_pair_index",
        type=int,
        default=0,
        help="Which pair in feedback JSON to use for canonical object names (default 0).",
    )
    parser.add_argument("--multi_round", type=int, default=1,
                        help="1: auto multi-round correction (largest object as anchor when target/ref are not set)")
    parser.add_argument("--max_round_objects", type=int, default=4,
                        help="Number of scene objects for multi-round correction (clamped to 2-4)")
    parser.add_argument("--anchor_object", type=str, default=None,
                        help="Optional fixed anchor object name for multi-round mode")
    parser.add_argument(
        "--multi_round_scene_prefilter",
        type=int,
        default=1,
        help="1: before multi-round edits, call Gemini to skip ambiguous scenes (duplicate_objects / "
        "extra_unnamed_objects / severe artifacts / unclear objects), aligned with task1 eval cues. "
        "Still writes final_multi_object_result.png, layout_*.json, multi_object_history.json. 0=off.",
    )
    parser.add_argument("--_in_round", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=str, default="results/size_correction_T1")
    parser.add_argument("--weights_dir", type=str, default="./weights/499000")
    parser.add_argument(
        "--use_hf_inject",
        type=int,
        default=None,
        help="HF latent inject (VAE ref HF | zeros + hf_latent_inject.pt): "
        "None=auto if weights_dir/hf_latent_inject.pt exists, 1=on, 0=off",
    )
    parser.add_argument(
        "--hf_hp_radius",
        type=float,
        default=0.15,
        help="HiFi high-pass radius for ref HF map (match training src/data/base.py)",
    )
    parser.add_argument("--gpu_gen", type=int, default=1)
    parser.add_argument("--gpu_tools", type=int, default=1)
    parser.add_argument(
        "--min_scale",
        type=float,
        default=_SCALE_HARD_MIN,
        help="Min allowed scaling ratio (default: 1/3, matches gemini ratio tier envelope)",
    )
    parser.add_argument(
        "--max_scale",
        type=float,
        default=_SCALE_HARD_MAX,
        help="Max allowed scaling ratio (default: 3.0, matches gemini ratio tier envelope)",
    )
    parser.add_argument(
        "--min_scale_change",
        type=float,
        default=0.08,
        help="If abs(scale_factor-1) < this, skip editing and directly save original as final_result (default=0.08, 0=disable).",
    )
    parser.add_argument("--debug_draw_bbox", action="store_true", help="Save bbox overlay debug images")

    # crop ratio
    parser.add_argument("--crop_ratio", type=float, default=2.5,
                        help="Zoom-in crop ratio (bigger keeps more context; default=4)")

    # mask dilation controls
    parser.add_argument("--mask_dilate_kernel", type=int, default=25,
                        help="Dilation kernel size (square). default=25")
    parser.add_argument("--mask_dilate_iter", type=int, default=3,
                        help="Dilation iterations. default=3")
    parser.add_argument("--mask_dilate_boost", type=float, default=1.35,
                        help="Extra dilation strength multiplier for generation mask (default=1.35)")
    parser.add_argument("--mask_dilate_iter_boost", type=float, default=1.25,
                        help="Extra dilation iteration multiplier for generation mask (default=1.25)")
    parser.add_argument("--mask_dilate_depth_aware", type=int, default=1,
                        help="1: shape-preserving dilation via distance transform (keeps mask consistent "
                             "with depth shape); 0: legacy uniform square-kernel dilation (default=1)")
    parser.add_argument("--mask_dilate_ratio", type=float, default=0.30,
                        help="Border expansion as fraction of object min dimension when depth-aware "
                             "dilation is enabled (default=0.30)")
    parser.add_argument(
        "--mask_inpaint_regularize",
        type=str,
        default="close_convex_hull",
        choices=[
            "none",
            "close",
            "convex_hull",
            "close_convex_hull",
            "fill_holes",
            "fill_holes_close",
        ],
        help="After dilation, regularize inpaint mask (fill occlusion notches). "
        "Default close_convex_hull (aligned with inference_size_correction_t2.py). "
        "Use none for raw SAM/fused shape; avoid strong convex_hull on long thin anchors.",
    )
    parser.add_argument(
        "--mask_regularize_close_ksz",
        type=int,
        default=15,
        help="Ellipse kernel size for mask MORPH_CLOSE when regularize uses close (odd; default 15, T2-style).",
    )
    parser.add_argument(
        "--mask_regularize_close_iter",
        type=int,
        default=2,
        help="Iterations for mask MORPH_CLOSE when regularize uses close (default 2).",
    )
    parser.add_argument(
        "--mask_crop_fill_holes",
        type=int,
        default=1,
        help="After resizing mask to generation resolution, flood-fill enclosed holes (donut SAM / NEAREST gaps). "
        "1=on (default), 0=off.",
    )
    parser.add_argument(
        "--mask_crop_post_dilate",
        type=int,
        default=1,
        help="After fill_holes on crop mask, dilate with 3x3 ellipse kernel this many iterations "
        "(closes hairline gaps; default 1, T2-style).",
    )
    parser.add_argument("--ref_upscale_threshold", type=int, default=128,
                        help="If ref_clean max dimension < this, upscale with Gemini (default=128, 0=disable)")
    parser.add_argument("--ref_upscale_target", type=int, default=338,
                        help="Target longest side for Gemini upscale (default=338)")
    parser.add_argument("--blend_feather_border_px", type=int, default=16,
                        help="Erode blend mask inward by N px; only the eroded border ring is feathered (default=16)")
    parser.add_argument("--blend_alpha_blur_sigma", type=float, default=5.0,
                        help="Gaussian blur sigma for alpha smoothing (default=5.0)")
    parser.add_argument("--disable_blend_feather", type=int, default=0,
                        help="1: disable feathering and use hard alpha paste-back; 0: enable feathering")
    parser.add_argument("--bg_removal_mode", type=str, default="gemini", choices=["flux", "gemini", "editing"],
                        help="Background removal mode: flux/editing (open-weights) or gemini (API image-edit).")
    parser.add_argument("--editing_steps", type=int, default=28,
                        help="Editing inference steps")
    parser.add_argument("--editing_guidance_scale", type=float, default=4.0,
                        help="Guidance scale for editing model")
    parser.add_argument("--bg_remove_max_retries", type=int, default=2,
                        help="Max retries for editing-based removal before Gemini cleanup fallback")
    parser.add_argument("--bg_remove_diff_thresh", type=float, default=0.065,
                        help="Mean absolute difference threshold in target mask for successful removal")
    parser.add_argument("--bg_remove_changed_ratio_thresh", type=float, default=0.20,
                        help="Changed-pixel ratio threshold in target mask for successful removal")
    parser.add_argument("--bg_remove_pixel_diff_cutoff", type=float, default=0.06,
                        help="Per-pixel diff cutoff used for changed_ratio computation inside target mask")
    parser.add_argument("--bg_preserve_diff_mean_max", type=float, default=0.05,
                        help="Max mean RGB diff allowed outside target mask (lower is stricter)")
    parser.add_argument("--bg_preserve_changed_ratio_max", type=float, default=0.2,
                        help="Max changed-pixel ratio allowed outside target mask")
    parser.add_argument("--enforce_bg_preserve", type=int, default=0,
                        help="1: enforce non-target preservation gate; 0: ignore preserve gate")
    parser.add_argument("--preserve_object_names", type=str, default="",
                        help="Comma-separated non-target object labels that must remain after object removal.")
    parser.add_argument("--bg_overremove_check", type=int, default=0,
                        help="1: use Gemini vision JSON check to reject removal candidates that damage preserved objects.")
    parser.add_argument("--bg_overremove_model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model for cheap over-removal QA (default flash).")
    parser.add_argument("--editing_model_id", type=str, default="",
                        help="Compatibility arg for external scripts; local editing remover uses configured FLUX/Kontext path.")
    parser.add_argument("--prefer_gemini_cleanup", type=int, default=0,
                        help="1: try Gemini image cleanup before Flux editing remover")
    parser.add_argument("--gemini_cleanup_only", type=int, default=0,
                        help="1: only use Gemini image cleanup for removal and skip Flux remover")
    parser.add_argument("--use_gemini_cleanup_fallback", type=int, default=1,
                        help="1: if editing cleanup fails quality check, try Gemini cleanup fallback")
    parser.add_argument("--use_seg_bbox_for_transform", type=int, default=1,
                        help="1: use SAM2 mask tight bbox for final transform; 0: keep Gemini bbox")
    parser.add_argument("--num_steps", type=int, default=30,
                        help="InsertAnything denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=30.0,
                        help="InsertAnything guidance scale")
    parser.add_argument("--controlnet_scale", type=float, default=0.7,
                        help="Depth ControlNet scale when depth control is enabled")
    parser.add_argument("--controlnet_end", type=float, default=0.6,
                        help="Depth ControlNet end ratio when depth control is enabled")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    parser.add_argument("--presence_diff_thresh", type=float, default=0.03,
                        help="Min mean RGB diff inside target mask to consider object as generated (lower=harder to trigger nodepth)")
    parser.add_argument("--nodepth_prefer_ratio", type=float, default=2.0,
                        help="Prefer nodepth only when its in-mask change is this many times depth result (higher=more conservative)")
    parser.add_argument("--enable_nodepth_compare", type=int, default=0,
                        help="1: also generate nodepth and auto-choose; 0: depth-only (default, faster)")
    parser.add_argument(
        "--force_no_depth_control",
        type=int,
        default=0,
        help="1: force disable depth ControlNet during generation (mask-only), even if depth fusion produced use_depth=True.",
    )
    parser.add_argument("--bottom_center_depth_bias", type=float, default=50.0,
                        help="Max depth bias; actual bias is adaptive based on fg/bg depth separation")
    parser.add_argument("--depth_coverage_nodepth_gate", type=float, default=0.60,
                        help="If depth fusion coverage is below this, prefer nodepth when it is not worse")
    parser.add_argument("--override_scale_factor", type=float, default=None,
                        help="If set, use this scale factor directly and skip Gemini scale planning")
    parser.add_argument("--override_anchor_point", type=str, default=None,
                        choices=["BOTTOM_CENTER", "TOP_CENTER", "CENTER", "LEFT_CENTER", "RIGHT_CENTER"],
                        help="If set with override_scale_factor, use this anchor and skip Gemini scale planning")
    parser.add_argument("--cache_models_cpu", type=int, default=1,
                        help="1: keep models in process-level cache on CPU and move to GPU on demand (default=1)")
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default="",
        help="If set to an existing file, use it as ref_clean (letterboxed to match scene SAM ref size) "
             "instead of the scene crop; SAM mask/bg removal still use the scene. Empty = use scene ref only.",
    )

    # P0: post-correction Gemini re-score + rollback.
    parser.add_argument("--enable_rescore", type=int, default=1,
                        help="1: run Gemini re-score on corrected image and rollback if score worsens (default=1)")
    parser.add_argument("--rescore_object_a", type=str, default="",
                        help="Anchor object name for re-score prompt (Object A)")
    parser.add_argument("--rescore_object_b", type=str, default="",
                        help="Target object name for re-score prompt (Object B)")
    parser.add_argument("--rescore_len_a_cm", type=float, default=0.0,
                        help="Anchor object typical physical length in cm (for re-score prompt)")
    parser.add_argument("--rescore_len_b_cm", type=float, default=0.0,
                        help="Target object typical physical length in cm (for re-score prompt)")
    parser.add_argument("--rescore_scenario", type=str, default="",
                        help="Scenario tag for re-score prompt (e.g. S2_Extreme_Contrast)")
    parser.add_argument("--google_api_key", type=str, default="",
                        help="Google API key for Gemini re-score (falls back to GOOGLE_API_KEY env var)")
    parser.add_argument("--gemini_model", type=str, default="gemini-3.1-pro-preview",
                        help="Gemini model version for re-score (default: gemini-3.1-pro-preview)")

    args = parser.parse_args()

    # Model Definitions
    FLUX_FILL = os.environ.get("FLUX_FILL_PATH", "black-forest-labs/FLUX.1-Fill-dev")
    FLUX_REDUX = os.environ.get("FLUX_REDUX_PATH", "black-forest-labs/FLUX.1-Redux-dev")
    FLUX_KONTEXT = os.environ.get("FLUX_KONTEXT_PATH", "black-forest-labs/FLUX.1-Kontext-dev")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires at least one GPU.")

    num_devices = torch.cuda.device_count()
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"🖥️  CUDA_VISIBLE_DEVICES={visible_env}, torch sees {num_devices} device(s).")

    # Scheduler environments (SCC/Slurm/SGE) usually expose assigned GPU(s) via
    # CUDA_VISIBLE_DEVICES. When only one visible GPU exists, always use cuda:0.
    if num_devices == 1:
        args.gpu_gen = 0
        args.gpu_tools = 0
    else:
        if args.gpu_gen >= num_devices or args.gpu_gen < 0:
            args.gpu_gen = 0
        if args.gpu_tools >= num_devices or args.gpu_tools < 0:
            args.gpu_tools = 0

    # Early GPU context probe: fail fast with actionable message on SCC
    try:
        _ = torch.zeros((1,), device=f"cuda:{args.gpu_tools}")
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(
            "Failed to create CUDA context on assigned GPU. "
            "This is usually a scheduler/runtime issue (GPU busy/unavailable), not model code. "
            "Please check `nvidia-smi` in the same qrsh shell, then release and request a new GPU job if needed. "
            f"Original error: {e}"
        )

    analyzer = GeminiBBoxAnalyzer(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model_version="gemini-3.1-pro-preview",
        feedback_json_path=args.feedback_json_path,
        use_feedback_planner=int(args.use_feedback_planner),
        feedback_lookup_task_id=str(getattr(args, "feedback_lookup_task_id", "") or ""),
    )

    # Optional multi-round controller:
    # keep the largest object as anchor, and correct other objects one-by-one.
    multi_round_attempted = False
    multi_round_root = None
    multi_round_abort_reason = None
    if (
        int(args.multi_round) == 1
        and int(args._in_round) == 0
        and (not args.target_object and not args.ref_object)
    ):
        multi_round_attempted = True
        print("\n🔁 Multi-round mode enabled: detecting scene objects...", flush=True)
        max_scene_objs = max(2, min(4, int(args.max_round_objects)))
        _feedback_task_id = str(getattr(args, "feedback_lookup_task_id", "") or "").strip() or os.path.splitext(
            os.path.basename(args.image_path)
        )[0]
        _feedback_entry = _load_feedback_entry_for_task(args.feedback_json_path, _feedback_task_id)
        scene_objects = _feedback_scene_object_names(_feedback_entry)
        if scene_objects:
            print(
                f"   [feedback] Using evaluator object labels from score JSON: {scene_objects}",
                flush=True,
            )
        else:
            scene_objects = analyzer.identify_scene_objects(
                args.image_path, max_objects=max_scene_objs
            )
        if args.anchor_object and args.anchor_object not in scene_objects:
            scene_objects = [args.anchor_object] + scene_objects

        dedup = []
        seen = set()
        for name in scene_objects:
            n = str(name).strip()
            if not n:
                continue
            k = n.lower()
            if k in seen:
                continue
            seen.add(k)
            dedup.append(n)
        scene_objects = dedup[:max_scene_objs]

        if len(scene_objects) >= 2:
            probe_img = Image.open(args.image_path).convert("RGB")
            probe_w, probe_h = probe_img.size
            del probe_img

            detected = []
            for name in scene_objects:
                raw_box = analyzer.detect_target_bbox(args.image_path, name)
                if raw_box is None:
                    continue
                box_px = bbox_raw_to_pixels(raw_box, probe_w, probe_h)
                if box_px is None:
                    continue
                y1b, x1b, y2b, x2b = box_px
                area = max(0, y2b - y1b) * max(0, x2b - x1b)
                if area < 100:
                    continue
                detected.append({"name": name, "area": float(area), "bbox": box_px})

            if len(detected) >= 2:
                # Remove tiny detections that are likely sub-parts / false positives.
                # Keep at least two objects by falling back to top-area objects.
                largest_area = max(d["area"] for d in detected)
                min_keep_area = max(100.0, largest_area * 0.005)  # 0.5% of largest object
                filtered = [d for d in detected if d["area"] >= min_keep_area]
                if len(filtered) >= 2:
                    detected = filtered
                else:
                    detected = sorted(detected, key=lambda z: z["area"], reverse=True)[:2]

                # Suppress heavily overlapping "sub-part" objects (e.g. phone stand within smartphone).
                # Keep larger detection when overlap is very high.
                dedup_kept = []
                for cand in sorted(detected, key=lambda z: z["area"], reverse=True):
                    overlap_with_kept = False
                    for kept in dedup_kept:
                        iom = bbox_intersection_over_min_area(cand["bbox"], kept["bbox"])
                        iou = bbox_iou_yxyx(cand["bbox"], kept["bbox"])
                        if iom >= 0.75 or iou >= 0.60:
                            overlap_with_kept = True
                            break
                    if not overlap_with_kept:
                        dedup_kept.append(cand)
                if len(dedup_kept) >= 2:
                    detected = dedup_kept

                _tid_fb = str(getattr(args, "feedback_lookup_task_id", "") or "").strip() or os.path.splitext(
                    os.path.basename(args.image_path)
                )[0]
                _apply_feedback_canonical_names_to_detected(
                    analyzer,
                    detected,
                    args.image_path,
                    probe_w,
                    probe_h,
                    args.feedback_json_path,
                    _tid_fb,
                    int(getattr(args, "feedback_pair_index", 0)),
                    int(args.use_feedback_planner),
                )

                # Anchor = largest detected object; user-forced anchor overrides.
                anchor_name = None
                if args.anchor_object:
                    for d in detected:
                        if d["name"].lower() == str(args.anchor_object).lower():
                            anchor_name = d["name"]
                            break
                if anchor_name is None:
                    anchor_name = max(detected, key=lambda z: z["area"])["name"]

                targets = sorted(
                    [d for d in detected if d["name"].lower() != anchor_name.lower()],
                    key=lambda z: z["area"],
                )

                if targets:
                    print(
                        f"   Anchor object: '{anchor_name}', rounds={len(targets)}",
                        flush=True,
                    )
                    img_stem = os.path.splitext(os.path.basename(args.image_path))[0]
                    target_names = "_".join(t["name"].replace(" ", "-") for t in targets)
                    anchor_short = anchor_name.replace(" ", "-")
                    session_name = f"{img_stem}__{target_names}_vs_{anchor_short}"
                    session_name = re.sub(r'[^\w\-]', '_', session_name)[:120]
                    candidate = os.path.join(args.output_dir, session_name)
                    if os.path.exists(candidate):
                        idx = 1
                        while os.path.exists(f"{candidate}_{idx}"):
                            idx += 1
                        candidate = f"{candidate}_{idx}"
                    multi_root = candidate
                    multi_round_root = multi_root
                    os.makedirs(multi_root, exist_ok=True)

                    # Live layout_final doc: clone of layout_original after grounding; each successful
                    # round patches the edited target's bbox from correction_meta (no end re-grounding).
                    layout_final_doc = None
                    creati_original_json_path = os.path.join(multi_root, "layout_original.json")
                    _layout_orig_payload = {
                        "image_path": str(args.image_path),
                        "image_size": [int(probe_w), int(probe_h)],
                        "anchor_object": anchor_name,
                        "objects": [
                            {"name": d["name"], "bbox_yxyx": [int(x) for x in d["bbox"]]}
                            for d in detected
                        ],
                    }
                    try:
                        with open(creati_original_json_path, "w", encoding="utf-8") as f_orig:
                            json.dump(_layout_orig_payload, f_orig, indent=2, ensure_ascii=False)
                        layout_final_doc = copy.deepcopy(_layout_orig_payload)
                        print(f"   [Layout] Saved original layout to: {creati_original_json_path}", flush=True)
                    except Exception as e_orig:
                        print(f"   [Layout] Warning: failed to save original layout: {e_orig}", flush=True)

                    current_image_path = args.image_path
                    multi_object_history = []
                    skip_corrections = False
                    scene_prefilter = None
                    if int(getattr(args, "multi_round_scene_prefilter", 1)) == 1:
                        scene_prefilter = _load_scene_prefilter_from_feedback_json(
                            args.feedback_json_path,
                            _feedback_task_id,
                            args.image_path,
                        )
                        if scene_prefilter is None:
                            scene_prefilter = _missing_scene_prefilter_record(
                                args.feedback_json_path,
                                _feedback_task_id,
                                args.image_path,
                            )
                            print(
                                "   [Scene prefilter] No scene_prefilter in score JSON; proceeding with edits.",
                                flush=True,
                            )
                        elif scene_prefilter.get("skip_size_correction"):
                            skip_corrections = True
                            _pf_dbg = {
                                k: scene_prefilter.get(k)
                                for k in (
                                    "duplicate_objects",
                                    "extra_unnamed_objects",
                                    "severe_generation_artifacts",
                                    "objects_individually_clear",
                                    "skip_size_correction",
                                )
                            }
                            print(
                                "   [Scene prefilter] score JSON says SKIP size correction "
                                "(ambiguous / flawed scene). "
                                f"flags={_pf_dbg} reason={scene_prefilter.get('brief_reason', '')!r}",
                                flush=True,
                            )
                    else:
                        scene_prefilter = {
                            "prefilter_enabled": False,
                            "skip_size_correction": False,
                            "prefilter_source": "disabled_by_arg",
                            "task_id": str(_feedback_task_id),
                            "prefilter_image_path": os.path.abspath(str(args.image_path)),
                        }

                    if not skip_corrections:
                        for i, item in enumerate(targets, 1):
                            tgt = item["name"]
                            round_out = os.path.join(multi_root, f"round_{i:02d}")
                            os.makedirs(round_out, exist_ok=True)
                            print(
                                f"\n--- Multi-round {i}/{len(targets)}: target='{tgt}' ref='{anchor_name}' ---",
                                flush=True,
                            )

                            round_args = [
                                "--image_path", current_image_path,
                                "--output_dir", round_out,
                                "--target_object", tgt,
                                "--ref_object", anchor_name,
                                "--multi_round", "0",
                                "--_in_round", "1",
                                "--feedback_json_path", str(args.feedback_json_path or ""),
                                "--use_feedback_planner", str(int(args.use_feedback_planner)),
                                "--min_scale_change", str(float(getattr(args, "min_scale_change", 0.0) or 0.0)),
                                "--feedback_lookup_task_id", str(img_stem),
                                "--feedback_pair_index", str(int(getattr(args, "feedback_pair_index", 0))),
                                "--weights_dir", str(args.weights_dir),
                                "--gpu_gen", str(int(args.gpu_gen)),
                                "--gpu_tools", str(int(args.gpu_tools)),
                                "--min_scale", str(float(args.min_scale)),
                                "--max_scale", str(float(args.max_scale)),
                                "--crop_ratio", str(float(args.crop_ratio)),
                                "--mask_dilate_kernel", str(int(args.mask_dilate_kernel)),
                                "--mask_dilate_iter", str(int(args.mask_dilate_iter)),
                                "--mask_dilate_boost", str(float(args.mask_dilate_boost)),
                                "--mask_dilate_iter_boost", str(float(args.mask_dilate_iter_boost)),
                                "--mask_dilate_depth_aware", str(int(args.mask_dilate_depth_aware)),
                                "--mask_dilate_ratio", str(float(args.mask_dilate_ratio)),
                                "--mask_inpaint_regularize", str(args.mask_inpaint_regularize),
                                "--mask_regularize_close_ksz", str(int(args.mask_regularize_close_ksz)),
                                "--mask_regularize_close_iter", str(int(args.mask_regularize_close_iter)),
                                "--mask_crop_fill_holes", str(int(args.mask_crop_fill_holes)),
                                "--mask_crop_post_dilate", str(int(args.mask_crop_post_dilate)),
                                "--ref_upscale_threshold", str(int(args.ref_upscale_threshold)),
                                "--ref_upscale_target", str(int(args.ref_upscale_target)),
                                "--blend_feather_border_px", str(int(args.blend_feather_border_px)),
                                "--blend_alpha_blur_sigma", str(float(args.blend_alpha_blur_sigma)),
                                "--disable_blend_feather", str(int(args.disable_blend_feather)),
                                "--bg_removal_mode", str(args.bg_removal_mode),
                                "--editing_steps", str(int(args.editing_steps)),
                                "--editing_guidance_scale", str(float(args.editing_guidance_scale)),
                                "--bg_remove_max_retries", str(int(args.bg_remove_max_retries)),
                                "--bg_remove_diff_thresh", str(float(args.bg_remove_diff_thresh)),
                                "--bg_remove_changed_ratio_thresh", str(float(args.bg_remove_changed_ratio_thresh)),
                                "--bg_remove_pixel_diff_cutoff", str(float(args.bg_remove_pixel_diff_cutoff)),
                                "--bg_preserve_diff_mean_max", str(float(args.bg_preserve_diff_mean_max)),
                                "--bg_preserve_changed_ratio_max", str(float(args.bg_preserve_changed_ratio_max)),
                                "--enforce_bg_preserve", str(int(args.enforce_bg_preserve)),
                                "--prefer_gemini_cleanup", str(int(args.prefer_gemini_cleanup)),
                                "--gemini_cleanup_only", str(int(args.gemini_cleanup_only)),
                                "--use_gemini_cleanup_fallback", str(int(args.use_gemini_cleanup_fallback)),
                                "--use_seg_bbox_for_transform", str(int(args.use_seg_bbox_for_transform)),
                                "--num_steps", str(int(args.num_steps)),
                                "--guidance_scale", str(float(args.guidance_scale)),
                                "--controlnet_scale", str(float(args.controlnet_scale)),
                                "--controlnet_end", str(float(args.controlnet_end)),
                                "--seed", str(int(args.seed)),
                                "--presence_diff_thresh", str(float(args.presence_diff_thresh)),
                                "--nodepth_prefer_ratio", str(float(args.nodepth_prefer_ratio)),
                                "--enable_nodepth_compare", str(int(args.enable_nodepth_compare)),
                                "--bottom_center_depth_bias", str(float(args.bottom_center_depth_bias)),
                                "--depth_coverage_nodepth_gate", str(float(args.depth_coverage_nodepth_gate)),
                                "--cache_models_cpu", str(int(args.cache_models_cpu)),
                            ]
                            if str(getattr(args, "reference_image_path", "") or "").strip():
                                round_args.extend([
                                    "--reference_image_path",
                                    str(args.reference_image_path).strip(),
                                ])
                            if args.override_scale_factor is not None and args.override_anchor_point is not None:
                                round_args.extend([
                                    "--override_scale_factor", str(float(args.override_scale_factor)),
                                    "--override_anchor_point", str(args.override_anchor_point),
                                ])
                            if args.debug_draw_bbox:
                                round_args.append("--debug_draw_bbox")

                            round_result_dir = run_with_args(round_args)
                            if not round_result_dir:
                                print("   Round failed, stop multi-round chain.", flush=True)
                                multi_round_abort_reason = f"round_{i:02d}_failed"
                                multi_object_history.append(
                                    {
                                        "round": i,
                                        "target": tgt,
                                        "anchor": anchor_name,
                                        "output": None,
                                        "plan": {"need_correction": True, "source": "inference_size_correction_multi_round", "round_folder": f"round_{i:02d}"},
                                        "scale_exec": None,
                                        "correction_meta_path": None,
                                        "correction_meta": None,
                                        "skipped": True,
                                        "skip_reason": "round_result_dir_none",
                                    }
                                )
                                break
                            round_final = os.path.join(round_result_dir, "final_result.png")
                            if not os.path.exists(round_final):
                                print("   Round output missing final_result.png, stop chain.", flush=True)
                                multi_round_abort_reason = f"round_{i:02d}_missing_final_result"
                                multi_object_history.append(
                                    {
                                        "round": i,
                                        "target": tgt,
                                        "anchor": anchor_name,
                                        "output": None,
                                        "plan": {"need_correction": True, "source": "inference_size_correction_multi_round", "round_folder": f"round_{i:02d}"},
                                        "scale_exec": None,
                                        "correction_meta_path": None,
                                        "correction_meta": None,
                                        "skipped": True,
                                        "skip_reason": "final_result_png_missing",
                                    }
                                )
                                break
                            cm_path = os.path.join(round_result_dir, "correction_meta.json")
                            correction_meta_round = None
                            if os.path.isfile(cm_path):
                                with open(cm_path, "r", encoding="utf-8") as _cmf:
                                    correction_meta_round = json.load(_cmf)
                            _fb_summary = None
                            if correction_meta_round and layout_final_doc is not None:
                                if _should_apply_bbox_patch_from_meta(correction_meta_round):
                                    _bb_meta = correction_meta_round.get("bbox_target_scaled_pixels_yxyx")
                                    if _bb_meta is not None and len(_bb_meta) == 4:
                                        if not _layout_update_object_bbox(
                                            layout_final_doc, tgt, _bb_meta
                                        ):
                                            print(
                                                f"   [Layout] warn: could not patch bbox for target '{tgt}' "
                                                f"(no matching name in layout_final).",
                                                flush=True,
                                            )
                            if correction_meta_round:
                                _fp = (correction_meta_round.get("feedback_planner") or {})
                                _blob = _fp.get("feedback_blob") or {}
                                _fb_summary = {
                                    "hint": _fp.get("hint", ""),
                                    "size_score": _blob.get("size_score"),
                                    "direction_hint": _blob.get("direction_hint", ""),
                                    "name_matched_pair": _blob.get("name_matched_pair"),
                                    "pair_match_inverted_ab": _blob.get("pair_match_inverted_ab"),
                                    "object_a": (_blob.get("object_a") or {}).get("name"),
                                    "object_b": (_blob.get("object_b") or {}).get("name"),
                                    "scale_factor_applied": correction_meta_round.get("scale_factor_effective"),
                                    "anchor_point": correction_meta_round.get("anchor_point"),
                                    "gemini_score_before": (correction_meta_round.get("rescore") or {}).get("gemini_score_before"),
                                    "gemini_score_after": (correction_meta_round.get("rescore") or {}).get("gemini_score_after"),
                                    "rollback": (correction_meta_round.get("rescore") or {}).get("did_rollback"),
                                }
                            multi_object_history.append(
                                {
                                    "round": i,
                                    "target": tgt,
                                    "anchor": anchor_name,
                                    "output": os.path.abspath(round_final),
                                    "feedback_hint": _fb_summary,
                                    "plan": {
                                        "need_correction": True,
                                        "source": "inference_size_correction_multi_round",
                                        "round_folder": f"round_{i:02d}",
                                    },
                                    "scale_exec": None,
                                    "correction_meta_path": os.path.abspath(cm_path)
                                    if correction_meta_round is not None
                                    else None,
                                    "correction_meta": correction_meta_round,
                                }
                            )
                            current_image_path = round_final

                    # Always save final_multi_object_result.png + multi_object_history.json
                    # even if some (or all) rounds failed.  This ensures a consistent
                    # directory structure for downstream evaluation.
                    _save_final_img = current_image_path if os.path.exists(current_image_path) else args.image_path
                    final_dir = os.path.join(multi_root, "final")
                    os.makedirs(final_dir, exist_ok=True)
                    shutil.copy2(_save_final_img, os.path.join(final_dir, "final_result.png"))
                    top_final = os.path.join(multi_root, "final_multi_object_result.png")
                    shutil.copy2(_save_final_img, top_final)
                    # Append a record for every planned target that was never attempted
                    # (because an earlier round aborted the chain). Skip when scene prefilter
                    # already decided to keep the whole image unchanged (round 0 placeholder only).
                    if not skip_corrections:
                        _attempted_rounds = {entry["round"] for entry in multi_object_history}
                        for _i_miss, _item_miss in enumerate(targets, 1):
                            if _i_miss not in _attempted_rounds:
                                multi_object_history.append(
                                    {
                                        "round": _i_miss,
                                        "target": _item_miss["name"],
                                        "anchor": anchor_name,
                                        "output": None,
                                        "plan": {
                                            "need_correction": True,
                                            "source": "inference_size_correction_multi_round",
                                            "round_folder": f"round_{_i_miss:02d}",
                                        },
                                        "scale_exec": None,
                                        "correction_meta_path": None,
                                        "correction_meta": None,
                                        "skipped": True,
                                        "skip_reason": multi_round_abort_reason or "earlier_round_failed",
                                    }
                                )
                    with open(
                        os.path.join(multi_root, "multi_object_history.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            {
                                "scene_prefilter": scene_prefilter,
                                "multi_object_history": multi_object_history,
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

                    # layout_final.json: started as clone of layout_original; bboxes patched per round
                    # from correction_meta.bbox_target_scaled_pixels_yxyx (no Gemini re-ground at end).
                    creati_final_json_path = os.path.join(multi_root, "layout_final.json")
                    try:
                        if layout_final_doc is not None:
                            layout_final_doc["image_path"] = os.path.abspath(top_final)
                            with open(creati_final_json_path, "w", encoding="utf-8") as f_fin:
                                json.dump(layout_final_doc, f_fin, indent=2, ensure_ascii=False)
                            print(f"   [Layout] Saved layout_final.json: {creati_final_json_path}", flush=True)
                        elif os.path.isfile(creati_original_json_path):
                            with open(creati_original_json_path, "r", encoding="utf-8") as _lf_r:
                                _lf_fb = json.load(_lf_r)
                            _lf_fb["image_path"] = os.path.abspath(top_final)
                            with open(creati_final_json_path, "w", encoding="utf-8") as f_fin:
                                json.dump(_lf_fb, f_fin, indent=2, ensure_ascii=False)
                            print(
                                f"   [Layout] Wrote layout_final from layout_original (fallback): "
                                f"{creati_final_json_path}",
                                flush=True,
                            )
                    except Exception as e_fin:
                        print(f"   [Layout] Warning: failed to save layout_final.json: {e_fin}", flush=True)

                    _any_success = skip_corrections or (current_image_path != args.image_path)
                    if skip_corrections:
                        _mr_msg = "skipped (scene prefilter); artifacts saved unchanged"
                    elif _any_success:
                        _mr_msg = "finished"
                    else:
                        _mr_msg = "aborted"
                    print(
                        f"{'✅' if _any_success or skip_corrections else '⚠️'} Multi-round {_mr_msg}. "
                        f"Final: {final_dir}/final_result.png; {top_final}; multi_object_history.json",
                        flush=True,
                    )
                    if _any_success or skip_corrections:
                        return final_dir

        reason_msg = (
            f" (reason: {multi_round_abort_reason})"
            if multi_round_abort_reason
            else ""
        )
        print(f"⚠️ Multi-round detection unavailable, fallback to single-round auto mode.{reason_msg}", flush=True)
        if multi_round_root is None:
            img_stem = os.path.splitext(os.path.basename(args.image_path))[0]
            fallback_name = f"{img_stem}__fallback"
            candidate = os.path.join(args.output_dir, fallback_name)
            if os.path.exists(candidate):
                idx = 1
                while os.path.exists(f"{candidate}_{idx}"):
                    idx += 1
                candidate = f"{candidate}_{idx}"
            multi_round_root = candidate
        args.output_dir = os.path.join(multi_round_root, "fallback_single")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"   Fallback single-round output root: {args.output_dir}", flush=True)

    if not args.target_object or not args.ref_object:
        detected_target, detected_ref = analyzer.identify_scaling_targets(args.image_path)
        if detected_target and detected_ref:
            args.target_object = detected_target
            args.ref_object = detected_ref
        else:
            print("❌ Automatic detection failed. Please specify objects manually.")
            return

    session_name = f"{args.target_object}_vs_{args.ref_object}".replace("/", "_").replace("\\", "_")
    final_output_dir = os.path.join(args.output_dir, session_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📁 Output Directory: {final_output_dir}")

    print(f"🖼️  Processing: {args.image_path}")
    orig_img = Image.open(args.image_path).convert("RGB")
    W1, H1 = orig_img.size
    orig_img.save(f"{final_output_dir}/original_input.png")

    print("\n🧠 [1/5] Gemini Analysis (Adjustment Plan)...")
    if (args.override_scale_factor is None) ^ (args.override_anchor_point is None):
        print("⚠️  override_scale_factor and override_anchor_point must be provided together; fallback to Gemini full planning.")
    use_override = args.override_scale_factor is not None and args.override_anchor_point is not None
    if use_override:
        raw_bbox = analyzer.detect_target_bbox(args.image_path, args.target_object)
        if raw_bbox is None:
            print("❌ Gemini bbox detection failed for override mode.")
            # Save original as final for eval completeness.
            save_path = f"{final_output_dir}/final_result.png"
            orig_img.save(save_path)
            print(f"  Saved: {save_path}", flush=True)
            return final_output_dir
        plan = {
            "original_bbox": raw_bbox,
            "scale_factor": float(args.override_scale_factor),
            "anchor_point": args.override_anchor_point,
        }
        print(
            f"   Using external override plan: scale={plan['scale_factor']}, "
            f"anchor={plan['anchor_point']} (bbox from Gemini detect-only).",
            flush=True,
        )
    else:
        plan = analyzer.get_adjustment_plan(args.image_path, args.target_object, args.ref_object)
        if not plan:
            # Save original as final for eval completeness.
            save_path = f"{final_output_dir}/final_result.png"
            orig_img.save(save_path)
            print(f"  Saved: {save_path}", flush=True)
            return final_output_dir

    if plan.get("_skip_correction"):
        print(
            "✅ Feedback size_score=3 (name-matched pair): relative scale already OK. "
            "Skip editing; save original as final_result.",
            flush=True,
        )
        save_path = f"{final_output_dir}/final_result.png"
        orig_img.save(save_path)
        print(f"  Saved: {save_path}", flush=True)
        return final_output_dir

    orig_bbox_raw = plan["original_bbox"]
    if isinstance(orig_bbox_raw[0], list):
        orig_bbox_raw = orig_bbox_raw[0]
    orig_bbox_raw = [float(x) for x in orig_bbox_raw]

    is_normalized = (max(orig_bbox_raw) <= 1.0)
    scale_y = H1 if is_normalized else H1 / 1000.0
    scale_x = W1 if is_normalized else W1 / 1000.0

    orig_y1 = int(orig_bbox_raw[0] * scale_y)
    orig_x1 = int(orig_bbox_raw[1] * scale_x)
    orig_y2 = int(orig_bbox_raw[2] * scale_y)
    orig_x2 = int(orig_bbox_raw[3] * scale_x)
    orig_box_px = [orig_y1, orig_x1, orig_y2, orig_x2]

    if (orig_x2 - orig_x1) < 10 or (orig_y2 - orig_y1) < 10:
        print(f"❌ Error: Gemini returned an invalid/empty BBox for target: {orig_box_px}.")
        save_path = f"{final_output_dir}/final_result.png"
        orig_img.save(save_path)
        print(f"  Saved: {save_path}", flush=True)
        return final_output_dir

    scale_factor = float(plan["scale_factor"])
    anchor = plan["anchor_point"]
    # Hard cap: never apply more than 2.5× scaling in a single step (prevents extreme overcorrection).
    effective_min_scale = max(_SCALE_HARD_MIN, float(args.min_scale))
    effective_max_scale = min(_SCALE_HARD_MAX, max(effective_min_scale, float(args.max_scale)))
    scale_factor = max(effective_min_scale, min(effective_max_scale, scale_factor))
    # Clip scale so the enlarged bbox fits within the image boundary.
    scale_factor = clip_scale_to_image_boundary(orig_box_px, scale_factor, anchor, W1, H1)

    new_box_px = get_new_bbox_from_anchor_unclamped(orig_box_px, scale_factor, anchor)

    print(f"   Original Box (Orig Px): {orig_box_px}")
    print(f"   Target Box   (Orig Px): {new_box_px} (Anchor: {anchor}, Scale: {scale_factor})")

    # If the plan is effectively identity, skip expensive steps and keep original image.
    min_delta = float(getattr(args, "min_scale_change", 0.0) or 0.0)
    if min_delta > 0.0 and abs(float(scale_factor) - 1.0) < min_delta:
        print(
            f"✅ Scale change too small (abs({scale_factor:.3f}-1) < {min_delta:.3f}). "
            "Skip editing; save original as final_result.",
            flush=True,
        )
        save_path = f"{final_output_dir}/final_result.png"
        orig_img.save(save_path)
        print(f"  Saved: {save_path}", flush=True)
        return final_output_dir

    if args.debug_draw_bbox:
        draw_bbox(orig_img, orig_box_px, f"{final_output_dir}/debug_bbox_orig.png")
        draw_bbox(orig_img, new_box_px,  f"{final_output_dir}/debug_bbox_new_on_orig.png", color=(0, 255, 0))

    # --------------------------------------------------------------------------
    # 2. Extract Reference (SAM2 box prompt — robust to curved objects)
    # --------------------------------------------------------------------------
    print("\n✂️ [2/5] Extracting Reference (SAM2 box prompt)...")
    y1_orig, x1_orig, y2_orig, x2_orig = orig_box_px

    if int(args.cache_models_cpu) == 1:
        sam_key = ("sam2", int(args.gpu_tools))
        sam = _cache_get(sam_key, lambda: SAM2Segmentor(args.gpu_tools))
        sam.to_device(f"cuda:{args.gpu_tools}")
    else:
        sam = SAM2Segmentor(args.gpu_tools)
    orig_img_np = np.array(orig_img)
    # SAM2 box prompt on the full image — robust to concave/curved objects
    bbox_xyxy = [x1_orig, y1_orig, x2_orig, y2_orig]
    full_mask = sam.segment_with_box(orig_img_np, bbox_xyxy)
    full_mask = select_best_component_in_bbox(full_mask, [y1_orig, x1_orig, y2_orig, x2_orig])
    if int(args.cache_models_cpu) == 1:
        sam.to_device("cpu")
        torch.cuda.empty_cache()
    else:
        sam.unload()
        del sam

    # Recompute transform bbox from SAM2 mask for better geometric stability.
    if int(args.use_seg_bbox_for_transform) == 1:
        rows = np.any(full_mask > 0, axis=1)
        cols = np.any(full_mask > 0, axis=0)
        if np.any(rows) and np.any(cols):
            my1, my2 = np.where(rows)[0][[0, -1]]
            mx1, mx2 = np.where(cols)[0][[0, -1]]
            seg_box_px = [int(my1), int(mx1), int(my2) + 1, int(mx2) + 1]
            if (seg_box_px[3] - seg_box_px[1]) >= 10 and (seg_box_px[2] - seg_box_px[0]) >= 10:
                # Re-clip with tighter SAM2 bbox.
                scale_factor = clip_scale_to_image_boundary(seg_box_px, scale_factor, anchor, W1, H1)
                new_box_px = get_new_bbox_from_anchor_unclamped(seg_box_px, scale_factor, anchor)
                orig_box_px = seg_box_px
                y1_orig, x1_orig, y2_orig, x2_orig = orig_box_px
                print(f"   Using SAM2 tight bbox for transform: {orig_box_px}", flush=True)
                print(f"   Recomputed target box: {new_box_px} (Anchor: {anchor}, Scale: {scale_factor})", flush=True)

    m = 10
    cy1 = max(0, y1_orig - m)
    cx1 = max(0, x1_orig - m)
    cy2 = min(H1, y2_orig + m)
    cx2 = min(W1, x2_orig + m)
    ref_crop = orig_img_np[cy1:cy2, cx1:cx2]
    mask_crop = full_mask[cy1:cy2, cx1:cx2]

    print("\n[Step 1b] Estimating depths ...", flush=True)
    if int(args.cache_models_cpu) == 1:
        depth_key = ("depth_anything_v2", int(args.gpu_tools))
        depth_est = _cache_get(depth_key, lambda: DepthAnythingV2Estimator(args.gpu_tools))
        depth_est.to_device(f"cuda:{args.gpu_tools}")
    else:
        depth_est = DepthAnythingV2Estimator(args.gpu_tools)
    orig_depth_map = depth_est.estimate_depth(orig_img)
    if int(args.cache_models_cpu) == 1:
        depth_est.to_device("cpu")
        torch.cuda.empty_cache()
    else:
        depth_est.unload()
        del depth_est

    orig_depth_crop = orig_depth_map[cy1:cy2, cx1:cx2]
    obj_depth_values = orig_depth_crop[mask_crop > 0]
    target_depth_range = (
        float(np.percentile(obj_depth_values, 5)),
        float(np.percentile(obj_depth_values, 95))
    ) if len(obj_depth_values) > 0 else None

    # Edge color decontamination: boundary pixels in the original image are a
    # blend of object + background color. Instead of eroding/blurring the mask
    # (which destroys thin features like phone bezels), we keep the mask shape
    # intact and fix the boundary pixels' COLOR by inpainting them from interior
    # object pixels. This removes the dark fringe without changing object shape.
    # SAM2 masks sometimes contain small holes inside the object (specular highlights, reflections),
    # which show up as white dots in ref_clean. Clean + fill holes before compositing.
    mask_bin = (mask_crop > 0).astype(np.uint8) * 255
    mask_bin = _clean_binary_mask(mask_bin, open_ksz=3, close_ksz=5)
    mask_bin = _fill_binary_holes(mask_bin)
    mask_bin = (mask_bin > 127).astype(np.uint8)
    interior = cv2.erode(mask_bin, np.ones((3, 3), np.uint8), iterations=1)
    edge_band = ((mask_bin - interior) > 0).astype(np.uint8)

    if int(edge_band.sum()) > 0:
        decontam_ref = cv2.inpaint(ref_crop, edge_band * 255, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        decontam_ref = ref_crop

    ref_crop_pil = Image.fromarray(decontam_ref)
    mask_pil = Image.fromarray((mask_bin * 255).astype(np.uint8))
    clean_ref = Image.new("RGB", ref_crop_pil.size, (255, 255, 255))
    clean_ref.paste(ref_crop_pil, mask=mask_pil)
    clean_ref.save(f"{final_output_dir}/ref_clean.png")
    # Save ref object mask (same canvas as ref_clean) for downstream inpainting comparisons.
    mask_pil.save(f"{final_output_dir}/ref_mask.png")

    ref_override = str(getattr(args, "reference_image_path", "") or "").strip()
    ref_override_was_applied = False
    if ref_override and os.path.isfile(ref_override):
        scene_ref_backup = clean_ref.copy()
        try:
            print(f"\n   [Ref] Using external reference image (e.g. benchmark catalog): {ref_override}", flush=True)
            bench_img = Image.open(ref_override).convert("RGB")
            tw, th = clean_ref.size
            clean_ref = letterbox_pil_to_canvas(bench_img, tw, th)
            scene_ref_backup.save(f"{final_output_dir}/ref_clean_from_scene.png")
            clean_ref.save(f"{final_output_dir}/ref_clean.png")
            ref_override_was_applied = True
            print(f"   [Ref] Letterboxed catalog image to {tw}x{th} to match scene ref canvas.", flush=True)
        except Exception as e:
            print(f"   [Ref] Override failed ({e}); keeping scene-based ref_clean.", flush=True)
            clean_ref = scene_ref_backup
            clean_ref.save(f"{final_output_dir}/ref_clean.png")
    elif ref_override:
        print(f"\n   [Ref] reference_image_path not found: {ref_override} — using scene ref.", flush=True)

    # Upscale low-res ref with Gemini to recover detail for the generation model.
    ref_w, ref_h = clean_ref.size
    ref_max_dim = max(ref_w, ref_h)
    upscale_thresh = int(args.ref_upscale_threshold)
    if ref_max_dim < upscale_thresh and upscale_thresh > 0:
        upscale_target = int(args.ref_upscale_target)
        print(f"   ref_clean is small ({ref_w}x{ref_h}, max={ref_max_dim} < {upscale_thresh}). "
              f"Attempting Gemini upscale to ~{upscale_target}px ...", flush=True)
        upscaled = _gemini_upscale_ref(clean_ref, upscale_target)
        if upscaled is not None:
            clean_ref = upscaled
            clean_ref.save(f"{final_output_dir}/ref_clean_upscaled.png")
            print(f"   Upscaled ref_clean: {ref_w}x{ref_h} -> {clean_ref.size[0]}x{clean_ref.size[1]}", flush=True)
        else:
            print("   Gemini upscale failed, using original ref_clean.", flush=True)

    # --------------------------------------------------------------------------
    # 3. Clean Background (instruction editing preferred)
    # --------------------------------------------------------------------------
    removal_kernel = np.ones((15, 15), np.uint8)
    removal_mask = cv2.dilate(full_mask * 255, removal_kernel, iterations=3)

    def _run_gemini_edit_fallback():
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            print("   [Gemini cleanup] GOOGLE_API_KEY not set; skip Gemini cleanup.", flush=True)
            return None
        try:
            buf = io.BytesIO()
            orig_img.save(buf, format="PNG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-3.1-flash-image-preview:generateContent?key={api_key}"
            )
            prompt = (
                f"Remove the {args.target_object} completely from this image. "
                f"Also remove physical effects caused by the {args.target_object} "
                "(reflection, mirror highlight, cast shadow, contact shadow, glow). "
                "Fill missing regions with a natural photorealistic background. "
                "Do not modify any other object in the image. "
                "Preserve all unrelated objects, geometry, perspective, lighting, and texture."
            )
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/png", "data": b64_img}},
                    ]
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                },
            }
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=90,
            )
            if resp.status_code != 200:
                print(f"   [Gemini cleanup] API error {resp.status_code}: {resp.text[:400]}", flush=True)
                return None
            result = resp.json()
            parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for p in parts:
                inline = p.get("inlineData", {})
                data = inline.get("data", None)
                if data:
                    img_bytes = base64.b64decode(data)
                    out = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if out.size != (W1, H1):
                        print(f"   [Gemini cleanup] resize output {out.size} -> {(W1, H1)}", flush=True)
                        out = out.resize((W1, H1), Image.LANCZOS)
                    return out
            print("   [Gemini cleanup] no image data in response.", flush=True)
            return None
        except Exception as e:
            print(f"   [Gemini cleanup] failed: {e}", flush=True)
            return None

    def _evaluate_remove_quality(orig_pil, cleaned_pil, mask255):
        """
        Return cleanup quality tuple:
          (target_diff_mean, target_changed_ratio, non_target_diff_mean, non_target_changed_ratio)
        """
        target = mask255 > 0
        non_target = ~target

        if np.sum(target) == 0:
            return 0.0, 0.0, 0.0, 0.0

        a = np.array(orig_pil).astype(np.float32) / 255.0
        b = np.array(cleaned_pil).astype(np.float32) / 255.0
        d = np.mean(np.abs(a - b), axis=2)

        target_diff_mean = float(d[target].mean())
        target_changed_ratio = float((d[target] > float(args.bg_remove_pixel_diff_cutoff)).mean())

        if np.sum(non_target) > 0:
            non_target_diff_mean = float(d[non_target].mean())
            # Outside target we use a slightly stricter "changed" cutoff.
            non_target_cutoff = max(0.02, float(args.bg_remove_pixel_diff_cutoff) * 0.75)
            non_target_changed_ratio = float((d[non_target] > non_target_cutoff).mean())
        else:
            non_target_diff_mean = 0.0
            non_target_changed_ratio = 0.0

        return (
            target_diff_mean,
            target_changed_ratio,
            non_target_diff_mean,
            non_target_changed_ratio,
        )

    def _remove_quality_pass(metrics):
        td, tc, nd, nc = metrics
        target_ok = (
            td >= float(args.bg_remove_diff_thresh)
            and tc >= float(args.bg_remove_changed_ratio_thresh)
        )
        preserve_ok = True
        if int(args.enforce_bg_preserve) == 1:
            preserve_ok = (
                nd <= float(args.bg_preserve_diff_mean_max)
                and nc <= float(args.bg_preserve_changed_ratio_max)
            )
        return target_ok and preserve_ok, target_ok, preserve_ok

    def _metrics_str(metrics):
        td, tc, nd, nc = metrics
        return (
            f"target_diff={td:.3f}, target_changed={tc:.3f}, "
            f"non_target_diff={nd:.3f}, non_target_changed={nc:.3f}"
        )

    def _parse_gemini_json_text(text: str):
        text = (text or "").strip()
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        if start < 0:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _gemini_overremove_check(cleaned_pil):
        """
        Cheap semantic QA after removal.  Pixel diff gates cannot tell whether a
        neighboring object was erased together with the target; Gemini flash can
        catch that before generation consumes bg_clean.
        """
        if int(getattr(args, "bg_overremove_check", 0)) != 1:
            return True, {"enabled": False}
        preserve_names = [
            x.strip()
            for x in str(getattr(args, "preserve_object_names", "") or "").split(",")
            if x.strip()
        ]
        if not preserve_names:
            return True, {"enabled": True, "skipped": "no_preserve_object_names"}
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return True, {"enabled": True, "skipped": "GOOGLE_API_KEY_missing"}

        def _b64_png(im):
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "You are checking an object-removal result.\n"
            "You will see two images: Image 1 is the original image, Image 2 is the image after removal.\n"
            f"Target object that SHOULD be removed: {args.target_object!r}.\n"
            f"Objects that MUST remain visible and not substantially damaged: {preserve_names!r}.\n\n"
            "Return JSON only with:\n"
            "{\n"
            '  "target_removed": true/false,\n'
            '  "over_removed": true/false,\n'
            '  "missing_or_damaged_objects": ["..."],\n'
            '  "brief_reason": "..."\n'
            "}\n\n"
            "Set over_removed=true if any preserved object is erased, mostly erased, moved, merged, "
            "severely blurred, or replaced by background. Ignore small texture/color changes."
        )
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{str(getattr(args, 'bg_overremove_model', 'gemini-3-flash-preview'))}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"text": "Image 1: original"},
                        {"inline_data": {"mime_type": "image/png", "data": _b64_png(orig_img)}},
                        {"text": "Image 2: after removal"},
                        {"inline_data": {"mime_type": "image/png", "data": _b64_png(cleaned_pil)}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"},
        }
        try:
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            if resp.status_code != 200:
                return True, {"enabled": True, "api_failed": f"HTTP {resp.status_code}", "body": resp.text[:400]}
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            obj = _parse_gemini_json_text(text)
            if not isinstance(obj, dict):
                return True, {"enabled": True, "api_failed": "parse_failed", "raw": text[:400]}
            over = bool(obj.get("over_removed"))
            target_removed = obj.get("target_removed")
            ok = (not over) and (target_removed is not False)
            obj["enabled"] = True
            return ok, obj
        except Exception as e:
            return True, {"enabled": True, "api_failed": repr(e)}

    def _try_gemini_cleanup_with_retries(tag_prefix, max_trials=2):
        max_trials = max(1, int(max_trials))
        last_candidate = None
        for t in range(max_trials):
            candidate = _run_gemini_edit_fallback()
            if candidate is None:
                print(f"   [{tag_prefix} {t+1}/{max_trials}] no image returned.", flush=True)
                continue
            last_candidate = candidate
            metrics = _evaluate_remove_quality(orig_img, candidate, removal_mask)
            overall_ok, target_ok, preserve_ok = _remove_quality_pass(metrics)
            semantic_ok, semantic_info = _gemini_overremove_check(candidate)
            print(
                f"   [{tag_prefix} {t+1}/{max_trials}] {_metrics_str(metrics)} "
                f"(target_ok={target_ok}, preserve_ok={preserve_ok}, semantic_ok={semantic_ok})",
                flush=True,
            )
            if not semantic_ok:
                print(f"      [overremove] reject candidate: {semantic_info}", flush=True)
            if overall_ok and semantic_ok:
                return candidate, metrics, True
        if last_candidate is None:
            return None, None, False
        return last_candidate, _evaluate_remove_quality(orig_img, last_candidate, removal_mask), False

    removal_mode = str(args.bg_removal_mode).lower()
    if removal_mode == "editing":
        removal_mode = "flux"

    if removal_mode == "gemini":
        print("\n🧹 [3/5] Cleaning Background (Gemini image-edit only) ...")
        clean_bg, metrics, passed = _try_gemini_cleanup_with_retries(
            tag_prefix="Gemini cleanup only", max_trials=3
        )
        if clean_bg is None:
            print("   -> Gemini cleanup unavailable/failed; stop this case.", flush=True)
            return
        if not passed:
            print("   [Gemini cleanup only] quality check failed; stop this case.", flush=True)
            return
    elif removal_mode == "flux":
        print("\n🧹 [3/5] Cleaning Background (Open-weights Image Editing) ...")
        clean_bg = None
        passed = False

        if int(args.prefer_gemini_cleanup) == 1 or int(args.gemini_cleanup_only) == 1:
            print("   -> Trying Gemini image-edit cleanup first ...", flush=True)
            clean_bg, metrics, passed = _try_gemini_cleanup_with_retries(
                tag_prefix="Gemini cleanup first", max_trials=3
            )
            if passed:
                print("   ✅ Gemini cleanup passed quality check.", flush=True)
            else:
                clean_bg = None
                print("   [Gemini cleanup first] quality check failed.", flush=True)

            if int(args.gemini_cleanup_only) == 1 and not passed:
                print("   -> gemini_cleanup_only=1 and Gemini cleanup failed; stop this case.", flush=True)
                return

        remover = None
        if not passed:
            if int(args.cache_models_cpu) == 1:
                remover_key = ("flux_kontext_remover", int(args.gpu_tools), FLUX_KONTEXT)
                remover = _cache_get(
                    remover_key,
                    lambda: InstructionEditingRemover(
                        device_id=args.gpu_tools,
                        kontext_path=FLUX_KONTEXT,
                    ),
                )
                remover.to_device(f"cuda:{args.gpu_tools}")
            else:
                remover = InstructionEditingRemover(
                    device_id=args.gpu_tools,
                    kontext_path=FLUX_KONTEXT,
                )

        prompts = [
            (
                "Remove the {obj} completely. Also remove its related physical effects "
                "(reflection, mirror highlight, cast shadow, contact shadow, glow) if caused by the {obj}. "
                "Fill the missing area with natural and photorealistic background. "
                "STRICT: Preserve all unrelated objects exactly. Do not erase, move, resize, recolor, or blur any non-target object. "
                "Preserve geometry, perspective, lighting, and texture outside target area."
            ),
            (
                "Delete every visible part of the {obj}. No {obj} should remain. "
                "Also clean correlated reflection/shadow traces tied to the {obj}. "
                "STRICT: Do not alter anything unrelated to the {obj}; all other objects must remain unchanged. "
                "Reconstruct the original background and surface texture naturally."
            ),
            (
                "Erase the {obj} entirely including boundaries, shadow/contact artifacts, reflection traces, and color traces. "
                "Keep scene composition unchanged and restore plausible background details only where {obj} was. "
                "Do not modify unrelated scene content."
            ),
        ]
        preserve_names_for_prompt = [
            x.strip()
            for x in str(getattr(args, "preserve_object_names", "") or "").split(",")
            if x.strip()
        ]
        preserve_prompt = ""
        if preserve_names_for_prompt:
            preserve_prompt = (
                " Critical preserved objects that must remain visible and unchanged: "
                + ", ".join(preserve_names_for_prompt)
                + "."
            )

        if not passed:
            max_trials = max(1, int(args.bg_remove_max_retries))
            for i in range(max_trials):
                prompt_i = prompts[min(i, len(prompts) - 1)].format(obj=args.target_object) + preserve_prompt
                candidate = remover.remove_object(
                    orig_img,
                    args.target_object,
                    num_steps=int(args.editing_steps),
                    guidance_scale=float(args.editing_guidance_scale),
                    seed=42 + i * 13,
                    prompt_override=prompt_i,
                )
                metrics = _evaluate_remove_quality(orig_img, candidate, removal_mask)
                overall_ok, target_ok, preserve_ok = _remove_quality_pass(metrics)
                semantic_ok, semantic_info = _gemini_overremove_check(candidate)
                print(
                    f"   [Remove trial {i+1}/{max_trials}] {_metrics_str(metrics)} "
                    f"(target_ok={target_ok}, preserve_ok={preserve_ok}, semantic_ok={semantic_ok})",
                    flush=True,
                )
                if not semantic_ok:
                    print(f"      [overremove] reject candidate: {semantic_info}", flush=True)
                if overall_ok and semantic_ok:
                    clean_bg = candidate
                    passed = True
                    print("   ✅ Editing-based removal passed quality check.", flush=True)
                    break
                clean_bg = candidate

        if not passed:
            print("   ⚠️ Editing removal did not pass quality check.", flush=True)
            clean_bg = None
            if int(args.use_gemini_cleanup_fallback) == 1:
                print("   -> Trying Gemini image-edit cleanup fallback ...", flush=True)
                clean_bg, metrics, fallback_passed = _try_gemini_cleanup_with_retries(
                    tag_prefix="Gemini cleanup fallback", max_trials=3
                )
                if clean_bg is not None and not fallback_passed:
                    print("   [Gemini cleanup] quality check failed; stop this case.", flush=True)
                    return
            if clean_bg is None:
                print("   -> Gemini cleanup unavailable/failed; stop this case.", flush=True)
                return

        if remover is not None:
            if int(args.cache_models_cpu) == 1:
                remover.to_device("cpu")
                torch.cuda.empty_cache()
            else:
                remover.unload()
                del remover
    else:
        print(f"\n[Step 2/5] Unsupported bg_removal_mode='{args.bg_removal_mode}'; stop this case.", flush=True)
        return

    if clean_bg.size != (W1, H1):
        print(f"⚠️ Background editor changed size {clean_bg.size} -> {(W1, H1)}; resizing back.")
        clean_bg = clean_bg.resize((W1, H1), Image.LANCZOS)
    clean_bg.save(f"{final_output_dir}/bg_clean.png")

    # FLUX.1-Fill preserves resolution (no Scheme B scaling needed)
    W, H = clean_bg.size
    y1_new, x1_new, y2_new, x2_new = new_box_px

    if args.debug_draw_bbox:
        draw_bbox(clean_bg, orig_box_px, f"{final_output_dir}/debug_bbox_orig_on_bg.png")
        draw_bbox(clean_bg, new_box_px,  f"{final_output_dir}/debug_bbox_new_on_bg.png", color=(0, 255, 0))

    # --------------------------------------------------------------------------
    # 4. Adaptive Depth Fusion (mask decoupled from depth)
    # --------------------------------------------------------------------------
    print("\n📐 [4/5] Adaptive Depth Fusion...")
    if int(args.cache_models_cpu) == 1:
        depth_key = ("depth_anything_v2", int(args.gpu_tools))
        depth_est = _cache_get(depth_key, lambda: DepthAnythingV2Estimator(args.gpu_tools))
        depth_est.to_device(f"cuda:{args.gpu_tools}")
    else:
        depth_est = DepthAnythingV2Estimator(args.gpu_tools)
    bg_depth = depth_est.estimate_depth(clean_bg)
    ref_depth = estimate_ref_depth_safe(depth_est, clean_ref)
    if int(args.cache_models_cpu) == 1:
        depth_est.to_device("cpu")
        torch.cuda.empty_cache()
    else:
        depth_est.unload()

    # When catalog ref replaces scene crop, SAM mask no longer matches ref_depth.
    # Use mask + depth range from clean_ref so fusion is bg_clean depth + ref_depth only.
    if ref_override_was_applied:
        mask_fusion_src = build_mask_for_ref_depth_fusion(clean_ref)
        if int(np.sum(mask_fusion_src > 127)) < 32:
            print(
                "   [Depth] ref-derived mask nearly empty; falling back to scene SAM mask for fusion.",
                flush=True,
            )
            mask_fusion_src = mask_crop
        else:
            cv2.imwrite(f"{final_output_dir}/debug_depth_fusion_mask_ref.png", mask_fusion_src)
            rd_vals = ref_depth.astype(np.float32)
            m = mask_fusion_src > 127
            target_depth_range = (
                float(np.percentile(rd_vals[m], 5)),
                float(np.percentile(rd_vals[m], 95)),
            )
            print(
                f"   [Depth] catalog ref: fusion mask + ref_depth_range from clean_ref "
                f"(range={target_depth_range[0]:.1f}-{target_depth_range[1]:.1f})",
                flush=True,
            )
    else:
        mask_fusion_src = mask_crop

    # Adaptive depth bias: only add bias when fg/bg depth values are too close
    # to distinguish. Compute expected object depth vs background depth at target.
    max_bias = float(args.bottom_center_depth_bias)
    norm_bg = cv2.normalize(bg_depth.copy().astype(np.float32), None, 0, 255,
                            cv2.NORM_MINMAX, cv2.CV_32F)
    ty1, tx1 = max(0, y1_new), max(0, x1_new)
    ty2, tx2 = min(H, y2_new), min(W, x2_new)
    fg_bg_gap = -1.0
    depth_bias = 0.0
    if ty2 > ty1 and tx2 > tx1:
        bg_patch_depth = norm_bg[ty1:ty2, tx1:tx2]
        bg_median = float(np.median(bg_patch_depth))
        obj_median = float(np.median(target_depth_range)) if target_depth_range else 200.0
        fg_bg_gap = abs(obj_median - bg_median)
        if fg_bg_gap < 30:
            depth_bias = max_bias
        elif fg_bg_gap < 60:
            depth_bias = max_bias * 0.3
        else:
            depth_bias = 0.0
    fused_depth, fused_mask, use_depth, _, _ = adaptive_depth_fusion(
        bg_depth.copy().astype(np.float32),
        ref_depth.astype(np.float32),
        mask_fusion_src,
        [y1_new, x1_new, y2_new, x2_new],
        target_depth_range=target_depth_range,
        coverage_threshold=0.5,
        ref_depth_bias=depth_bias,
    )
    print(f"   Depth mode: use_depth={use_depth}, bias={depth_bias:.1f} (fg_bg_gap={fg_bg_gap:.1f})", flush=True)

    tgt_h = max(1, y2_new - y1_new)
    tgt_w = max(1, x2_new - x1_new)
    tgt_min_dim = min(tgt_h, tgt_w)

    if int(args.mask_dilate_depth_aware) == 1:
        # Shape-preserving dilation: expand the object contour uniformly by
        # border_px using distance transform.  Unlike square-kernel dilation
        # (Minkowski sum with a square), this performs a circular expansion
        # that keeps the mask aspect ratio close to the depth shape — matching
        # the training distribution where mask ≈ dilated GT object mask.
        dilate_ratio = float(args.mask_dilate_ratio)
        border_px = max(3, int(tgt_min_dim * dilate_ratio))
        mask_bin = (fused_mask > 127).astype(np.uint8)
        inv_mask = (1 - mask_bin).astype(np.uint8)
        dist_outside = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
        fused_mask = (dist_outside <= border_px).astype(np.uint8) * 255
        args._depth_aware_border_px = border_px
        print(f"   Mask dilation (depth-aware): border={border_px}px "
              f"(ratio={dilate_ratio:.2f}, target_box={tgt_h}x{tgt_w}, "
              f"min_dim={tgt_min_dim})", flush=True)
    else:
        _db = float(args.mask_dilate_boost)
        _dib = float(args.mask_dilate_iter_boost)
        adaptive_ksz = max(
            3,
            min(
                int(round(float(args.mask_dilate_kernel) * _db)),
                int(tgt_min_dim * 0.32),
            ),
        )
        if adaptive_ksz % 2 == 0:
            adaptive_ksz += 1
        adaptive_iter = max(
            1,
            min(
                int(round(float(args.mask_dilate_iter) * _dib)),
                int(tgt_min_dim * 0.20),
            ),
        )
        args._depth_aware_border_px = None
        print(f"   Mask dilation (uniform): kernel={adaptive_ksz}, iter={adaptive_iter} "
              f"(target_box={tgt_h}x{tgt_w}, min_dim={tgt_min_dim})", flush=True)
        kernel = np.ones((adaptive_ksz, adaptive_ksz), np.uint8)
        fused_mask = cv2.dilate(fused_mask, kernel, iterations=adaptive_iter)

    _reg = str(getattr(args, "mask_inpaint_regularize", "close_convex_hull") or "none").lower()
    if _reg not in ("none", "0", "off", ""):
        cv2.imwrite(f"{final_output_dir}/debug_inpaint_mask_before_regularize.png", fused_mask)
        fused_mask = regularize_inpaint_mask(
            fused_mask,
            mode=_reg,
            close_ksz=int(getattr(args, "mask_regularize_close_ksz", 15)),
            close_iter=int(getattr(args, "mask_regularize_close_iter", 2)),
        )
        print(
            f"   Mask regularize: mode={_reg} "
            f"(close ksz={int(getattr(args, 'mask_regularize_close_ksz', 15))}, "
            f"iter={int(getattr(args, 'mask_regularize_close_iter', 2))})",
            flush=True,
        )

    cv2.imwrite(f"{final_output_dir}/depth_fused.png", fused_depth.astype(np.uint8))
    depth_pil = Image.fromarray(fused_depth.astype(np.uint8)).convert("RGB")
    target_mask_pil = Image.fromarray(fused_mask).convert("L")
    # Save full-resolution target mask for downstream inpainting comparisons.
    target_mask_pil.save(f"{final_output_dir}/mask_fused.png")
    # --------------------------------------------------------------------------
    # 5. Generate (Zoom-in crop around target box)
    # --------------------------------------------------------------------------
    force_nodepth = int(getattr(args, "force_no_depth_control", 0)) == 1
    use_depth_for_gen = bool(use_depth) and (not force_nodepth)
    print(f"\n✨ [5/5] Generating Result (use_depth={use_depth_for_gen})...")
    if force_nodepth and use_depth:
        print("   [Gen] force_no_depth_control=1: overriding use_depth=True -> False (mask-only).", flush=True)

    bbox_yx = [y1_new, x1_new, y2_new, x2_new]

    crop_yx, crop_center = get_padded_square_crop_coords(
        bbox_yx,
        ratio=float(args.crop_ratio),
        image_size=(W, H),
        bias_y=0.1,
        bias_x=0.00,
    )

    crop_yx = shift_crop_to_bounds(crop_yx, W, H)

    print(f"   Zooming into Center: {crop_center}, Shifted Crop Coords: {crop_yx}, crop_ratio={args.crop_ratio}")

    # padding 仍保留兜底，但通常不会再出现大白边
    bg_crop = smart_crop_with_padding(clean_bg, crop_yx, pad_value=(255, 255, 255))
    mask_crop = smart_crop_with_padding(target_mask_pil, crop_yx, pad_value=0)
    depth_crop = smart_crop_with_padding(depth_pil, crop_yx, pad_value=0)

    bg_crop_re = bg_crop.resize((768, 768), Image.LANCZOS)
    mask_crop_re = mask_crop.resize((768, 768), Image.NEAREST)
    depth_crop_re = depth_crop.resize((768, 768), Image.NEAREST)

    bg_crop_re.save(f"{final_output_dir}/debug_crop_input.png")
    depth_crop_re.save(f"{final_output_dir}/debug_crop_depth.png")

    mask_arr = np.array(mask_crop_re)
    if int(getattr(args, "mask_crop_fill_holes", 1)) == 1:
        mask_arr = _fill_binary_holes(mask_arr)
    _pd = int(getattr(args, "mask_crop_post_dilate", 0))
    if _pd > 0:
        _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_arr = cv2.dilate((mask_arr > 127).astype(np.uint8) * 255, _k, iterations=_pd)
    mask_crop_re = Image.fromarray(mask_arr.astype(np.uint8), mode="L")
    mask_crop_re.save(f"{final_output_dir}/debug_crop_mask.png")

    if np.sum(mask_arr > 0) < 100:
        print("Warning: mask is nearly empty after cropping. Skipping generation.", flush=True)
        return final_output_dir

    _use_hf = None if args.use_hf_inject is None else bool(int(args.use_hf_inject))
    _hf_r = float(args.hf_hp_radius)

    if int(args.cache_models_cpu) == 1:
        infer_key = (
            "insert_anything",
            args.weights_dir,
            FLUX_FILL,
            FLUX_REDUX,
            int(args.gpu_gen),
            _use_hf,
            _hf_r,
        )
        inferencer = _cache_get(
            infer_key,
            lambda: InsertAnythingInferencer(
                args.weights_dir,
                FLUX_FILL,
                FLUX_REDUX,
                f"cuda:{args.gpu_gen}",
                use_hf_inject=_use_hf,
                hf_hp_radius=_hf_r,
            ),
        )
        inferencer.to_device(f"cuda:{args.gpu_gen}")
    else:
        inferencer = InsertAnythingInferencer(
            args.weights_dir,
            FLUX_FILL,
            FLUX_REDUX,
            f"cuda:{args.gpu_gen}",
            use_hf_inject=_use_hf,
            hf_hp_radius=_hf_r,
        )

    result_crop = inferencer.generate(
        clean_ref,
        bg_crop_re,
        mask_crop_re,
        depth_crop_re,
        num_steps=int(args.num_steps),
        guidance_scale=float(args.guidance_scale),
        controlnet_scale=float(args.controlnet_scale),
        controlnet_end=float(args.controlnet_end),
        seed=int(args.seed),
        use_depth_control=use_depth_for_gen,
    )
    result_crop.save(f"{final_output_dir}/debug_crop_result_raw.png")

    # --- Save HF visualizations for debugging detail preservation ---
    try:
        _hf_radius = float(args.hf_hp_radius)
        _ref_np = np.array(clean_ref.convert("RGB").resize((768, 768), Image.LANCZOS))
        _res_np = np.array(result_crop.convert("RGB"))
        hf_ref_vis = high_frequency_map_rgb_numpy(_ref_np, radius_frac=_hf_radius)
        hf_res_vis = high_frequency_map_rgb_numpy(_res_np, radius_frac=_hf_radius)
        Image.fromarray((hf_ref_vis * 255).clip(0, 255).astype(np.uint8)).save(
            f"{final_output_dir}/debug_hf_reference.png"
        )
        Image.fromarray((hf_res_vis * 255).clip(0, 255).astype(np.uint8)).save(
            f"{final_output_dir}/debug_hf_result.png"
        )
        # Side-by-side: ref | result | hf_ref | hf_result
        _side = np.concatenate([
            _ref_np,
            _res_np,
            (hf_ref_vis * 255).clip(0, 255).astype(np.uint8),
            (hf_res_vis * 255).clip(0, 255).astype(np.uint8),
        ], axis=1)
        Image.fromarray(_side).save(f"{final_output_dir}/debug_hf_comparison.png")
        print(f"  HF visualizations saved to {final_output_dir}/debug_hf_*.png")
    except Exception as _hf_e:
        print(f"  [warn] HF visualization failed: {_hf_e}")

    result_crop_nodepth = None
    choose_nodepth = False
    if int(args.enable_nodepth_compare) == 1:
        result_crop_nodepth = inferencer.generate(
            clean_ref,
            bg_crop_re,
            mask_crop_re,
            depth_crop_re,
            num_steps=int(args.num_steps),
            guidance_scale=float(args.guidance_scale),
            controlnet_scale=0.4,
            controlnet_end=0.5,
            seed=int(args.seed),
            use_depth_control=False,
        )
        result_crop_nodepth.save(f"{final_output_dir}/debug_crop_result_raw_nodepth.png")

        def _object_presence_score(base_img, cand_img, mask_img):
            m = np.array(mask_img.convert("L")) > 127
            if m.sum() < 16:
                return 0.0
            a = np.array(base_img).astype(np.float32) / 255.0
            b = np.array(cand_img).astype(np.float32) / 255.0
            d = np.mean(np.abs(a - b), axis=2)
            return float(d[m].mean())

        score_depth = _object_presence_score(bg_crop_re, result_crop, mask_crop_re)
        score_nodepth = _object_presence_score(bg_crop_re, result_crop_nodepth, mask_crop_re)
        choose_nodepth = (
            (score_depth < float(args.presence_diff_thresh) and score_nodepth > score_depth)
            or (score_nodepth > score_depth * float(args.nodepth_prefer_ratio))
        )
        print(
            f"   Presence score: depth={score_depth:.4f}, nodepth={score_nodepth:.4f}, "
            f"choose_nodepth={choose_nodepth}",
            flush=True,
        )
    else:
        print("   Depth-only mode: skip nodepth generation and force depth result.", flush=True)

    selected_crop = result_crop_nodepth if (choose_nodepth and result_crop_nodepth is not None) else result_crop


    result_crop_orig = selected_crop.resize(bg_crop.size, Image.LANCZOS)

    y1_c, x1_c, y2_c, x2_c = crop_yx
    ix1, iy1 = max(0, x1_c), max(0, y1_c)
    ix2, iy2 = min(W, x2_c), min(H, y2_c)

    if ix2 > ix1 and iy2 > iy1:
        ox, oy = ix1 - x1_c, iy1 - y1_c
        valid_w, valid_h = ix2 - ix1, iy2 - iy1

        valid_result = result_crop_orig.crop((ox, oy, ox + valid_w, oy + valid_h))

        # Alpha strategy (border-erosion approach):
        #   Erode blend_mask inward by N px  -> hard_bin  (alpha = 1, NEVER touched)
        #   Outer border ring                -> feathered (alpha 1->0 towards edge)
        #   Outside blend_mask               -> alpha = 0
        # This guarantees the generated object body is fully opaque regardless
        # of how well object_mask aligns with the actual generated pixels.
        blend_mask_back = mask_crop_re.resize(bg_crop.size, Image.NEAREST)
        valid_blend_mask = blend_mask_back.crop((ox, oy, ox + valid_w, oy + valid_h)).convert("L")

        blend_bin = (np.array(valid_blend_mask).astype(np.uint8) > 127).astype(np.uint8)

        # Adaptive feathering: cap to dilation border so feathering never
        # erodes into the actual object.  _depth_aware_border_px is set
        # during mask dilation; fall back to the user arg when unavailable.
        requested_feather = max(1, int(args.blend_feather_border_px))
        if hasattr(args, '_depth_aware_border_px') and args._depth_aware_border_px is not None:
            feather_border_px = min(requested_feather, int(args._depth_aware_border_px))
            feather_border_px = max(1, feather_border_px)
        else:
            feather_border_px = requested_feather
        ek = 2 * feather_border_px + 1
        erode_kernel = np.ones((ek, ek), np.uint8)
        hard_bin = cv2.erode(blend_bin, erode_kernel, iterations=1)

        border_bin = np.logical_and(blend_bin == 1, hard_bin == 0).astype(np.uint8)

        alpha = np.zeros_like(blend_bin, dtype=np.float32)
        alpha[hard_bin == 1] = 1.0

        if int(args.disable_blend_feather) == 1:
            alpha[blend_bin == 1] = 1.0
        else:
            border_count = int(border_bin.sum())
            if border_count > 0:
                dist_from_outside = cv2.distanceTransform(blend_bin, cv2.DIST_L2, 3)
                max_border_dist = float(feather_border_px)
                border_alpha = np.clip(dist_from_outside / max(1.0, max_border_dist), 0.0, 1.0)
                alpha[border_bin == 1] = border_alpha[border_bin == 1]

            blur_sigma = max(0.0, float(args.blend_alpha_blur_sigma))
            if blur_sigma > 0:
                alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=blur_sigma)
            alpha[hard_bin == 1] = 1.0
        alpha[blend_bin == 0] = 0.0

        alpha_mask = Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L")

        # Composite: mask outside = always bg; dilation border ring = feathered blend;
        # object core (hard_bin) = fully from generated result, no bg bleed-through.
        def _make_final(result_img_crop_orig):
            composite = clean_bg.copy()
            patch_result = result_img_crop_orig.crop((ox, oy, ox + valid_w, oy + valid_h))
            patch_bg = composite.crop((ix1, iy1, ix1 + valid_w, iy1 + valid_h))
            blended = Image.composite(patch_result, patch_bg, alpha_mask)
            composite.paste(blended, (ix1, iy1))
            return composite

        final_image = _make_final(result_crop_orig)

        save_path = f"{final_output_dir}/final_result.png"

        # ------------------------------------------------------------------
        # P0: Post-correction Gemini re-score + rollback.
        # If feedback_json_path provided and the call succeeds:
        #   - score_before: original size_score from the feedback JSON.
        #   - score_after:  Gemini judge on the corrected image.
        # If score_after moves *further from 3* (or stays same), rollback to
        # original image.  Always write correction_meta.json with audit trail.
        # ------------------------------------------------------------------
        rescore_enabled = (
            int(getattr(args, "enable_rescore", 1)) == 1
            and bool(getattr(args, "feedback_json_path", ""))
            and bool(getattr(args, "rescore_object_a", ""))
            and bool(getattr(args, "rescore_object_b", ""))
            and bool(getattr(args, "rescore_len_a_cm", None) is not None)
            and bool(getattr(args, "rescore_len_b_cm", None) is not None)
        )
        score_before = None
        score_after = None
        did_rollback = False
        if rescore_enabled:
            # Temporarily save corrected image so Gemini can read it.
            _tmp_path = f"{final_output_dir}/_tmp_corrected_for_rescore.png"
            final_image.save(_tmp_path)
            print("   [rescore] Calling Gemini judge on corrected image...", flush=True)
            score_after = _gemini_rescore_pair(
                _tmp_path,
                str(args.rescore_object_a),
                float(args.rescore_len_a_cm),
                str(args.rescore_object_b),
                float(args.rescore_len_b_cm),
                api_key=str(args.google_api_key or os.environ.get("GOOGLE_API_KEY", "")),
                model_version=str(getattr(args, "gemini_model", "gemini-3.1-pro-preview")),
                scenario=str(getattr(args, "rescore_scenario", "")),
            )
            try:
                os.remove(_tmp_path)
            except Exception:
                pass

            # Parse score_before from feedback JSON.
            try:
                _fb_path = str(args.feedback_json_path)
                _task_id = str(
                    getattr(args, "feedback_lookup_task_id", "")
                    or os.path.splitext(os.path.basename(args.image_path))[0]
                )
                with open(_fb_path, "r", encoding="utf-8") as _f:
                    _fb_data = json.load(_f)
                for _it in (_fb_data if isinstance(_fb_data, list) else []):
                    if str(_it.get("task_id", "")) == _task_id:
                        for _p in (_it.get("pairs", []) or []):
                            _a = str((_p.get("object_a") or {}).get("name", "")).lower()
                            _b = str((_p.get("object_b") or {}).get("name", "")).lower()
                            _ta = str(args.rescore_object_a).lower()
                            _tb = str(args.rescore_object_b).lower()
                            if ((_a == _ta and _b == _tb) or (_a == _tb and _b == _ta)):
                                score_before = _p.get("size_score")
                                break
                        break
            except Exception:
                pass

            if score_after is not None and score_before is not None:
                dist_before = abs(int(score_before) - 3)
                dist_after = abs(int(score_after) - 3)
                if dist_after >= dist_before:
                    # Correction did not improve (or worsened) → rollback to original.
                    print(
                        f"   [rescore] score {score_before}→{score_after} "
                        f"(dist {dist_before}→{dist_after}): ROLLBACK to original.",
                        flush=True,
                    )
                    final_image = orig_img.copy()
                    did_rollback = True
                else:
                    print(
                        f"   [rescore] score {score_before}→{score_after} "
                        f"(dist {dist_before}→{dist_after}): correction accepted.",
                        flush=True,
                    )
            elif score_after is None:
                print("   [rescore] re-score failed (API error); keeping corrected image.", flush=True)

        final_image.save(save_path)
        print(f"🎉 Done! Saved to: {save_path}")

        # ------------------------------------------------------------------
        # Write correction_meta.json: full pipeline audit for downstream
        # comparison experiments (other inpainting models, ablation, etc.)
        # Fields mirror the modal script's correction_meta schema.
        # ------------------------------------------------------------------
        _meta_path = f"{final_output_dir}/correction_meta.json"
        correction_meta = {
            "schema_version": 1,
            "pipeline": "local_size_correction",
            "status": "ok",
            "image_path": str(args.image_path),
            "image_size_wh": [int(W1), int(H1)],
            "target_object": str(args.target_object),
            "ref_object": str(getattr(args, "ref_object", "") or ""),
            "plan": {
                "original_bbox": [int(x) for x in orig_box_px],
                "scale_factor": float(scale_factor),
                "anchor_point": str(anchor),
            },
            "feedback_planner": {
                "use_feedback_planner": int(getattr(args, "use_feedback_planner", 0) or 0),
                "feedback_json_path": str(getattr(args, "feedback_json_path", "") or ""),
                "hint": str(plan.get("_feedback_hint", "") or "") if isinstance(plan, dict) else "",
                "feedback_blob": plan.get("_feedback_blob") if isinstance(plan, dict) else None,
            },
            "bbox_object_pixels_yxyx": [int(x) for x in orig_box_px],
            "bbox_target_scaled_pixels_yxyx": [int(x) for x in new_box_px],
            "scale_factor_effective": float(scale_factor),
            "effective_min_scale": float(effective_min_scale),
            "effective_max_scale": float(effective_max_scale),
            "anchor_point": str(anchor),
            # ref_crop_roi: bounding box of the ref crop in original image coords [y1,x1,y2,x2]
            "ref_crop_roi_pixels_yxyx": [int(cy1), int(cx1), int(cy2), int(cx2)],
            "gen_crop_square_pixels_yxyx": [int(x) for x in crop_yx],
            "gen_resolution": 768,
            "use_depth_for_generation": bool(use_depth_for_gen),
            # Depth range passed to adaptive_depth_fusion (p5–p95 on original-scene
            # object before removal, unless catalog ref overwrites with ref crop).
            "target_depth_range_percentile_5_95": (
                [float(target_depth_range[0]), float(target_depth_range[1])]
                if target_depth_range is not None
                else None
            ),
            "output_artifacts": {
                "final_result": "final_result.png",
                "ref_clean": "ref_clean.png",
                "ref_mask": "ref_mask.png",
                "bg_clean": "bg_clean.png",
                "mask_fused": "mask_fused.png",
                "depth_fused": "depth_fused.png",
                "debug_crop_input": "debug_crop_input.png",
                "debug_crop_mask": "debug_crop_mask.png",
            },
        }
        if rescore_enabled:
            correction_meta["rescore"] = {
                "gemini_score_before": score_before,
                "gemini_score_after": score_after,
                "did_rollback": did_rollback,
                "object_a": str(getattr(args, "rescore_object_a", "")),
                "object_b": str(getattr(args, "rescore_object_b", "")),
                "len_a_cm": float(getattr(args, "rescore_len_a_cm", 0) or 0),
                "len_b_cm": float(getattr(args, "rescore_len_b_cm", 0) or 0),
            }
        try:
            with open(_meta_path, "w", encoding="utf-8") as _mf:
                json.dump(correction_meta, _mf, indent=2)
            print(f"  Saved: {_meta_path}", flush=True)
        except Exception as _e:
            print(f"  [warn] Failed to write correction_meta.json: {_e}", flush=True)

        # Optional: also save no-depth final for side-by-side comparison.
        if result_crop_nodepth is not None and not did_rollback:
            result_crop_nodepth_orig = result_crop_nodepth.resize(bg_crop.size, Image.LANCZOS)
            final_image_nodepth = _make_final(result_crop_nodepth_orig)

            save_path_nodepth = f"{final_output_dir}/final_result_nodepth.png"
            final_image_nodepth.save(save_path_nodepth)
            print(f"🎉 Done! Saved to: {save_path_nodepth}")

    else:
        print("⚠️ Crop completely out of bounds, skipping paste.")

    if int(args.cache_models_cpu) == 1:
        try:
            inferencer.to_device("cpu")
            torch.cuda.empty_cache()
        except Exception:
            pass

    return final_output_dir


def run_with_args(arg_list):
    old_argv = list(sys.argv)
    try:
        sys.argv = [old_argv[0]] + list(arg_list)
        return main()
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    main()
