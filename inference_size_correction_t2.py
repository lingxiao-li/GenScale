"""
Task 2 size correction inference script.

Task 2 images depict product-human interactions. The human body part is ALWAYS
the anchor (immutable reference). Only the product needs correction (single round).

Requires: GOOGLE_API_KEY environment variable for Gemini API.

Usage:
    # Single test image (output under results/size_correction_T2, same layout as T1):
    python inference_size_correction_t2.py --image_path path/to/T2_0000.png

    # Same without benchmark JSON (provide objects explicitly):
    python inference_size_correction_t2.py --image_path test.png --product "..." --human_part "human hand"

    # Batch (model folder):
    python inference_size_correction_t2.py --image_dir path/to/FLUX_1_Kontext_dev_Edit --max_images 2

    # All images in folder (max_images=0 means all):
    python inference_size_correction_t2.py --image_dir path/to/model_folder --max_images 0

    # Task2 default: ref is extracted from the scene image (handles viewpoint consistency).
    # Depth ControlNet is off by default (--use_depth_guidance 0): fusion may still run for logging,
    # but generation is mask-only. Mask dilation defaults are mild (tight inpaint region).
    # Inpaint mask regularize (--mask_inpaint_regularize close by default) smooths the SAM mask;
    # avoid close_convex_hull on long thin objects (e.g. knives) — convex hull fills the whole
    # span between endpoints and looks like a full-crop mask, harming the anchor (hand).
    python inference_size_correction_t2.py --image_path ...

Re-runs: if the session folder already exists, a new run uses the same base name with _1, _2, ...
(aligned with T1 multi-round naming), instead of skipping.
"""

import argparse
import glob
import importlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import shared components from multi_object_inference
from multi_object_inference import (
    GeminiClient,
    compute_scale_exec,
    normalize_bbox_to_pixels,
    pick_object_by_expected,
    sanitize_anchor_point,
)

# Default paths (script lives in insert-anything/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
SUBMISSION_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK = SUBMISSION_ROOT / "GenScale_Benchmark_v3_final_anonymous.json"
DEFAULT_IMAGE_DIR = SUBMISSION_ROOT / "images" / "task2" / "input_images"
DEFAULT_OUTPUT_DIR = SUBMISSION_ROOT / "outputs" / "size_correction_T2"
_SC_MODULE = None

# Same direct scale tiers used by Task1 v2 / Gemini judge.
_TIER_DIRECT_B = {
    1: (1.6, 3.0),
    2: (1.2, 1.6),
    4: (0.625, 0.84),
    5: (0.33, 0.625),
}


def run_single_correction_cached(script_path: str, image_path: str, output_dir: str, target: str, anchor: str, extra_args: Dict):
    """
    In-process single correction call to avoid per-image subprocess startup and
    allow model cache reuse in inference_size_correction.py.
    """
    global _SC_MODULE
    if _SC_MODULE is None:
        if Path(script_path).resolve() == (PROJECT_ROOT / "inference_size_correction.py").resolve():
            _SC_MODULE = importlib.import_module("inference_size_correction")
        else:
            spec = importlib.util.spec_from_file_location("inference_size_correction_custom", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import single inference script: {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _SC_MODULE = module

    arg_list = [
        "--image_path", image_path,
        "--output_dir", output_dir,
        "--target_object", target,
        "--ref_object", anchor,
    ]
    for k, v in extra_args.items():
        if v is None:
            continue
        arg_list.extend([f"--{k}", str(v)])

    if not hasattr(_SC_MODULE, "run_with_args"):
        raise RuntimeError("inference_size_correction.py must expose run_with_args(arg_list)")
    _SC_MODULE.run_with_args(arg_list)

    session_dir = os.path.join(output_dir, f"{target}_vs_{anchor}".replace("/", "_").replace("\\", "_"))
    direct = os.path.join(session_dir, "final_result.png")
    if os.path.exists(direct):
        return direct
    cands = glob.glob(os.path.join(output_dir, "**", "final_result.png"), recursive=True)
    if not cands:
        raise RuntimeError(f"No final_result.png found under {output_dir}")
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def load_t2_entries(benchmark_json: str) -> Dict[str, dict]:
    """Load benchmark and return {task_id: entry} for T2 only."""
    if not benchmark_json or not os.path.isfile(benchmark_json):
        return {}
    with open(benchmark_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {e["task_id"]: e for e in data if e.get("task_id", "").startswith("T2_")}


def simplify_product_name_for_detection(name: str) -> str:
    """
    Short label for detectors / Gemini (DINO + Gemini) from benchmark ``objects_included``.

    v3 merged JSON uses short phrases (e.g. ``shampoo bottle``). Older JSON may use long
    Amazon listing titles; those hurt detection — strip listing noise and cap length.
    """
    s = (name or "").strip()
    if not s:
        return s
    low = s.lower()
    if len(s) <= 52 and s.count(",") <= 1 and len(s.split()) <= 8:
        return s
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    if "," in s:
        s = s.split(",")[0].strip()
    if " - " in s:
        parts = [p.strip() for p in s.split(" - ") if p.strip()]
        if len(parts) >= 2:
            tail = parts[-1]
            if len(tail) <= 60 and len(tail.split()) <= 10:
                s = tail
            else:
                s = parts[0]
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if len(words) > 8:
        s = " ".join(words[:8])
    return s[:80].rstrip()


def extract_benchmark_product_size_hint(entry: Optional[dict]) -> Optional[str]:
    """
    Pull a size / dimension line from the benchmark ``prompt`` (T2 v3 includes cm sizes).

    Passed to Gemini scale planning as a weak prior for ``desired_ratio``.
    """
    if not entry:
        return None
    prompt = (entry.get("prompt") or "").strip()
    if not prompt:
        return None
    m = re.search(
        r"(?:measuring|approximately|about|exactly)\s+.{20,450}?\.(?:\s|$)",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(0).strip()[:520]
    m2 = re.search(r"[^.]{10,450}?\d[^.]*(?:cm|centimeter)[^.]{0,220}\.", prompt, re.IGNORECASE)
    if m2:
        return m2.group(0).strip()[:520]
    return None


def extract_product_and_human(entry: dict) -> Tuple[Optional[str], Optional[str]]:
    """From T2 entry objects_included, return (product, human_body_part) with short product label."""
    objs = entry.get("objects_included", [])
    product, human_part = None, None
    for obj in objs:
        o = obj if isinstance(obj, str) else str(obj or "")
        if "human" in o.lower():
            human_part = o.strip()
        else:
            product = o.strip()
    if product:
        product = simplify_product_name_for_detection(product)
    return product, human_part


# Shorter than T1 multi-round [:120] — keeps paths readable on disk.
_SESSION_NAME_MAX_LEN = 80
_PRODUCT_LABEL_MAX = 36
_ANCHOR_LABEL_MAX = 24


def _compact_session_label(s: str, max_chars: int) -> str:
    """Short sanitized token for folder names."""
    s = (s or "").strip().replace(" ", "-")
    s = re.sub(r"[^\w\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("-_")
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip("-_")


def t2_session_dir_name(img_stem: str, product: str, human_part: str) -> str:
    """{stem}__{short_product}_vs_{short_anchor}, sanitized (compact middle segment)."""
    tgt = _compact_session_label(product, _PRODUCT_LABEL_MAX)
    anc = _compact_session_label(human_part, _ANCHOR_LABEL_MAX)
    name = f"{img_stem}__{tgt}_vs_{anc}"
    name = re.sub(r"[^\w\-]", "_", name)[:_SESSION_NAME_MAX_LEN]
    return name


def allocate_unique_session_subdir_name(output_root: str, session_name: str) -> str:
    """
    Match T1 multi-round: use output_root/session_name if missing;
    else session_name_1, session_name_2, ... (never skip a run because folder exists).
    """
    candidate = os.path.join(output_root, session_name)
    if not os.path.exists(candidate):
        return session_name
    idx = 1
    while os.path.exists(os.path.join(output_root, f"{session_name}_{idx}")):
        idx += 1
    return f"{session_name}_{idx}"


def _layout_name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _names_match(a: str, b: str) -> bool:
    ka = _layout_name_key(a)
    kb = _layout_name_key(b)
    return bool(ka and kb and (ka == kb or ka in kb or kb in ka))


def _clamp_bbox_yxyx(bbox: List[int], width: int, height: int) -> List[int]:
    y1, x1, y2, x2 = [int(round(float(v))) for v in bbox]
    y1 = max(0, min(height - 1, y1))
    x1 = max(0, min(width - 1, x1))
    y2 = max(y1 + 1, min(height, y2))
    x2 = max(x1 + 1, min(width, x2))
    return [y1, x1, y2, x2]


def _det_bbox_pixels(obj: Optional[Dict], width: int, height: int) -> Optional[List[int]]:
    if not obj or not obj.get("bbox"):
        return None
    try:
        return _clamp_bbox_yxyx(normalize_bbox_to_pixels(obj["bbox"], width, height), width, height)
    except Exception:
        return None


def _layout_update_object_bbox(layout_doc: dict, object_name: str, bbox_yxyx) -> bool:
    if not isinstance(layout_doc, dict) or bbox_yxyx is None or len(bbox_yxyx) != 4:
        return False
    key = _layout_name_key(object_name)
    for obj in layout_doc.get("objects") or []:
        if _layout_name_key(str(obj.get("name", ""))) == key:
            obj["bbox_yxyx"] = [int(x) for x in bbox_yxyx]
            return True
    return False


def _save_layout_bbox_overlay(layout_doc: dict, image_path: str, out_png: str) -> None:
    try:
        from PIL import Image as PILImage, ImageDraw

        img = PILImage.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = {
            "target": (255, 64, 64),
            "anchor": (64, 180, 255),
        }
        for obj in layout_doc.get("objects") or []:
            bb = obj.get("bbox_yxyx")
            if not isinstance(bb, list) or len(bb) != 4:
                continue
            y1, x1, y2, x2 = [int(v) for v in bb]
            role = str(obj.get("role", "target"))
            color = colors.get(role, (255, 220, 64))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label = f"{role}: {obj.get('name', '')}"
            draw.text((x1 + 4, max(0, y1 - 16)), label, fill=color)
        img.save(out_png)
        print(f"   [Layout] Saved layout overlay: {out_png}", flush=True)
    except Exception as exc:
        print(f"   [Layout] Warning: failed to save overlay {out_png}: {exc}", flush=True)


def _save_layout_json_and_vis(layout_doc: dict, image_path: str, json_path: str, vis_path: str) -> None:
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(layout_doc, f, indent=2, ensure_ascii=False)
        print(f"   [Layout] Saved layout JSON: {json_path}", flush=True)
    except Exception as exc:
        print(f"   [Layout] Warning: failed to save layout JSON {json_path}: {exc}", flush=True)
    _save_layout_bbox_overlay(layout_doc, image_path, vis_path)


def _load_feedback_entry(feedback_json_path: str, task_id: str) -> Optional[dict]:
    fp = str(feedback_json_path or "").strip()
    if not fp or not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data if isinstance(data, list) else []:
            if str(item.get("task_id", "")) == str(task_id):
                return item
    except Exception as exc:
        print(f"   [feedback] failed to load {fp}: {exc}", flush=True)
    return None


def _tier_gate_lo_hi_for_product(size_score: int, product_is_object_b: bool) -> Tuple[Optional[float], Optional[float]]:
    if size_score not in _TIER_DIRECT_B:
        return None, None
    lo_b, hi_b = _TIER_DIRECT_B[size_score]
    if product_is_object_b:
        return float(lo_b), float(hi_b)
    inv_lo = 1.0 / float(hi_b)
    inv_hi = 1.0 / float(lo_b)
    if inv_lo > inv_hi:
        inv_lo, inv_hi = inv_hi, inv_lo
    return float(inv_lo), float(inv_hi)


def _select_t2_feedback_pair(
    feedback_entry: Optional[dict],
    product: str,
    human_part: str,
) -> Tuple[Optional[dict], bool, bool]:
    """
    Return (pair, product_is_object_b, name_matched).
    Task2 evaluator normally stores human as object_a and product as object_b.
    """
    pairs = (feedback_entry or {}).get("pairs") or []
    best_pair = None
    product_is_b = True
    name_matched = False

    for pair in pairs:
        a = ((pair.get("object_a") or {}).get("name") or "")
        b = ((pair.get("object_b") or {}).get("name") or "")
        if _names_match(a, human_part) and _names_match(b, product):
            return pair, True, True
        if _names_match(a, product) and _names_match(b, human_part):
            return pair, False, True

    for pair in pairs:
        a = ((pair.get("object_a") or {}).get("name") or "")
        b = ((pair.get("object_b") or {}).get("name") or "")
        if _names_match(b, product):
            best_pair, product_is_b = pair, True
            break
        if _names_match(a, product):
            best_pair, product_is_b = pair, False
            break

    if best_pair is None and pairs:
        best_pair = pairs[0]
        a = ((best_pair.get("object_a") or {}).get("name") or "")
        b = ((best_pair.get("object_b") or {}).get("name") or "")
        product_is_b = _names_match(b, product) or not _names_match(a, product)
    return best_pair, product_is_b, name_matched


def _feedback_hint_from_score(score: Any, product_is_object_b: bool) -> str:
    try:
        s = int(score)
    except Exception:
        return "unknown"
    if s in (1, 2):
        hint = "enlarge"
    elif s in (4, 5):
        hint = "shrink"
    else:
        hint = "unknown"
    if not product_is_object_b:
        if hint == "enlarge":
            return "shrink"
        if hint == "shrink":
            return "enlarge"
    return hint


def _build_feedback_blob(
    feedback_entry: Optional[dict],
    product: str,
    human_part: str,
) -> Tuple[Optional[dict], str, bool]:
    pair, product_is_b, name_matched = _select_t2_feedback_pair(feedback_entry, product, human_part)
    if not pair:
        return None, "unknown", False
    score = pair.get("size_score")
    hint = _feedback_hint_from_score(score, product_is_b)
    try:
        score_i = int(score)
    except Exception:
        score_i = None
    lo, hi = _tier_gate_lo_hi_for_product(score_i, product_is_b) if score_i is not None else (None, None)
    blob = {
        "task_id": str((feedback_entry or {}).get("task_id", "")),
        "size_score": score,
        "both_objects_present": pair.get("both_objects_present"),
        "object_a": pair.get("object_a"),
        "object_b": pair.get("object_b"),
        "reasoning_detection": pair.get("reasoning_detection"),
        "reasoning_scale_and_interaction": pair.get("reasoning_scale_and_interaction")
        or pair.get("reasoning_depth_and_perspective"),
        "all_size_scores": pair.get("all_size_scores"),
        "size_score_definition": (
            "Task2: object_a is the human anchor; object_b is the target product. "
            "1/2 means product too small, 4/5 means product too large, 3 means plausible."
        ),
        "direction_hint": hint,
        "product_is_object_b": bool(product_is_b),
        "name_matched_pair": bool(name_matched),
        "size_score_tier_gate": {
            "lo": lo,
            "hi": hi,
            "note": "Clamp product scale_factor to this interval before the single edit.",
        },
    }
    skip = bool(name_matched and score_i == 3 and pair.get("both_objects_present", True))
    if skip:
        blob["skip_correction"] = True
        blob["skip_reason"] = "task2_feedback_size_score_3_name_matched"
    return blob, hint, skip


def _apply_t2_feedback_and_residual_scale(raw_scale: float, feedback_blob: Optional[dict], args) -> Tuple[float, dict]:
    scale = float(raw_scale) if math.isfinite(float(raw_scale)) else 1.0
    notes: List[str] = []
    residual = {
        "enabled": bool(int(getattr(args, "t2_use_residual_correction", 1)) == 1),
        "scale_before": scale,
        "log_nudge": 0.0,
        "scale_after": scale,
        "notes": notes,
    }

    hint = str((feedback_blob or {}).get("direction_hint") or "unknown")
    if hint == "shrink" and scale >= 1.0:
        scale = 0.999
        notes.append("direction_gate:shrink")
    elif hint == "enlarge" and scale <= 1.0:
        scale = 1.001
        notes.append("direction_gate:enlarge")

    if feedback_blob and residual["enabled"] and hint in ("shrink", "enlarge"):
        try:
            score_i = int(feedback_blob.get("size_score"))
        except Exception:
            score_i = 3
        severity = 2.0 if score_i in (1, 5) else 1.0
        sign = -1.0 if hint == "shrink" else 1.0
        raw_nudge = sign * float(getattr(args, "t2_residual_log_step", 0.10)) * severity
        max_step = abs(float(getattr(args, "t2_residual_max_log_step", 0.25)))
        log_nudge = max(-max_step, min(max_step, raw_nudge))
        scale *= math.exp(log_nudge)
        residual["log_nudge"] = float(log_nudge)
        notes.append(f"residual_log_nudge:{raw_nudge:.3f}->{log_nudge:.3f}")

    gate = (feedback_blob or {}).get("size_score_tier_gate") or {}
    lo, hi = gate.get("lo"), gate.get("hi")
    try:
        lo_f = float(lo) if lo is not None else None
        hi_f = float(hi) if hi is not None else None
    except Exception:
        lo_f, hi_f = None, None
    if lo_f is not None and hi_f is not None:
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        before = scale
        scale = max(lo_f, min(hi_f, scale))
        if abs(scale - before) > 1e-6:
            notes.append(f"tier_gate:{before:.3f}->{scale:.3f}")

    scale = max(float(args.min_scale), min(float(args.max_scale), scale))
    residual["scale_after"] = float(scale)
    return float(scale), residual


def estimate_t2_product_scale_plan(
    gemini: GeminiClient,
    image_path: str,
    product: str,
    human_part: str,
    product_bbox_px: Optional[List[int]],
    human_bbox_px: Optional[List[int]],
    product_size_hint: Optional[str],
    feedback_blob: Optional[dict],
    args,
) -> Tuple[dict, float]:
    size_block = ""
    if product_size_hint and str(product_size_hint).strip():
        size_block = f"Benchmark product-size hint:\n{str(product_size_hint).strip()}\n"
    feedback_block = ""
    if feedback_blob:
        feedback_block = (
            "Evaluator feedback for this exact Task2 pair (use as a strong prior):\n"
            f"{json.dumps(feedback_blob, ensure_ascii=False)}\n"
        )

    prompt = f"""
You are planning a conservative Task2 product-size correction.

Task2 scene:
- HUMAN ANCHOR (must stay unchanged): '{human_part}'
- TARGET PRODUCT (the only thing allowed to resize): '{product}'

Known layout:
- product bbox [ymin,xmin,ymax,xmax] in pixels: {product_bbox_px if product_bbox_px else "unknown"}
- human anchor bbox [ymin,xmin,ymax,xmax] in pixels: {human_bbox_px if human_bbox_px else "unknown"}

{size_block}{feedback_block}
Rules:
1) Judge ONLY whether the product's real-world physical size looks plausible in this human interaction.
2) Do NOT propose changing the human body part. Treat the human anchor as fixed and correctly scaled.
3) Do NOT optimize generic relative object proportions; this is not a multi-object Task1 scene.
4) Product-human interactions often have occlusion, cropped hands/faces/feet, foreshortening, and product-forward catalog framing. Be conservative.
5) Use LINEAR characteristic length scale. scale_factor < 1 shrinks product; > 1 enlarges product.
6) Choose anchor_point to preserve contact with the human part: CENTER for held/grasped products unless a clear contact edge suggests TOP/BOTTOM/LEFT/RIGHT_CENTER.

Return ONLY valid JSON:
{{
  "need_correction": true,
  "current_ratio": 0.30,
  "desired_ratio": 0.25,
  "ratio_type": "linear",
  "scale_factor": 0.83,
  "anchor_point": "CENTER",
  "confidence": 0.0,
  "reason": "short product-only rationale"
}}
"""
    try:
        plan = gemini._call(prompt, image_path)
    except Exception as exc:
        print(f"   Gemini T2 product-only planning failed: {exc}", flush=True)
        plan = {}

    normalized = {
        "need_correction": bool(plan.get("need_correction", True)),
        "current_ratio": float(plan.get("current_ratio", 0.0) or 0.0),
        "desired_ratio": float(plan.get("desired_ratio", 0.0) or 0.0),
        "ratio_type": str(plan.get("ratio_type", "linear") or "linear"),
        "scale_factor": float(plan.get("scale_factor", 1.0) or 1.0),
        "anchor_point": sanitize_anchor_point(str(plan.get("anchor_point", "CENTER") or "CENTER")),
        "confidence": float(plan.get("confidence", 0.0) or 0.0),
        "reason": str(plan.get("reason", "")),
        "source": "task2_product_only_gemini",
        "feedback_blob": feedback_blob,
    }
    raw_scale = compute_scale_exec(normalized, args.min_scale, args.max_scale)
    scale, residual = _apply_t2_feedback_and_residual_scale(raw_scale, feedback_blob, args)
    normalized["scale_factor_raw"] = float(raw_scale)
    normalized["scale_factor"] = float(scale)
    normalized["residual_correction"] = residual
    return normalized, float(scale)


def _read_correction_meta(round_final_path: str) -> Optional[dict]:
    cm_path = os.path.join(os.path.dirname(round_final_path), "correction_meta.json")
    if os.path.isfile(cm_path):
        try:
            with open(cm_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def human_body_part_dino_queries(human_part: str) -> List[str]:
    """Return DINO-friendly query variants for human body parts."""
    raw = (human_part or "").strip().lower()
    if "face" in raw or "head" in raw:
        return ["human face", "human head", "human face/head", raw]
    if "foot" in raw or "leg" in raw:
        return ["human foot", "human leg", "human foot/leg", raw]
    if "hand" in raw:
        return ["human hand", raw]
    return [raw, "human"]


class DINOClient:
    """
    GroundingDINO detector via transformers.
    Used for fast detection; Gemini fallback when DINO misses.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        fallback_box_threshold: float = 0.15,
        fallback_text_threshold: float = 0.10,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.fallback_box_threshold = float(fallback_box_threshold)
        self.fallback_text_threshold = float(fallback_text_threshold)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _product_variants(obj_name: str) -> List[str]:
        """Simplified variants for product names (often long)."""
        raw = (obj_name or "").strip()
        # Take first meaningful part (e.g., "Amazon Brand - Solimo Sonic..." -> "Solimo Sonic")
        if " - " in raw:
            parts = raw.split(" - ", 1)
            if len(parts) > 1 and len(parts[1]) > 3:
                raw = parts[1]
        simple = re.sub(r"\s*\(.*?\)\s*", "", raw).strip()
        out = [raw[:80] if len(raw) > 80 else raw, simple[:80] if len(simple) > 80 else simple]
        return [x for x in dict.fromkeys(out) if x]

    def _detect_single(
        self,
        image_pil,
        object_name: str,
        is_human_part: bool = False,
        human_queries: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        import inspect

        W, H = image_pil.size
        queries = human_queries if is_human_part and human_queries else self._product_variants(object_name)
        trials = [
            (self.box_threshold, self.text_threshold),
            (self.fallback_box_threshold, self.fallback_text_threshold),
        ]
        for query in queries:
            text_query = (query or object_name).lower().strip()
            if not text_query:
                continue
            for b_thr, t_thr in trials:
                inputs = self.processor(images=image_pil, text=text_query, return_tensors="pt")
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                pp = self.processor.post_process_grounded_object_detection
                params = inspect.signature(pp).parameters
                pp_kwargs = {"target_sizes": [(H, W)]}
                if "box_threshold" in params:
                    pp_kwargs["box_threshold"] = b_thr
                elif "threshold" in params:
                    pp_kwargs["threshold"] = b_thr
                if "text_threshold" in params:
                    pp_kwargs["text_threshold"] = t_thr

                results = pp(outputs, inputs["input_ids"], **pp_kwargs)[0]
                if len(results["boxes"]) == 0:
                    continue
                scores = results["scores"].detach().cpu().tolist()
                boxes = results["boxes"].detach().cpu().tolist()
                best_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
                x1, y1, x2, y2 = boxes[best_idx]
                x1 = int(max(0, min(W - 1, x1)))
                y1 = int(max(0, min(H - 1, y1)))
                x2 = int(max(0, min(W - 1, x2)))
                y2 = int(max(0, min(H - 1, y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                return {
                    "name": object_name,
                    "bbox": [y1, x1, y2, x2],
                    "confidence": float(scores[best_idx]),
                    "source": "dino",
                }
        return None

    def detect_product_and_human(
        self,
        image_path: str,
        product: str,
        human_part: str,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Detect both product and human body part. Returns (product_det, human_det)."""
        from PIL import Image as PILImage

        image = PILImage.open(image_path).convert("RGB")
        human_queries = human_body_part_dino_queries(human_part)
        product_det = self._detect_single(image, product, is_human_part=False)
        human_det = self._detect_single(image, human_part, is_human_part=True, human_queries=human_queries)
        return product_det, human_det


def collect_image_paths(image_path: str, image_dir: str, max_images: int) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths: List[str] = []
    if image_dir:
        for p in sorted(Path(image_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))
    elif image_path:
        paths = [image_path]
    else:
        raise ValueError("Provide either --image_path or --image_dir")
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def run_t2_for_image(
    gemini: GeminiClient,
    dino: Optional[DINOClient],
    image_path: str,
    case_output_dir: str,
    product: str,
    human_part: str,
    args,
    product_size_hint: Optional[str] = None,
) -> str:
    """
    Run single-round T2 correction: product vs human body part (anchor).
    """
    from shutil import copyfile

    os.makedirs(case_output_dir, exist_ok=True)
    final_path = os.path.join(case_output_dir, "final_result.png")
    final_path_legacy = os.path.join(case_output_dir, "final_t2_result.png")
    history_path = os.path.join(case_output_dir, "multi_object_history.json")

    from PIL import Image as PILImage

    W, H = PILImage.open(image_path).size
    layout_objects: List[dict] = []
    layout_original = {
        "task_type": "T2",
        "image_path": os.path.abspath(image_path),
        "image_size": [int(W), int(H)],
        "anchor_object": human_part,
        "target_object": product,
        "objects": layout_objects,
    }
    layout_final = json.loads(json.dumps(layout_original))

    # Detection: DINO first (if available), Gemini fallback for missing or Gemini-only
    target_obj, anchor_obj = None, None
    if dino is not None:
        target_obj, anchor_obj = dino.detect_product_and_human(image_path, product, human_part)
        if target_obj is not None:
            target_obj["name"] = product
        if anchor_obj is not None:
            anchor_obj["name"] = human_part

    missing = []
    if target_obj is None:
        missing.append(product)
    if anchor_obj is None:
        missing.append(human_part)

    if missing:
        if dino is not None:
            print(f"   DINO missed {missing}, fallback to Gemini ...", flush=True)
        else:
            print(f"   Using Gemini for detection ...", flush=True)
        try:
            gemini_det = gemini.detect_objects(image_path, [product, human_part], max_objects=2)
            if target_obj is None:
                target_obj = pick_object_by_expected(gemini_det, product)
                if target_obj:
                    target_obj["name"] = product
            if anchor_obj is None:
                anchor_obj = pick_object_by_expected(gemini_det, human_part)
                if anchor_obj:
                    anchor_obj["name"] = human_part
        except Exception as e:
            print(f"   Gemini fallback failed: {e}", flush=True)

    product_bbox_px = _det_bbox_pixels(target_obj, W, H)
    human_bbox_px = _det_bbox_pixels(anchor_obj, W, H)
    layout_objects[:] = []
    if product_bbox_px is not None:
        layout_objects.append({"name": product, "role": "target", "bbox_yxyx": [int(x) for x in product_bbox_px]})
    if human_bbox_px is not None:
        layout_objects.append({"name": human_part, "role": "anchor", "bbox_yxyx": [int(x) for x in human_bbox_px]})
    layout_final = json.loads(json.dumps(layout_original))

    if target_obj is None or anchor_obj is None:
        print(f"   Cannot detect both product and human. Skip correction, save original.", flush=True)
        _save_layout_json_and_vis(
            layout_original,
            image_path,
            os.path.join(case_output_dir, "layout_original.json"),
            os.path.join(case_output_dir, "layout_original_vis.png"),
        )
        layout_final["image_path"] = os.path.abspath(final_path)
        copyfile(image_path, final_path)
        if final_path != final_path_legacy:
            copyfile(image_path, final_path_legacy)
        _save_layout_json_and_vis(
            layout_final,
            final_path,
            os.path.join(case_output_dir, "layout_final.json"),
            os.path.join(case_output_dir, "layout_final_vis.png"),
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_type": "T2",
                    "multi_object_history": [{
                        "round": 0,
                        "target": product,
                        "anchor": human_part,
                        "output": os.path.abspath(final_path),
                        "skipped": True,
                        "skip_reason": "detection_failed",
                    }],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return final_path

    # Refine bboxes via Gemini
    try:
        target_obj["bbox"] = gemini.refine_bbox(image_path, product, target_obj["bbox"])
    except Exception as e:
        print(f"   Bbox refine failed for product: {e}", flush=True)
    try:
        anchor_obj["bbox"] = gemini.refine_bbox(image_path, human_part, anchor_obj["bbox"])
    except Exception as e:
        print(f"   Bbox refine failed for human: {e}", flush=True)

    product_bbox_px = _det_bbox_pixels(target_obj, W, H)
    human_bbox_px = _det_bbox_pixels(anchor_obj, W, H)
    layout_original["objects"] = []
    if product_bbox_px is not None:
        layout_original["objects"].append(
            {"name": product, "role": "target", "bbox_yxyx": [int(x) for x in product_bbox_px]}
        )
    if human_bbox_px is not None:
        layout_original["objects"].append(
            {"name": human_part, "role": "anchor", "bbox_yxyx": [int(x) for x in human_bbox_px]}
        )
    layout_final = json.loads(json.dumps(layout_original))
    _save_layout_json_and_vis(
        layout_original,
        image_path,
        os.path.join(case_output_dir, "layout_original.json"),
        os.path.join(case_output_dir, "layout_original_vis.png"),
    )

    feedback_entry = None
    if int(getattr(args, "use_feedback_planner", 1)) == 1:
        feedback_entry = _load_feedback_entry(getattr(args, "feedback_json_path", ""), Path(image_path).stem)
    feedback_blob, feedback_hint, feedback_skip = _build_feedback_blob(feedback_entry, product, human_part)
    if feedback_blob:
        print(
            f"   [feedback] task_id={feedback_blob.get('task_id')} score={feedback_blob.get('size_score')} "
            f"hint={feedback_hint} name_matched={feedback_blob.get('name_matched_pair')}",
            flush=True,
        )

    # Product-only Task2 scale estimation.
    plan, sf = estimate_t2_product_scale_plan(
        gemini,
        image_path,
        product,
        human_part,
        product_bbox_px,
        human_bbox_px,
        product_size_hint,
        feedback_blob,
        args,
    )

    cur_r = float(plan.get("current_ratio", 0.0) or 0.0)
    des_r = float(plan.get("desired_ratio", 0.0) or 0.0)
    ratio_log_err = abs(math.log((cur_r + 1e-6) / (des_r + 1e-6))) if cur_r > 0 and des_r > 0 else 0.0
    need = (
        (not feedback_skip)
        and (
            (ratio_log_err >= args.min_ratio_log_error and abs(sf - 1.0) >= args.min_scale_delta)
            or (bool(plan["need_correction"]) and abs(sf - 1.0) >= args.min_scale_delta)
            or (feedback_hint in ("shrink", "enlarge") and abs(sf - 1.0) >= args.min_scale_delta)
        )
    )

    print(
        f"   Product-only plan: need_gemini={plan['need_correction']} "
        f"cur_r={cur_r:.4f} des_r={des_r:.4f} "
        f"ratio_log_err={ratio_log_err:.3f} scale_exec={sf:.3f}",
        flush=True,
    )

    if not need:
        skip_reason = "feedback_size_score_3" if feedback_skip else "scale_delta_below_threshold"
        print(f"   Skip: {skip_reason}. Save original.", flush=True)
        copyfile(image_path, final_path)
        if final_path != final_path_legacy:
            copyfile(image_path, final_path_legacy)
        layout_final["image_path"] = os.path.abspath(final_path)
        _save_layout_json_and_vis(
            layout_final,
            final_path,
            os.path.join(case_output_dir, "layout_final.json"),
            os.path.join(case_output_dir, "layout_final_vis.png"),
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_type": "T2",
                    "global_plan": plan,
                    "multi_object_history": [{
                        "round": 1,
                        "target": product,
                        "anchor": human_part,
                        "output": os.path.abspath(final_path),
                        "plan": plan,
                        "scale_exec": float(sf),
                        "skipped": True,
                        "skip_reason": skip_reason,
                    }],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return final_path

    # Single-round correction
    step_dir = os.path.join(case_output_dir, "round_1", product.replace("/", "_")[:60])
    os.makedirs(step_dir, exist_ok=True)

    # Depth ControlNet during *generation*: Task2 defaults to mask-only (occlusion makes fusion unreliable).
    # --use_depth_guidance 1 allows depth CN when fusion sets use_depth; --force_no_depth_control overrides if set.
    if getattr(args, "force_no_depth_control", None) is not None:
        _force_no_depth = int(args.force_no_depth_control)
    else:
        _force_no_depth = 1 - int(args.use_depth_guidance)

    extra_args = {
        "weights_dir": args.weights_dir,
        "gpu_gen": args.gpu_gen,
        "gpu_tools": args.gpu_tools,
        "multi_round": 0,
        "_in_round": 1,
        "use_hf_inject": args.use_hf_inject,
        "hf_hp_radius": args.hf_hp_radius,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "min_scale_change": args.min_scale_delta,
        "crop_ratio": args.crop_ratio,
        "mask_dilate_kernel": args.mask_dilate_kernel,
        "mask_dilate_iter": args.mask_dilate_iter,
        "mask_dilate_boost": args.mask_dilate_boost,
        "mask_dilate_iter_boost": args.mask_dilate_iter_boost,
        "mask_dilate_depth_aware": args.mask_dilate_depth_aware,
        "mask_dilate_ratio": args.mask_dilate_ratio,
        "mask_inpaint_regularize": args.mask_inpaint_regularize,
        "mask_regularize_close_ksz": args.mask_regularize_close_ksz,
        "mask_regularize_close_iter": args.mask_regularize_close_iter,
        "mask_crop_fill_holes": args.mask_crop_fill_holes,
        "mask_crop_post_dilate": args.mask_crop_post_dilate,
        "ref_upscale_threshold": args.ref_upscale_threshold,
        "ref_upscale_target": args.ref_upscale_target,
        "blend_feather_border_px": args.blend_feather_border_px,
        "blend_alpha_blur_sigma": args.blend_alpha_blur_sigma,
        "disable_blend_feather": args.disable_blend_feather,
        "bg_removal_mode": args.bg_removal_mode,
        "editing_steps": args.editing_steps,
        "editing_guidance_scale": args.editing_guidance_scale,
        "bg_remove_max_retries": args.bg_remove_max_retries,
        "bg_remove_diff_thresh": args.bg_remove_diff_thresh,
        "bg_remove_changed_ratio_thresh": args.bg_remove_changed_ratio_thresh,
        "bg_remove_pixel_diff_cutoff": args.bg_remove_pixel_diff_cutoff,
        "bg_preserve_diff_mean_max": args.bg_preserve_diff_mean_max,
        "bg_preserve_changed_ratio_max": args.bg_preserve_changed_ratio_max,
        "enforce_bg_preserve": args.enforce_bg_preserve,
        "preserve_object_names": human_part,
        "bg_overremove_check": args.bg_overremove_check,
        "bg_overremove_model": args.bg_overremove_model,
        "prefer_gemini_cleanup": 1 if int(args.prefer_gemini_cleanup) == 1 else 0,
        "gemini_cleanup_only": 1 if int(args.gemini_cleanup_only) == 1 else 0,
        "use_gemini_cleanup_fallback": args.use_gemini_cleanup_fallback,
        "use_seg_bbox_for_transform": args.use_seg_bbox_for_transform,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "controlnet_end": args.controlnet_end,
        "seed": args.seed,
        "presence_diff_thresh": args.presence_diff_thresh,
        "nodepth_prefer_ratio": args.nodepth_prefer_ratio,
        "enable_nodepth_compare": args.enable_nodepth_compare,
        "depth_coverage_nodepth_gate": args.depth_coverage_nodepth_gate,
        "bottom_center_depth_bias": args.bottom_center_depth_bias,
        "force_no_depth_control": _force_no_depth,
        "cache_models_cpu": 1 if int(args.cache_models_cpu) == 1 else 0,
        "enable_rescore": 0,
        "override_scale_factor": sf,
        "override_anchor_point": sanitize_anchor_point(str(plan.get("anchor_point", "CENTER"))),
    }

    try:
        next_img = run_single_correction_cached(
            args.single_infer_script,
            image_path,
            step_dir,
            product,
            human_part,
            extra_args=extra_args,
        )
        copyfile(next_img, final_path)
        if final_path != final_path_legacy:
            copyfile(next_img, final_path_legacy)
        correction_meta = _read_correction_meta(next_img)
        if correction_meta and str(correction_meta.get("status") or "") == "ok":
            bb = correction_meta.get("bbox_target_scaled_pixels_yxyx")
            if isinstance(bb, list) and len(bb) == 4:
                _layout_update_object_bbox(layout_final, product, bb)
        layout_final["image_path"] = os.path.abspath(final_path)
        _save_layout_json_and_vis(
            layout_final,
            final_path,
            os.path.join(case_output_dir, "layout_final.json"),
            os.path.join(case_output_dir, "layout_final_vis.png"),
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_type": "T2",
                    "global_plan": plan,
                    "multi_object_history": [{
                        "round": 1,
                        "target": product,
                        "anchor": human_part,
                        "output": os.path.abspath(final_path),
                        "plan": plan,
                        "scale_exec": float(sf),
                        "correction_meta": correction_meta,
                        "correction_meta_path": (
                            os.path.abspath(os.path.join(os.path.dirname(next_img), "correction_meta.json"))
                            if correction_meta is not None
                            else None
                        ),
                        "skipped": correction_meta is None,
                        "skip_reason": None if correction_meta is not None else "missing_correction_meta",
                    }],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"   Final saved to: {final_path}", flush=True)
    except Exception as e:
        print(f"   Size correction failed: {e}. Save original.", flush=True)
        copyfile(image_path, final_path)
        if final_path != final_path_legacy:
            copyfile(image_path, final_path_legacy)
        layout_final["image_path"] = os.path.abspath(final_path)
        _save_layout_json_and_vis(
            layout_final,
            final_path,
            os.path.join(case_output_dir, "layout_final.json"),
            os.path.join(case_output_dir, "layout_final_vis.png"),
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_type": "T2",
                    "global_plan": plan,
                    "multi_object_history": [{
                        "round": 1,
                        "target": product,
                        "anchor": human_part,
                        "output": os.path.abspath(final_path),
                        "plan": plan,
                        "scale_exec": float(sf),
                        "skipped": True,
                        "skip_reason": f"exception:{e}",
                    }],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Task 2 size correction: product vs human body anchor")
    parser.add_argument("--image_path", type=str,
                        default="")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="Directory containing T2_XXXX.png images",
    )
    parser.add_argument("--max_images", type=int, default=1, help="Max images to process. 0 = all.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--single_infer_script",
        type=str,
        default=str(PROJECT_ROOT / "inference_size_correction.py"),
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.environ.get("INSERTANYTHING_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "314000")),
        help="LoRA + ControlNet checkpoint dir (and optional hf_latent_inject.pt); forwarded to inference_size_correction.py",
    )
    parser.add_argument("--benchmark_json", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument(
        "--feedback_json_path",
        type=str,
        default="",
        help="Optional Task2 gemini_scores_*.json. If present, use its single pair score as product-only feedback.",
    )
    parser.add_argument(
        "--use_feedback_planner",
        type=int,
        default=1,
        help="1: use --feedback_json_path for skip/direction/tier gates; 0: ignore feedback JSON.",
    )
    parser.add_argument(
        "--product",
        type=str,
        default="",
        help="Product (target) name when the image stem is not in benchmark JSON",
    )
    parser.add_argument(
        "--human_part",
        type=str,
        default="",
        help="Human body anchor name when the image stem is not in benchmark JSON",
    )
    parser.add_argument(
        "--use_depth_guidance",
        type=int,
        default=0,
        help="0 (default): no depth ControlNet at generation time (mask-only). "
        "1: allow depth ControlNet when fusion enables it. Occlusion-heavy Task2: keep 0.",
    )
    parser.add_argument(
        "--force_no_depth_control",
        type=int,
        default=None,
        help="Optional override: 1=mask-only (drop depth CN), 0=allow depth CN. "
        "If unset, derived from --use_depth_guidance (legacy compatibility).",
    )
    parser.add_argument("--model_name", type=str, default="",
                        help="Optional model folder name for output grouping. If empty, infer from image_dir only.")
    parser.add_argument("--use_dino", action="store_true", help="Use DINO for detection (requires transformers)")
    parser.add_argument("--min_scale_delta", type=float, default=0.12)
    parser.add_argument("--min_ratio_log_error", type=float, default=0.16)
    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=2.5)
    parser.add_argument("--t2_use_residual_correction", type=int, default=1)
    parser.add_argument("--t2_residual_log_step", type=float, default=0.10)
    parser.add_argument("--t2_residual_max_log_step", type=float, default=0.25)
    parser.add_argument("--model_version", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--gpu_gen", type=int, default=0)
    parser.add_argument("--gpu_tools", type=int, default=0)
    # HF detail branch (forwarded to inference_size_correction.py)
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
    # Square crop side = ratio * max(bbox_h, bbox_w); larger ratio = wider crop = less zoom (more scene context).
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=3.3,
        help="Zoom crop: crop_size = ratio * max(bbox side). Higher = less aggressive zoom, more background (was 2.5).",
    )
    parser.add_argument(
        "--mask_dilate_kernel",
        type=int,
        default=11,
        help="Task2 default smaller than T1: mild square-kernel dilation when depth-aware is off",
    )
    parser.add_argument("--mask_dilate_iter", type=int, default=1)
    parser.add_argument(
        "--mask_dilate_boost",
        type=float,
        default=1.0,
        help="1.0 = no extra boost (Task2 default)",
    )
    parser.add_argument(
        "--mask_dilate_iter_boost",
        type=float,
        default=1.0,
        help="1.0 = no extra iter boost (Task2 default)",
    )
    parser.add_argument("--mask_dilate_depth_aware", type=int, default=1)
    parser.add_argument(
        "--mask_dilate_ratio",
        type=float,
        default=0.06,
        help="Depth-aware border fraction of min(target_h,target_w); T2 default 0.06 (tighter than 0.10).",
    )
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
        help="After dilation, simplify inpaint mask. Default 'close' (mild). "
        "'close_convex_hull' fills concave bites but balloons long thin masks (knife, tools) "
        "via convex hull — use only if SAM silhouette needs notch filling.",
    )
    parser.add_argument(
        "--mask_regularize_close_ksz",
        type=int,
        default=15,
        help="Ellipse kernel for MORPH_CLOSE in mask regularize (odd; default 15 for T2).",
    )
    parser.add_argument(
        "--mask_regularize_close_iter",
        type=int,
        default=2,
        help="Close iterations when regularize uses close (default 2).",
    )
    parser.add_argument(
        "--mask_crop_fill_holes",
        type=int,
        default=1,
        help="1: after 768 resize, flood-fill enclosed holes in mask (default on).",
    )
    parser.add_argument(
        "--mask_crop_post_dilate",
        type=int,
        default=0,
        help="After 768 resize + optional hole fill, extra 3x3 dilate iterations (default 0). Set 1 to close hairline gaps.",
    )
    parser.add_argument("--ref_upscale_threshold", type=int, default=128)
    parser.add_argument("--ref_upscale_target", type=int, default=338)
    parser.add_argument("--blend_feather_border_px", type=int, default=6)
    parser.add_argument("--blend_alpha_blur_sigma", type=float, default=0.8)
    parser.add_argument("--disable_blend_feather", type=int, default=0)
    parser.add_argument("--bg_removal_mode", type=str, default="gemini", choices=["flux", "gemini", "editing"])
    parser.add_argument("--editing_steps", type=int, default=28)
    parser.add_argument("--editing_guidance_scale", type=float, default=4.0)
    parser.add_argument("--bg_remove_max_retries", type=int, default=2)
    parser.add_argument("--bg_remove_diff_thresh", type=float, default=0.065)
    parser.add_argument("--bg_remove_changed_ratio_thresh", type=float, default=0.20)
    parser.add_argument("--bg_remove_pixel_diff_cutoff", type=float, default=0.06)
    parser.add_argument("--bg_preserve_diff_mean_max", type=float, default=0.05)
    parser.add_argument("--bg_preserve_changed_ratio_max", type=float, default=0.2)
    parser.add_argument("--enforce_bg_preserve", type=int, default=0)
    parser.add_argument("--bg_overremove_check", type=int, default=1,
                        help="1: Gemini QA rejects object-removal results that damage the human anchor.")
    parser.add_argument("--bg_overremove_model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--prefer_gemini_cleanup", type=int, default=1,
                        help="1: prefer Gemini image cleanup before Flux removal")
    parser.add_argument("--gemini_cleanup_only", type=int, default=1,
                        help="1: only use Gemini image cleanup for removal in T2")
    parser.add_argument("--use_gemini_cleanup_fallback", type=int, default=1)
    parser.add_argument("--use_seg_bbox_for_transform", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--controlnet_scale", type=float, default=0.6)
    parser.add_argument("--controlnet_end", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--presence_diff_thresh", type=float, default=0.03)
    parser.add_argument("--enable_nodepth_compare", type=int, default=0)
    parser.add_argument("--cache_models_cpu", type=int, default=1,
                        help="1: reuse single-image models across images (CPU cache, GPU on demand)")
    # T2-specific depth params (product-human occlusion)
    parser.add_argument("--depth_coverage_nodepth_gate", type=float, default=0.40)
    parser.add_argument("--bottom_center_depth_bias", type=float, default=10.0)
    parser.add_argument("--nodepth_prefer_ratio", type=float, default=1.1)
    parser.add_argument("--dry_run", action="store_true", help="Validate setup only, skip API and correction")
    args = parser.parse_args()

    if args.dry_run:
        t2_entries = load_t2_entries(args.benchmark_json)
        image_paths = collect_image_paths(args.image_path, args.image_dir, args.max_images)
        print(f"Dry run: {len(t2_entries)} T2 entries, {len(image_paths)} images to process.", flush=True)
        inferred_model_name = ""
        if args.model_name:
            inferred_model_name = args.model_name.strip()
        elif args.image_dir:
            inferred_model_name = os.path.basename(os.path.normpath(args.image_dir))
        out_root = args.output_dir
        if inferred_model_name:
            out_root = os.path.join(args.output_dir, inferred_model_name)
        for img in image_paths[:3]:
            stem = Path(img).stem
            entry = t2_entries.get(stem)
            if entry:
                p, h = extract_product_and_human(entry)
            elif args.product.strip() and args.human_part.strip():
                p, h = args.product.strip(), args.human_part.strip()
            else:
                p, h = None, None
            if p and h:
                sub = t2_session_dir_name(stem, p, h)
                print(f"  {stem}: product={p[:40]}... anchor={h} -> {out_root}/{sub}/", flush=True)
            else:
                print(f"  {stem}: no T2 entry and no --product/--human_part", flush=True)
        print("Dry run OK.", flush=True)
        return

    t2_entries = load_t2_entries(args.benchmark_json)
    if os.path.isfile(args.benchmark_json):
        print(f"Loaded {len(t2_entries)} T2 benchmark entries.", flush=True)
    else:
        print(f"Benchmark JSON not found ({args.benchmark_json}); using --product/--human_part only.", flush=True)

    inferred_model_name = ""
    if args.model_name:
        inferred_model_name = args.model_name.strip()
    elif args.image_dir:
        inferred_model_name = os.path.basename(os.path.normpath(args.image_dir))

    output_root = args.output_dir
    if inferred_model_name:
        output_root = os.path.join(args.output_dir, inferred_model_name)
    os.makedirs(output_root, exist_ok=True)
    print(f"Output root: {output_root}", flush=True)

    gemini = GeminiClient(model_version=args.model_version)

    dino = None
    if args.use_dino:
        try:
            dino = DINOClient(device="cuda")
            print("DINO loaded for detection.", flush=True)
        except Exception as e:
            print(f"DINO load failed ({e}), use Gemini only.", flush=True)

    image_paths = collect_image_paths(args.image_path, args.image_dir, args.max_images)
    print(f"Processing {len(image_paths)} image(s).", flush=True)

    summary = []
    for idx, img in enumerate(image_paths, start=1):
        stem = Path(img).stem
        task_id = stem

        entry = t2_entries.get(task_id)
        if entry:
            product, human_part = extract_product_and_human(entry)
        elif args.product.strip() and args.human_part.strip():
            product = simplify_product_name_for_detection(args.product.strip())
            human_part = args.human_part.strip()
        else:
            print(f"\n[{idx}/{len(image_paths)}] {stem} no T2 entry for {task_id}. "
                  f"Provide --product and --human_part or fix --benchmark_json. Skip.", flush=True)
            summary.append({"image_path": img, "task_id": task_id, "status": "no_entry"})
            continue

        if not product or not human_part:
            print(f"\n[{idx}/{len(image_paths)}] {stem} invalid objects. Skip.", flush=True)
            summary.append({"image_path": img, "task_id": task_id, "status": "invalid_objects"})
            continue

        base_session = t2_session_dir_name(stem, product, human_part)
        session_subdir = allocate_unique_session_subdir_name(output_root, base_session)
        case_dir = os.path.join(output_root, session_subdir)
        final_result = os.path.join(case_dir, "final_result.png")
        final_t2 = os.path.join(case_dir, "final_t2_result.png")

        print(f"\n================ [{idx}/{len(image_paths)}] {img} ================", flush=True)
        print(f"   Session dir: {case_dir}", flush=True)
        if session_subdir != base_session:
            print(f"   (base name '{base_session}' existed → using '{session_subdir}')", flush=True)
        print(f"   Product (detection label): {product[:80]}{'...' if len(product) > 80 else ''}", flush=True)
        print(f"   Anchor (human): {human_part}", flush=True)
        print(f"   Weights dir: {args.weights_dir}", flush=True)
        _size_hint = extract_benchmark_product_size_hint(entry)
        if _size_hint:
            print(f"   Size hint (benchmark prompt): {_size_hint[:160]}{'...' if len(_size_hint) > 160 else ''}", flush=True)

        try:
            final_path = run_t2_for_image(
                gemini, dino, img, case_dir, product, human_part, args,
                product_size_hint=_size_hint,
            )
            row = {
                "image_path": img,
                "task_id": task_id,
                "final_path": final_path,
                "status": "generated",
                "session_subdir": session_subdir,
            }
            summary.append(row)
        except Exception as e:
            print(f"   ERROR: {e}", flush=True)
            from shutil import copyfile

            os.makedirs(case_dir, exist_ok=True)
            copyfile(img, final_result)
            copyfile(img, final_t2)
            summary.append({"image_path": img, "task_id": task_id, "final_path": final_result, "status": "failed_exception", "error": str(e)})

    with open(os.path.join(output_root, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Summary: {len(summary)} images.", flush=True)


if __name__ == "__main__":
    main()
