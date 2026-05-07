"""
Task 3 size correction inference script.

Only process prompt_type == "precise_scale_instruction".
Scale factor is taken from Task3 benchmark (edit_plan / generated_ratio / target_ratio),
instead of asking Gemini to estimate ratios again.

Design rules:
1) Always edit the GT smaller object only.
2) Keep GT larger object as anchor (unchanged).
3) Ask Gemini only for direction/anchor-point hint; if conflict with GT plan, keep GT plan.

Usage:
    # Single image
    python inference_size_correction_t3.py --image_path path/to/T3_0001.png

    # Batch folder (supports nested prompt_type folders)
    python inference_size_correction_t3.py --image_dir path/to/model_folder --max_images 0
"""

import argparse
import glob
import importlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from multi_object_inference import GeminiClient, sanitize_anchor_point


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
SUBMISSION_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK = SUBMISSION_ROOT / "GenScale_Benchmark_v3_final_anonymous.json"
DEFAULT_OUTPUT_DIR = SUBMISSION_ROOT / "outputs" / "size_correction_T3"
_SC_MODULE = None


def run_single_correction_cached(
    script_path: str,
    image_path: str,
    output_dir: str,
    target: str,
    anchor: str,
    extra_args: Dict,
):
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


def load_t3_precise_entries(benchmark_json: str) -> Dict[str, dict]:
    with open(benchmark_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for e in data:
        if not str(e.get("task_id", "")).startswith("T3_"):
            continue
        if str(e.get("prompt_type", "")) != "precise_scale_instruction":
            continue
        out[e["task_id"]] = e
    return out


def collect_image_paths(image_path: str, image_dir: str, max_images: int) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths: List[str] = []
    if image_dir:
        # recursive to support model/prompt_type/T3_xxxx.png
        for p in sorted(Path(image_dir).rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))
    elif image_path:
        paths = [image_path]
    else:
        raise ValueError("Provide either --image_path or --image_dir")
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def parse_t3_plan(entry: dict, min_scale: float, max_scale: float) -> Tuple[str, str, float, str, str]:
    """
    Returns:
      (target_smaller, anchor_larger, scale_factor, gt_direction, pair_key)
    """
    pair_key = str(entry.get("pair_key", "")).strip()
    if "_to_" not in pair_key:
        gt = entry.get("gt_ratios", {})
        if len(gt) != 1:
            raise ValueError("Cannot resolve pair_key from entry.")
        pair_key = next(iter(gt.keys()))
    obj_a, obj_b = pair_key.split("_to_", 1)

    target_ratio = float(entry.get("target_ratio", 0.0) or 0.0)
    generated_ratio = float(entry.get("generated_ratio", 0.0) or 0.0)
    if target_ratio <= 0:
        gt = entry.get("gt_ratios", {}).get(pair_key, {})
        target_ratio = float(gt.get("target_ratio", 0.0) or 0.0)
    if target_ratio <= 0:
        raise ValueError(f"Invalid target_ratio for {entry.get('task_id')}")

    # STRICT: edit GT smaller object only.
    if target_ratio >= 1.0:
        # A/B >= 1 => B is smaller (editable), A fixed.
        target_smaller, anchor_larger = obj_b, obj_a
        sf_formula = generated_ratio / max(target_ratio, 1e-6) if generated_ratio > 0 else 1.0
    else:
        # A/B < 1 => A is smaller (editable), B fixed.
        target_smaller, anchor_larger = obj_a, obj_b
        sf_formula = target_ratio / max(generated_ratio, 1e-6) if generated_ratio > 0 else 1.0

    sf = None
    ep = entry.get("edit_plan", {})
    if isinstance(ep, dict):
        ep_sf = ep.get("scale_factor", None)
        if ep_sf is not None:
            try:
                sf = float(ep_sf)
            except Exception:
                sf = None

    if sf is None or not math.isfinite(sf) or sf <= 0:
        sf = sf_formula
    sf = max(min_scale, min(max_scale, float(sf)))
    gt_direction = "enlarge" if sf >= 1.0 else "shrink"
    return target_smaller, anchor_larger, sf, gt_direction, pair_key


def ask_direction_anchor_hint(
    gemini: GeminiClient,
    image_path: str,
    target_smaller: str,
    anchor_larger: str,
    gt_direction: str,
) -> Tuple[str, str, float]:
    """
    Ask Gemini for direction and anchor-point hint.
    If direction conflicts with GT plan, keep GT direction.
    """
    prompt = f"""
Analyze this image for targeted size correction.
Editable object: "{target_smaller}" (this is the smaller object to edit).
Anchor object: "{anchor_larger}" (must stay unchanged).

Return JSON only:
{{
  "direction": "enlarge or shrink",
  "anchor_point": "BOTTOM_CENTER or TOP_CENTER or CENTER or LEFT_CENTER or RIGHT_CENTER",
  "confidence": 0.0,
  "reason": "short"
}}
"""
    try:
        res = gemini._call(prompt, image_path)  # pylint: disable=protected-access
        pred_dir = str(res.get("direction", "")).strip().lower()
        ap = sanitize_anchor_point(str(res.get("anchor_point", "BOTTOM_CENTER")))
        conf = float(res.get("confidence", 0.0) or 0.0)
        if pred_dir not in {"enlarge", "shrink"}:
            pred_dir = gt_direction
        if pred_dir != gt_direction:
            # strict GT rule: do not flip target direction
            pred_dir = gt_direction
        return pred_dir, ap, conf
    except Exception:
        return gt_direction, "BOTTOM_CENTER", 0.0


def run_t3_for_image(
    gemini: GeminiClient,
    image_path: str,
    case_output_dir: str,
    entry: dict,
    args,
) -> str:
    from shutil import copyfile

    os.makedirs(case_output_dir, exist_ok=True)
    final_path = os.path.join(case_output_dir, "final_t3_result.png")

    target_smaller, anchor_larger, sf, gt_direction, pair_key = parse_t3_plan(
        entry=entry,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    dir_hint, anchor_point, conf = ask_direction_anchor_hint(
        gemini=gemini,
        image_path=image_path,
        target_smaller=target_smaller,
        anchor_larger=anchor_larger,
        gt_direction=gt_direction,
    )
    print(
        f"   Pair={pair_key} target(smaller)={target_smaller} anchor(larger)={anchor_larger} "
        f"scale_exec={sf:.4f} gt_dir={gt_direction} gemini_dir={dir_hint} "
        f"anchor_point={anchor_point} gemini_conf={conf:.2f}",
        flush=True,
    )

    # If direction somehow disagrees with scale sign, force GT-consistent scale.
    if gt_direction == "enlarge" and sf < 1.0:
        sf = max(1.0, 1.0 / max(sf, 1e-6))
    elif gt_direction == "shrink" and sf > 1.0:
        sf = min(1.0, 1.0 / max(sf, 1e-6))

    step_dir = os.path.join(case_output_dir, "round_1", target_smaller.replace("/", "_")[:80])
    os.makedirs(step_dir, exist_ok=True)

    extra_args = {
        "gpu_gen": args.gpu_gen,
        "gpu_tools": args.gpu_tools,
        "bg_removal_mode": args.bg_removal_mode,
        "editing_steps": args.editing_steps,
        "editing_guidance_scale": args.editing_guidance_scale,
        "prefer_gemini_cleanup": 1 if int(args.prefer_gemini_cleanup) == 1 else 0,
        "gemini_cleanup_only": 1 if int(args.gemini_cleanup_only) == 1 else 0,
        "cache_models_cpu": 1 if int(args.cache_models_cpu) == 1 else 0,
        "min_scale": args.min_scale,
        "override_scale_factor": sf,
        "override_anchor_point": sanitize_anchor_point(anchor_point),
        "depth_coverage_nodepth_gate": args.depth_coverage_nodepth_gate,
        "bottom_center_depth_bias": args.bottom_center_depth_bias,
        "nodepth_prefer_ratio": args.nodepth_prefer_ratio,
    }

    try:
        next_img = run_single_correction_cached(
            args.single_infer_script,
            image_path,
            step_dir,
            target_smaller,
            anchor_larger,
            extra_args=extra_args,
        )
        copyfile(next_img, final_path)
        print(f"   Final saved to: {final_path}", flush=True)
    except Exception as e:
        print(f"   Size correction failed: {e}. Save original.", flush=True)
        copyfile(image_path, final_path)
    return final_path


def infer_model_name(args) -> str:
    if args.model_name:
        return args.model_name.strip()
    if args.image_dir:
        return os.path.basename(os.path.normpath(args.image_dir))
    if args.image_path:
        return os.path.basename(os.path.dirname(os.path.abspath(args.image_path)))
    return ""


def infer_prompt_type_from_path(image_path: str) -> str:
    p = Path(image_path)
    for part in p.parts:
        if part in {"precise_scale_instruction", "hard_auto_discovery"}:
            return part
    return "unknown_prompt_type"


def main():
    parser = argparse.ArgumentParser(description="Task 3 size correction (precise instruction only)")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="", help="Optional folder filter for T3 images (recursive)")
    parser.add_argument("--max_images", type=int, default=0, help="Max images to process. 0 = all precise entries.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--single_infer_script", type=str, default=str(PROJECT_ROOT / "inference_size_correction.py"))
    parser.add_argument("--benchmark_json", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_version", type=str, default="gemini-3-flash-preview")

    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=2.5)
    parser.add_argument("--gpu_gen", type=int, default=0)
    parser.add_argument("--gpu_tools", type=int, default=0)
    parser.add_argument("--bg_removal_mode", type=str, default="editing")
    parser.add_argument("--editing_steps", type=int, default=28)
    parser.add_argument("--editing_guidance_scale", type=float, default=4.0)
    parser.add_argument("--prefer_gemini_cleanup", type=int, default=1)
    parser.add_argument("--gemini_cleanup_only", type=int, default=1)
    parser.add_argument("--cache_models_cpu", type=int, default=1)
    parser.add_argument("--depth_coverage_nodepth_gate", type=float, default=0.40)
    parser.add_argument("--bottom_center_depth_bias", type=float, default=10.0)
    parser.add_argument("--nodepth_prefer_ratio", type=float, default=1.1)
    parser.add_argument("--dry_run", action="store_true", help="Validate setup and filtering only.")
    args = parser.parse_args()

    t3_entries = load_t3_precise_entries(args.benchmark_json)
    print(f"Loaded {len(t3_entries)} Task3 precise entries from benchmark.", flush=True)

    # Default: benchmark-driven processing (no folder scan required).
    # Optional --image_path / --image_dir can be used as filters.
    selected_entries = []
    if args.image_path:
        stem = Path(args.image_path).stem
        e = t3_entries.get(stem)
        if e:
            selected_entries = [e]
    elif args.image_dir:
        filter_stems = set()
        for p in sorted(Path(args.image_dir).rglob("*")):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                filter_stems.add(p.stem)
        selected_entries = [e for tid, e in t3_entries.items() if tid in filter_stems]
    else:
        selected_entries = list(t3_entries.values())

    selected_entries.sort(key=lambda x: str(x.get("task_id", "")))
    if args.max_images and args.max_images > 0:
        selected_entries = selected_entries[:args.max_images]

    valid_entries = []
    for e in selected_entries:
        img = str(e.get("image_path", "")).strip()
        if img and os.path.exists(img):
            valid_entries.append(e)
    print(f"Will process {len(valid_entries)} precise entries (json-driven).", flush=True)

    if args.dry_run:
        for e in valid_entries[:5]:
            tid = str(e.get("task_id", ""))
            print(f"  {tid}: pair={e.get('pair_key')} prompt_type={e.get('prompt_type')}", flush=True)
        print("Dry run OK.", flush=True)
        return

    if not valid_entries:
        print("No valid precise_scale_instruction images found. Exit.", flush=True)
        return

    model_name = infer_model_name(args)
    output_root = os.path.join(args.output_dir, model_name) if model_name else args.output_dir
    os.makedirs(output_root, exist_ok=True)
    print(f"Output root: {output_root}", flush=True)

    gemini = GeminiClient(model_version=args.model_version)
    summary = []

    for idx, entry in enumerate(valid_entries, start=1):
        tid = str(entry.get("task_id", ""))
        img = str(entry.get("image_path", ""))
        prompt_type = "precise_scale_instruction"
        case_dir = os.path.join(output_root, prompt_type, tid)
        final_t3 = os.path.join(case_dir, "final_t3_result.png")

        if os.path.exists(final_t3):
            print(f"\n[{idx}/{len(precise_paths)}] {tid} already done, skip.", flush=True)
            summary.append({"image_path": img, "task_id": tid, "final_path": final_t3, "status": "skipped_exists"})
            continue

        print(f"\n================ [{idx}/{len(valid_entries)}] {img} ================", flush=True)
        try:
            final_path = run_t3_for_image(gemini, img, case_dir, entry, args)
            summary.append({"image_path": img, "task_id": tid, "final_path": final_path, "status": "generated"})
        except Exception as e:
            print(f"   ERROR: {e}", flush=True)
            from shutil import copyfile

            os.makedirs(case_dir, exist_ok=True)
            copyfile(img, final_t3)
            summary.append({
                "image_path": img,
                "task_id": tid,
                "final_path": final_t3,
                "status": "failed_exception",
                "error": str(e),
            })

    with open(os.path.join(output_root, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Summary: {len(summary)} images.", flush=True)


if __name__ == "__main__":
    main()
