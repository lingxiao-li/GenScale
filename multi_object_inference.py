import argparse
import base64
import glob
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_version: str = "gemini-3-processingInstruction(target, data)-preview"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_version}:generateContent?key={self.api_key}"
        )

    def _call(self, prompt: str, image_path: str, retries: int = 3):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": encoded}},
                ]
            }],
            "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"},
        }

        err = None
        for _ in range(retries):
            try:
                r = requests.post(self.url, headers={"Content-Type": "application/json"}, json=payload, timeout=90)
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}: {r.text[:400]}"
                    time.sleep(1.0)
                    continue
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                i, j = text.find("{"), text.rfind("}")
                if i == -1 or j == -1 or j <= i:
                    err = f"No JSON object found: {text[:300]}"
                    time.sleep(0.5)
                    continue
                return json.loads(text[i:j + 1])
            except Exception as e:
                err = str(e)
                time.sleep(1.0)
        raise RuntimeError(f"Gemini call failed: {err}")

    def detect_objects(self, image_path: str, expected_objects: Optional[List[str]], max_objects: int = 4):
        exp_text = ", ".join(expected_objects) if expected_objects else "unknown"
        prompt = f"""
Detect up to {max_objects} major foreground objects in this image.

If expected objects are provided, prioritize these names:
{exp_text}

Rules:
1) Return tight bounding boxes for visible object extent.
2) If expected object is missing, set present=false.
3) Use exact object names when possible.

Output JSON schema:
{{
  "objects": [
    {{
      "name": "object name",
      "present": true,
      "bbox": [ymin, xmin, ymax, xmax],
      "bbox_format": "pixel_or_0_to_1000",
      "confidence": 0.0
    }}
  ]
}}
"""
        res = self._call(prompt, image_path)
        objs = res.get("objects", [])
        valid = []
        for o in objs:
            if not o.get("present", True):
                continue
            b = o.get("bbox")
            if not b or len(b) != 4:
                continue
            valid.append(o)
        return valid

    def refine_bbox(self, image_path: str, object_name: str, coarse_bbox: List[float]):
        prompt = f"""
Refine the bounding box for object '{object_name}'.
Current bbox: {coarse_bbox}.
Return a tighter and accurate bbox covering the full object, with minimal background.

Output JSON:
{{"bbox":[ymin,xmin,ymax,xmax], "bbox_format":"pixel_or_0_to_1000"}}
"""
        res = self._call(prompt, image_path)
        b = res.get("bbox", coarse_bbox)
        if isinstance(b, list) and len(b) == 4:
            return b
        return coarse_bbox

    def estimate_scale_vs_anchor(
        self,
        image_path: str,
        target_object: str,
        anchor_object: str,
        product_size_hint: Optional[str] = None,
    ):
        size_block = ""
        if product_size_hint and str(product_size_hint).strip():
            size_block = f"""
Benchmark / catalog size prior for the TARGET product (from dataset metadata; use to inform desired_ratio):
{str(product_size_hint).strip()}
"""
        prompt = f"""
Analyze object size realism in this image using real-world knowledge.
Target object: '{target_object}'.
Anchor object: '{anchor_object}' (assume anchor size is correct and should stay unchanged).
{size_block}
Step 1) Estimate CURRENT apparent size ratio between target and anchor in this image.
Step 2) Estimate DESIRED plausible ratio between target and anchor in real world.
Step 3) Derive scale_factor for target:
  scale_factor ~= desired_ratio / current_ratio
  (<1 shrink, >1 enlarge).

Ratio definition:
- Use LINEAR size ratio (characteristic length), not area ratio.

Provide best anchor_point among:
BOTTOM_CENTER, TOP_CENTER, CENTER, LEFT_CENTER, RIGHT_CENTER

Output JSON:
{{
  "need_correction": true,
  "current_ratio": 0.30,
  "desired_ratio": 0.10,
  "ratio_type": "linear",
  "scale_factor": 0.75,
  "anchor_point": "BOTTOM_CENTER",
  "confidence": 0.0,
  "reason": "short reason"
}}
"""
        res = self._call(prompt, image_path)
        return {
            "need_correction": bool(res.get("need_correction", False)),
            "current_ratio": float(res.get("current_ratio", 0.0) or 0.0),
            "desired_ratio": float(res.get("desired_ratio", 0.0) or 0.0),
            "ratio_type": str(res.get("ratio_type", "linear") or "linear"),
            "scale_factor": float(res.get("scale_factor", 1.0)),
            "anchor_point": str(res.get("anchor_point", "BOTTOM_CENTER")),
            "confidence": float(res.get("confidence", 0.0)),
            "reason": str(res.get("reason", "")),
        }

    def check_global_done(self, image_path: str, object_names: List[str]):
        names = ", ".join(object_names)
        prompt = f"""
Check if object size proportions are plausible in this image among:
{names}

Output JSON:
{{"all_correct": true, "reason":"short reason"}}
"""
        res = self._call(prompt, image_path)
        return bool(res.get("all_correct", False)), str(res.get("reason", ""))


def normalize_bbox_to_pixels(bbox: List[float], W: int, H: int):
    is_norm = max(bbox) <= 1.0
    sy = H if is_norm else H / 1000.0
    sx = W if is_norm else W / 1000.0
    y1, x1, y2, x2 = [float(v) for v in bbox]
    return [int(y1 * sy), int(x1 * sx), int(y2 * sy), int(x2 * sx)]


def load_expected_objects(benchmark_json: str, task_id: str) -> Optional[List[str]]:
    if not benchmark_json or not task_id:
        return None
    with open(benchmark_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item.get("task_id") == task_id:
            return item.get("objects_included", None)
    return None


def normalize_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_object_by_expected(detected_objects: List[Dict], expected_name: str) -> Optional[Dict]:
    exp = normalize_name(expected_name)
    best = None
    best_score = -1.0
    for o in detected_objects:
        name = normalize_name(o.get("name", ""))
        if not name:
            continue
        score = 0.0
        if name == exp:
            score = 1.0
        elif exp in name or name in exp:
            score = 0.8
        else:
            a = set(name.split())
            b = set(exp.split())
            if a or b:
                score = len(a & b) / max(1, len(a | b))
        if score > best_score:
            best_score = score
            best = o
    return best if best_score >= 0.34 else None


def sanitize_anchor_point(anchor: str) -> str:
    valid = {"BOTTOM_CENTER", "TOP_CENTER", "CENTER", "LEFT_CENTER", "RIGHT_CENTER"}
    a = (anchor or "").strip().upper()
    return a if a in valid else "BOTTOM_CENTER"


def compute_scale_exec(plan: Dict, min_scale: float, max_scale: float) -> float:
    current_ratio = float(plan.get("current_ratio", 0.0) or 0.0)
    desired_ratio = float(plan.get("desired_ratio", 0.0) or 0.0)
    ratio_type = str(plan.get("ratio_type", "linear") or "linear").strip().lower()
    fallback = float(plan.get("scale_factor", 1.0) or 1.0)

    sf = fallback
    if current_ratio > 1e-8 and desired_ratio > 1e-8:
        ratio = desired_ratio / current_ratio
        if ratio_type == "area":
            sf = math.sqrt(max(1e-8, ratio))
        else:
            sf = ratio
    if not math.isfinite(sf):
        sf = fallback
    return max(min_scale, min(max_scale, float(sf)))


def run_single_correction(script_path: str, image_path: str, output_dir: str, target: str, anchor: str, extra_args: Dict):
    cmd = [
        "python", script_path,
        "--image_path", image_path,
        "--output_dir", output_dir,
        "--target_object", target,
        "--ref_object", anchor,
    ]
    for k, v in extra_args.items():
        if v is None:
            continue
        cmd.extend([f"--{k}", str(v)])

    print("   Running:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)

    session_dir = os.path.join(output_dir, f"{target}_vs_{anchor}")
    direct = os.path.join(session_dir, "final_result.png")
    if os.path.exists(direct):
        return direct

    cands = glob.glob(os.path.join(output_dir, "**", "final_result.png"), recursive=True)
    if not cands:
        raise RuntimeError(f"No final_result.png found under {output_dir}")
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def collect_image_paths(image_path: str, image_dir: str, max_images: int) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths: List[str] = []
    if image_path:
        paths = [image_path]
    elif image_dir:
        for p in sorted(Path(image_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))
    else:
        raise ValueError("Provide either --image_path or --image_dir")
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def run_multi_for_image(gemini: GeminiClient, image_path: str, case_output_dir: str, expected_objects: Optional[List[str]], args):
    os.makedirs(case_output_dir, exist_ok=True)
    current_image = image_path
    history = []
    if not expected_objects:
        raise RuntimeError("Expected objects are required. Pass --benchmark_json and valid task_id/image stem.")

    # Round 0: detect only expected objects and freeze largest as immutable anchor.
    seed_detect = gemini.detect_objects(current_image, expected_objects, max_objects=max(len(expected_objects), args.max_objects))
    if not seed_detect:
        raise RuntimeError("No objects detected at round-0.")

    from PIL import Image
    W, H = Image.open(current_image).size
    matched = []
    for exp in expected_objects:
        picked = pick_object_by_expected(seed_detect, exp)
        if picked is None:
            continue
        picked["bbox"] = gemini.refine_bbox(current_image, exp, picked["bbox"])
        y1, x1, y2, x2 = normalize_bbox_to_pixels(picked["bbox"], W, H)
        picked["name"] = exp
        picked["area"] = max(1, (y2 - y1) * (x2 - x1))
        matched.append(picked)
    if len(matched) < 2:
        raise RuntimeError(f"Expected objects not reliably detected. matched={len(matched)} expected={len(expected_objects)}")

    matched.sort(key=lambda z: z["area"], reverse=True)
    anchor = matched[0]["name"]
    targets = [x["name"] for x in matched[1:]]
    print(f"Anchor object (largest, immutable): {anchor}", flush=True)
    print(f"Targets ({len(targets)}): {targets}", flush=True)

    max_rounds = min(len(targets), max(1, args.max_rounds))
    for rd in range(1, max_rounds + 1):
        target = targets[rd - 1]
        print(f"\n========== Round {rd}/{max_rounds}: target={target} ==========", flush=True)

        # Re-detect target and anchor on current image, but keep names constrained.
        cur_objs = gemini.detect_objects(current_image, [anchor, target], max_objects=2)
        target_obj = pick_object_by_expected(cur_objs, target)
        anchor_obj = pick_object_by_expected(cur_objs, anchor)
        if target_obj is None or anchor_obj is None:
            print(f" - skip {target}: cannot reliably detect target/anchor this round.", flush=True)
            continue

        plan = gemini.estimate_scale_vs_anchor(current_image, target, anchor)
        sf_raw = float(plan["scale_factor"])
        sf = compute_scale_exec(plan, args.min_scale, args.max_scale)
        cur_r = float(plan.get("current_ratio", 0.0) or 0.0)
        des_r = float(plan.get("desired_ratio", 0.0) or 0.0)
        ratio_log_err = abs(math.log((cur_r + 1e-6) / (des_r + 1e-6))) if cur_r > 0 and des_r > 0 else 0.0
        need = (ratio_log_err >= args.min_ratio_log_error and abs(sf - 1.0) >= args.min_scale_delta) or (
            bool(plan["need_correction"]) and abs(sf - 1.0) >= args.min_scale_delta
        )
        print(
            f" - {target}: need_gemini={plan['need_correction']} "
            f"current_ratio={cur_r:.4f} desired_ratio={des_r:.4f} "
            f"ratio_log_err={ratio_log_err:.3f} "
            f"scale_raw={sf_raw:.3f} scale_exec={sf:.3f} conf={plan['confidence']:.2f}",
            flush=True,
        )
        if not need:
            print(f" - skip {target}: ratio error below threshold.", flush=True)
            continue

        step_dir = os.path.join(case_output_dir, f"round_{rd}", target.replace("/", "_"))
        os.makedirs(step_dir, exist_ok=True)
        next_img = run_single_correction(
            args.single_infer_script,
            current_image,
            step_dir,
            target,
            anchor,
            extra_args={
                "gpu_gen": args.gpu_gen,
                "gpu_tools": args.gpu_tools,
                "bg_removal_mode": args.bg_removal_mode,
                "editing_steps": args.editing_steps,
                "editing_guidance_scale": args.editing_guidance_scale,
                "min_scale": args.min_scale,
                "override_scale_factor": sf,
                "override_anchor_point": sanitize_anchor_point(str(plan.get("anchor_point", "BOTTOM_CENTER"))),
            },
        )
        current_image = next_img
        history.append({"round": rd, "target": target, "anchor": anchor, "output": next_img, "plan": plan, "scale_exec": sf})

        done, reason = gemini.check_global_done(current_image, expected_objects)
        print(f"Global check after round {rd}: done={done}, reason={reason}", flush=True)
        if done:
            print("Early stop: Gemini reports all expected objects are size-correct.", flush=True)
            break

    final_path = os.path.join(case_output_dir, "final_multi_object_result.png")
    from shutil import copyfile
    copyfile(current_image, final_path)

    with open(os.path.join(case_output_dir, "multi_object_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Final saved to: {final_path}", flush=True)
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str,
                        default="")
    parser.add_argument("--image_dir", type=str,
                        default="")
    parser.add_argument("--max_images", type=int, default=1,
                        help="Maximum number of images to process. 0 means all.")
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "outputs" / "size_correction_T1"))
    parser.add_argument("--single_infer_script", type=str, default=str(Path(__file__).resolve().parent / "inference_size_correction.py"))
    parser.add_argument("--benchmark_json", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "GenScale_Benchmark_v3_final_anonymous.json"))
    parser.add_argument("--task_id", type=str, default="")
    parser.add_argument("--max_objects", type=int, default=4)
    parser.add_argument("--max_rounds", type=int, default=3,
                        help="Upper bound; effective rounds are min(max_rounds, n_objects-1)")
    parser.add_argument("--min_scale_delta", type=float, default=0.12)
    parser.add_argument("--min_ratio_log_error", type=float, default=0.16,
                        help="Trigger correction when |log(current_ratio/desired_ratio)| exceeds this value")
    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=2.5)
    parser.add_argument("--model_version", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--gpu_gen", type=int, default=0)
    parser.add_argument("--gpu_tools", type=int, default=0)
    parser.add_argument("--bg_removal_mode", type=str, default="editing")
    parser.add_argument("--editing_steps", type=int, default=28)
    parser.add_argument("--editing_guidance_scale", type=float, default=4.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gemini = GeminiClient(model_version=args.model_version)
    image_paths = collect_image_paths(args.image_path, args.image_dir, args.max_images)
    print(f"Processing {len(image_paths)} image(s).", flush=True)

    summary = []
    for idx, img in enumerate(image_paths, start=1):
        stem = Path(img).stem
        case_dir = args.output_dir if len(image_paths) == 1 and args.image_path else os.path.join(args.output_dir, stem)
        task_id = args.task_id or stem
        expected_objects = load_expected_objects(args.benchmark_json, task_id) if args.benchmark_json else None
        print(f"\n================ [{idx}/{len(image_paths)}] {img} ================", flush=True)
        final_path = run_multi_for_image(gemini, img, case_dir, expected_objects, args)
        summary.append({"image_path": img, "task_id": task_id, "final_path": final_path})

    with open(os.path.join(args.output_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

