#!/usr/bin/env python3
"""
Gemini VLM-as-a-Judge evaluation for GenScale Task2 (human-product), v5.

v5 keeps the v4 scoring pipeline intact but uses a more human-calibrated prompt:
benchmark generation intent, explicit product/anchor ratio, and guidance for
bundles, folded products, packaging, jewelry, and human interaction scale.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import datetime
import json
import os
import re
import threading
import time
import itertools
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import RequestException, ReadTimeout


def _repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


DEFAULT_BENCHMARK = os.path.join(
    os.path.dirname(_repo_root()), "GenScale_Benchmark_v3_final_anonymous.json")
DEFAULT_KB = os.path.join(_repo_root(), "scripts",
                          "authoritative_kb_3d_100.csv")


def _normalize_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def load_kb_typical_len_cm(csv_path: str) -> Dict[str, float]:
    m: Dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _normalize_name(row.get("category_name", ""))
            if not name:
                continue
            try:
                v = float(row.get("typical_len_cm", "") or "nan")
            except Exception:
                continue
            if v > 0 and name not in m:
                m[name] = v
    return m


def load_task2_rows(benchmark_json: str) -> List[Dict[str, Any]]:
    with open(benchmark_json, encoding="utf-8") as f:
        data = json.load(f)
    return [x for x in data if str(x.get("task_id", "")).startswith("T2_")]


def read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                obj = json.loads(text[start: end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    raise ValueError(f"Gemini returned non-JSON text: {text[:500]}...")


def _parse_first_json_object_from_gemini_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Gemini sometimes returns JSON + extra prose. Parse only the first JSON object.
    """
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    start = t.find("{")
    if start < 0:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _end = dec.raw_decode(t[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _gemini_json_call_raw(
    api_key: str,
    model: str,
    prompt: str,
    image_b64: str,
    mime_type: str,
    timeout_s: int,
    temperature: float,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """
    Minimal Gemini call that expects a JSON object response (no schema validation).
    Used for scene prefilter.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": 512,
            # Do not set response_mime_type here: some Gemini endpoints reject unknown config
            # fields, causing 400 and making prefilter always fail-open.
        },
    }
    last_err: Optional[Exception] = None
    last_http_status: Optional[int] = None
    last_http_text: str = ""
    for req_try in range(5):
        backoff = min(60.0, 2.0 ** req_try)
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout_s,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_http_status = int(resp.status_code)
                last_http_text = (resp.text or "")[:2000]
                if verbose:
                    print(
                        f"      [prefilter retry] HTTP {resp.status_code} (try={req_try+1}/5), sleep {backoff:.1f}s",
                        flush=True,
                    )
                time.sleep(backoff)
                continue
            if resp.status_code != 200:
                last_http_status = int(resp.status_code)
                last_http_text = (resp.text or "")[:2000]
                last_err = RuntimeError(f"prefilter API Error {resp.status_code}: {last_http_text}")
                break
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(
                p.get("text", "")
                for p in parts
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ).strip()
            obj = _parse_first_json_object_from_gemini_text(text)
            return obj if isinstance(obj, dict) else None
        except (ReadTimeout, RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            if verbose:
                print(
                    f"      [prefilter retry] {type(e).__name__} (try={req_try+1}/5), sleep {backoff:.1f}s",
                    flush=True,
                )
            time.sleep(backoff)
            continue
    # Print failure details for debugging (do not save into output JSON).
    msg = f"      [prefilter FAILED] status={last_http_status} exc={repr(last_err)}"
    print(msg[:2000], flush=True)
    if last_http_text:
        print(f"      [prefilter FAILED body] {last_http_text[:400]}", flush=True)
    return None


def screen_scene_for_size_eval(
    api_key: str,
    model: str,
    image_b64: str,
    mime_type: str,
    primary_object_names: List[str],
    timeout_s: int,
    temperature: float,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """
    Scene-level prefilter to skip visibly flawed images that harm size judging.
    Mirrors the inference/backfill prefilter semantics.
    """
    names = [str(x).strip() for x in (primary_object_names or []) if str(x).strip()]
    names_csv = ", ".join(names[:10]) if names else "(unspecified)"
    prompt = f"""You audit a synthetic image BEFORE an automated object size-evaluation pipeline.

Prominent labels from our benchmark (reference only): {names_csv}.

The list may include evaluator synonyms for the same physical object (e.g. short vs parenthesized names).
Treat those as ONE intended label set — do not count them as separate extra clutter.

From the image alone, output ONE JSON object with:
- "duplicate_objects" (bool): two+ clearly separate instances of the SAME category so it is ambiguous which to judge (e.g. two identical eggs, two rulers).
- "extra_unnamed_objects" (bool): major extra props/clutter/repeated shapes beyond the intended label set so relative-scale reasoning is unreliable (not mere synonyms in the list above).
- "severe_generation_artifacts" (bool): obvious AI flaws (fused objects, melted geometry, incoherent boundaries, extra limbs) that would break size reasoning.
- "objects_individually_clear" (bool): each listed label could be matched to a distinct instance with usable boundaries; false if blur/heavy overlap/crop blocks that.

Set "skip_size_correction" (bool) true if ANY of: duplicate_objects, extra_unnamed_objects, severe_generation_artifacts, OR objects_individually_clear is false.
Add "brief_reason" (string, <= 35 words, English).

JSON only, no markdown."""
    res = _gemini_json_call_raw(
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_b64=image_b64,
        mime_type=mime_type,
        timeout_s=timeout_s,
        temperature=temperature,
        verbose=verbose,
    )
    if not res or not isinstance(res, dict):
        return None

    def _as_bool(v: Any, default: bool = False) -> bool:
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


def build_task2_prompt(
    object_a: str,
    len_a_cm: float,
    object_b: str,
    len_b_cm: float,
    scenario: str = "",
    benchmark_prompt: str = "",
) -> str:
    ratio = (float(len_b_cm) / float(len_a_cm)) if float(len_a_cm) > 0 else 0.0
    inv_ratio = (float(len_a_cm) / float(len_b_cm)) if float(len_b_cm) > 0 else 0.0
    benchmark_block = ""
    bp = (benchmark_prompt or "").strip()
    if bp:
        benchmark_block = (
            "### OPTIONAL GENERATION PROMPT (LOW PRIORITY):\n"
            "Human annotators did NOT see this text; they only saw the image, object names, and reference lengths. "
            "Use this prompt only to identify the intended product, packaging/bundle extent, or ambiguous interaction. "
            "Do NOT use exact centimeter claims in the prompt as a pixel ruler, and do NOT forgive or penalize a size "
            "relationship solely because the prompt intended it.\n"
            f"{bp}\n\n"
        )

    return (
        "You are an expert physical spatial reasoning engine and a professional photography adjudicator. "
        "Your task is to evaluate the physical size accuracy of objects in the provided generated image.\n"
        "Your scoring should closely mirror the aggregate judgement of human annotators using a quick visual GUI, "
        "not a purely literal pixel-measurement or product-spec calculator.\n\n"
        "### GROUND TRUTH REFERENCE:\n"
        f"- Object A (Human Anchor): {object_a}, Typical physical longest edge: {len_a_cm:.2f} cm.\n"
        f"- Object B (Target Product): {object_b}, Typical physical longest edge: {len_b_cm:.2f} cm.\n"
        f"- Expected 3D longest-edge ratio: Product B is about {ratio:.2f}x the human anchor "
        f"(the anchor is about {inv_ratio:.2f}x Product B).\n\n"
        f"{benchmark_block}"
        "### EVALUATION PROTOCOL (MATCH THE HUMAN GUI):\n"
        "This task focuses on the direct interaction between a human body (or body part) and a product. "
        "Since the human is interacting with the product, they are usually roughly at the SAME depth plane, but catalog photos "
        "often use close-up framing, partial hands/faces/feet, foreshortening, and product-forward composition.\n\n"
        "Please analyze the image step-by-step:\n"
        "1. Detection: Are BOTH Object A (Human/part) and Object B (Product) clearly identifiable? If either is very blurry, missing, fused, or not actually generated, set both_objects_present=false.\n"
        "2. Roles: Object A is the HUMAN ANCHOR. Treat it as correctly scaled. Score ONLY Object B, the TARGET PRODUCT. Do NOT flip roles.\n"
        "3. Human-anchor caution: hands, heads/faces, feet/legs, and full bodies may be cropped, angled, closer to the camera, or only partially visible. "
        "Do not infer exact centimeters from a cropped palm, a close-up face, or a partial foot/leg. Use them as approximate scale references.\n"
        "4. Product extent: Judge the intended product as presented, not a misleading subcomponent. "
        "For packs/bundles/stacks, use the full visible pack/bundle footprint; for folded bedding/clothing, use the folded visible package; "
        "for jewelry in a display box, judge the visible retail presentation as plausible rather than treating the ring diameter alone as the whole target; "
        "for paired products (shoes, gloves, etc.), judge the displayed pair/item as a normal product presentation and do not double-penalize because two units appear.\n"
        "5. Size relationship: Ask whether a typical human annotator would immediately feel the product is implausibly small/large in this interaction. "
        "Do not score a catalog-style close-up as oversized merely because the product occupies many pixels or is foregrounded.\n\n"
        "### FINAL JUDGMENT:\n"
        f"Assume Object A is its real-world physical size in 3D space. How accurate is the size of Object B compared to its stated typical length of {len_b_cm:.2f} cm?\n"
        "Use the same quantitative rubric shown to human annotators, but apply it perceptually rather than with exact pixel calipers:\n"
        "- Score 3 (Proportionate): inferred product size error is within about ±20%, OR the catalog interaction looks plausible after crop/pose/packaging correction.\n"
        "- Score 2 (Slightly undersized): product is clearly 20–60% too small.\n"
        "- Score 4 (Slightly oversized): product is clearly 20–60% too large.\n"
        "- Score 1 (Severely undersized): product is more than 60% too small and looks comically/impossibly tiny.\n"
        "- Score 5 (Severely oversized): product is more than 60% too large and looks comically/impossibly huge.\n"
        "Important Task2 calibration: human annotators usually give Score 3 for plausible product catalog interactions. "
        "Use 4/2 only for obvious size errors, and use 5/1 very rarely. If the only evidence for 5/1 is an exact ratio estimate from a cropped hand/head/foot or a close-up product-forward composition, choose 4/2 or 3 instead.\n"
        "Select exactly one category from the 1-5 scale below:\n"
        "1: Severely Undersized\n"
        "2: Slightly Undersized\n"
        "3: Proportionate\n"
        "4: Slightly Oversized\n"
        "5: Severely Oversized\n\n"
        "### OUTPUT FORMAT:\n"
        "You MUST output your response in valid JSON format. Do not include markdown code blocks.\n"
        "CRITICAL: Keep BOTH reasoning fields extremely short (<= 25 words each). Do NOT use ellipses (...).\n"
        "{\n"
        '  "reasoning_detection": "...",\n'
        '  "reasoning_scale_and_interaction": "...",\n'
        '  "both_objects_present": true,\n'
        '  "size_score": 3\n'
        "}\n"
    )


def gemini_score_pair(
    api_key: str,
    model: str,
    prompt: str,
    image_bytes: Optional[bytes] = None,
    image_b64: Optional[str] = None,
    mime_type: str = "image/png",
    timeout_s: int = 120,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Robust function to score a pair with clean retry logic."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    if image_b64 is not None:
        b64 = image_b64
    elif image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
    else:
        raise ValueError("gemini_score_pair requires image_bytes or image_b64")

    last_err: Optional[Exception] = None

    for parse_try in range(3):
        max_tokens = 512 if parse_try == 0 else (
            2048 if parse_try == 1 else 4096)
        extra_constraint = (
            "\n\nCRITICAL: Output MUST be minified JSON in a SINGLE LINE. "
            "The JSON MUST include all required keys and end with a closing brace '}'. "
            "Keep BOTH reasoning fields <= 25 words each."
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt + extra_constraint},
                        {"inlineData": {"mimeType": mime_type, "data": b64}},
                    ]
                }
            ],
            "generationConfig": {
                # For stability we default to deterministic decoding.
                "temperature": float(temperature),
                "maxOutputTokens": max_tokens,
            },
        }

        resp = None
        for req_try in range(5):
            backoff = min(60.0, 2.0 ** req_try)
            try:
                resp = requests.post(url, json=payload, headers={
                                     "Content-Type": "application/json"}, timeout=timeout_s)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if verbose:
                        print(
                            f"      [retry] HTTP {resp.status_code} (req_try={req_try+1}/5, parse_try={parse_try+1}/3), "
                            f"sleep {backoff:.1f}s",
                            flush=True,
                        )
                    time.sleep(backoff)
                    continue
                # Exit request loop on success or 4xx error (non-retryable)
                break
            except (ReadTimeout, RequestException) as e:
                last_err = e
                if verbose:
                    print(
                        f"      [retry] {type(e).__name__} (req_try={req_try+1}/5, parse_try={parse_try+1}/3), "
                        f"sleep {backoff:.1f}s",
                        flush=True,
                    )
                time.sleep(backoff)
                continue

        if resp is None or resp.status_code != 200:
            error_msg = resp.text[:2000] if resp else str(last_err)
            last_err = RuntimeError(
                f"API Error {resp.status_code if resp else 'N/A'}: {error_msg}")
            # Try again with larger token limit (unlikely to help, but maintains loop structure)
            continue

        try:
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if isinstance(
                p, dict) and isinstance(p.get("text"), str)).strip()

            obj = _safe_json_from_text(text)

            # Validate output structure
            both = bool(obj.get("both_objects_present"))
            score = obj.get("size_score", None)

            if both:
                if not isinstance(score, int) or score not in (1, 2, 3, 4, 5):
                    raise ValueError(f"Invalid size_score: {score!r}")
            else:
                score = None  # Ensure score is None if objects are missing

            return {
                "reasoning_detection": str(obj.get("reasoning_detection", "")).strip(),
                "reasoning_scale_and_interaction": str(obj.get("reasoning_scale_and_interaction", "")).strip(),
                "both_objects_present": both,
                "size_score": score,
            }

        except Exception as e:
            last_err = e
            continue  # Retry parsing with larger token limits

    # If all parsing retries exhaust
    raise RuntimeError(
        f"Gemini judge failed to return valid JSON after all retries: {last_err}")


def mean_abs_dev(scores: List[int]) -> Optional[float]:
    if not scores:
        return None
    return float(sum(abs(int(s) - 3) for s in scores) / float(len(scores)))


def load_existing_output_json(path: str) -> List[Dict[str, Any]]:
    """Load prior results if present; used to merge and skip already-scored tasks."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
        if isinstance(prev, list):
            return prev
    except Exception as e:
        print(f"Warning: could not load existing output {path!r} (starting empty): {e}", flush=True)
    return []


def select_rows_by_ranges(
    rows: List[Dict[str, Any]],
    ranges_spec: str,
    task_label: str,
) -> List[Dict[str, Any]]:
    """
    Select rows by 0-based half-open ranges, e.g. "0-100,200-300".
    A single index like "7" means just row 7. Duplicates are removed while preserving order.
    """
    spec = (ranges_spec or "").strip()
    if not spec:
        return rows

    selected_indices: List[int] = []
    seen: set[int] = set()
    n = len(rows)
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            start = int(a_s.strip())
            end = int(b_s.strip())
        else:
            start = int(part)
            end = start + 1
        if start < 0 or end < start or end > n:
            raise SystemExit(
                f"ERROR: invalid --ranges component {part!r}: "
                f"use 0 <= start <= end <= {n} for {task_label} rows."
            )
        for idx in range(start, end):
            if idx not in seen:
                seen.add(idx)
                selected_indices.append(idx)

    return [rows[i] for i in selected_indices]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--kb_csv", default=DEFAULT_KB)
    p.add_argument("--image_dir", required=True,
                   help="Directory containing generated T1_*.png files for one model.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument(
        "--model",
        default="gemini-3.1-flash-preview",
        help="Gemini model id for VLM judging.",
    )
    # Stability controls
    p.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Gemini generation temperature (e.g. 0.3–0.5 for self-consistency, 0.0 for fully deterministic).",
    )
    p.add_argument(
        "--samples_per_pair",
        type=int,
        default=1,
        help="How many independent Gemini judgements per object pair (>=1). "
        "Scores are aggregated by median; reasoning comes from the last sample.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help=(
            "Seconds to sleep after each Gemini API call (request pacing). "
            "Default 0 for maximum throughput; increase if you see 429s. "
            "See active limits: https://aistudio.google.com/rate-limit"
        ),
    )
    p.add_argument("--timeout_s", type=int, default=120, help="Per-request timeout (seconds) for Gemini API.")
    p.add_argument("--verbose", action="store_true", help="Print per-pair progress and retry logs.")
    p.add_argument(
        "--disable-prefilter",
        action="store_true",
        help=(
            "Disable scene-level prefilter. By default, this script runs a lightweight Gemini audit per image "
            "to skip obviously flawed generations (duplicates/artifacts/unclear objects) and records it in output JSON."
        ),
    )
    p.add_argument(
        "--backfill-prefilter",
        action="store_true",
        help=(
            "If a task_id already exists in --out and already has score results, only run scene prefilter and "
            "backfill the 'scene_prefilter' field (instead of skipping). "
            "Does nothing if that existing record already contains 'scene_prefilter'."
        ),
    )
    p.add_argument(
        "--start",
        type=int,
        default=None,
        help="First Task2 row index to score (0-based, inclusive). Default: 0 (beginning of benchmark list).",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="End row index (exclusive, same semantics as Python slice rows[start:end]). "
        "Default: len(benchmark Task2 rows) (score all).",
    )
    p.add_argument(
        "--ranges",
        type=str,
        default="",
        help=(
            "Comma-separated 0-based half-open row ranges to score, e.g. "
            "'0-100,200-300'. A single index like '7' scores only row 7. "
            "If provided, overrides --start/--end."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated: ignored. If --out already exists, it is always loaded and merged (existing tasks are skipped).",
    )
    p.add_argument(
        "--force-rescore",
        action="store_true",
        help="If a task_id already exists in --out, remove it and score again (replaces that entry).",
    )
    p.add_argument(
        "--pair-workers",
        type=int,
        default=6,
        help=(
            "Parallel Gemini requests per image (different object pairs). "
            "Benchmark Task2 has at most 6 pairs per image; default 6 uses full intra-image parallelism. "
            "Capped by the number of pairs in that image."
        ),
    )
    p.add_argument(
        "--image-workers",
        type=int,
        default=3,
        help=(
            "How many images (tasks) to score concurrently. "
            "Important when each image has only 1 pair — use >1 here or throughput stays ~1 API call at a time. "
            "If you hit 429, lower this or raise --sleep."
        ),
    )
    args = p.parse_args()

    print("evaluate_genscale_task2_gemini: starting (loading KB and benchmark next)...", flush=True)

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: GOOGLE_API_KEY is empty.")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"  KB: {args.kb_csv}", flush=True)
    kb = load_kb_typical_len_cm(args.kb_csv)
    print(f"  Benchmark: {args.benchmark}", flush=True)
    rows = load_task2_rows(args.benchmark)
    n_all = len(rows)
    if str(args.ranges or "").strip():
        rows = select_rows_by_ranges(rows, str(args.ranges), "Task2")
        print(f"Scoring Task2 row ranges {args.ranges!r} ({len(rows)} of {n_all} total).", flush=True)
    else:
        start = 0 if args.start is None else int(args.start)
        end = n_all if args.end is None else int(args.end)
        if start < 0 or end < start or end > n_all:
            raise SystemExit(
                f"ERROR: invalid --start/--end range: start={start} end={end} "
                f"(benchmark has {n_all} Task2 rows; use 0 <= start <= end <= {n_all})."
            )
        rows = rows[start:end]
        if start != 0 or end != n_all:
            print(f"Scoring Task2 rows [{start}:{end}] ({len(rows)} of {n_all} total).", flush=True)

    print(
        f"  Loading existing output (for resume/skip): {args.out!r} — can take a while on NFS if the file is large.",
        flush=True,
    )
    out_rows: List[Dict[str, Any]] = load_existing_output_json(args.out)
    if out_rows:
        print(
            f"Merged with existing output: {len(out_rows)} record(s) in {args.out!r} (same task_id will be skipped unless --force-rescore).",
            flush=True,
        )
    if args.resume:
        print(
            "Note: --resume is deprecated; existing --out is always merged when the file exists.",
            flush=True,
        )

    existing_tids = {str(r.get("task_id", "")) for r in out_rows if r.get("task_id")}
    tid_to_out_index: Dict[str, int] = {}
    for idx, r in enumerate(out_rows):
        tid = str(r.get("task_id", "") or "")
        if tid and tid not in tid_to_out_index:
            tid_to_out_index[tid] = idx

    def persist() -> None:
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.out)

    # Phase 1: sync validation + skip; collect rows that need Gemini.
    print(
        f"  Phase 1: scanning {len(rows)} benchmark row(s) (image files on NFS may be slow; no API yet)...",
        flush=True,
    )
    tasks_to_score: List[Tuple[int, Dict[str, Any]]] = []
    tasks_to_prefilter_only: List[Tuple[int, Dict[str, Any]]] = []
    for i, row in enumerate(rows):
        tid = str(row.get("task_id", ""))
        if not tid:
            print(f"[{i+1}/{len(rows)}] skip row with empty task_id", flush=True)
            continue

        if tid in existing_tids:
            if not args.force_rescore:
                if bool(args.backfill_prefilter):
                    out_idx = tid_to_out_index.get(tid)
                    existing_rec = out_rows[out_idx] if out_idx is not None else None
                    sp = (existing_rec or {}).get("scene_prefilter") if isinstance(existing_rec, dict) else None
                    needs_pf = False
                    if not (isinstance(existing_rec, dict) and ("scene_prefilter" in existing_rec)):
                        needs_pf = True
                    elif isinstance(sp, dict) and sp.get("api_failed") is True:
                        needs_pf = True
                    if needs_pf:
                        tasks_to_prefilter_only.append((i, row))
                        print(
                            f"[{i+1}/{len(rows)}] queue prefilter-only backfill for {tid} (already scored; missing/failed scene_prefilter)",
                            flush=True,
                        )
                        continue
                print(f"[{i+1}/{len(rows)}] skip {tid} (already in output file)", flush=True)
                continue
            out_rows[:] = [r for r in out_rows if str(r.get("task_id", "")) != tid]
            existing_tids.discard(tid)
            tid_to_out_index.pop(tid, None)

        img_path = os.path.join(args.image_dir, f"{tid}.png")
        if not os.path.isfile(img_path):
            out_rows.append({"task_id": tid, "error": "missing_image",
                            "pairs": [], "image_mean_abs_dev": None})
            existing_tids.add(tid)
            persist()
            continue

        objects = row.get("objects_included", [])
        if not (isinstance(objects, list) and len(objects) >= 2):
            out_rows.append({"task_id": tid, "error": "invalid_objects",
                            "pairs": [], "image_mean_abs_dev": None})
            existing_tids.add(tid)
            persist()
            continue

        # Task 2 uses product_scale and fallback logic inside score_one_image, so we bypass kb check here.

        tasks_to_score.append((i, row))

    print(
        f"  Phase 1 done: {len(tasks_to_prefilter_only)} image(s) queued for prefilter-only backfill; "
        f"{len(tasks_to_score)} image(s) queued for Gemini scoring; starting Phase 2...",
        flush=True,
    )

    image_workers = max(1, int(args.image_workers))
    io_lock = threading.Lock()

    def run_prefilter_only(bench_i: int, row: Dict[str, Any]) -> Dict[str, Any]:
        """Run scene prefilter for an already-scored record and return the scene_prefilter dict."""
        tid = str(row.get("task_id", ""))
        img_path = os.path.join(args.image_dir, f"{tid}.png")
        objects = row.get("objects_included", [])
        scene_prefilter: Dict[str, Any] = {
            "prefilter_enabled": not bool(args.disable_prefilter),
            "skip_size_correction": False,
            "duplicate_objects": None,
            "extra_unnamed_objects": None,
            "severe_generation_artifacts": None,
            "objects_individually_clear": None,
            "brief_reason": "",
            "prefilter_primary_labels": list(objects) if isinstance(objects, list) else [],
            "prefilter_model": str(args.model),
            "prefilter_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "api_failed": False if bool(args.disable_prefilter) else True,
        }
        if bool(args.disable_prefilter):
            return {"task_id": tid, "scene_prefilter": scene_prefilter, "error": None}
        if not os.path.isfile(img_path):
            return {"task_id": tid, "scene_prefilter": scene_prefilter, "error": "missing_image_for_prefilter"}
        img_bytes = read_image_bytes(img_path)
        image_b64_cache = base64.b64encode(img_bytes).decode("utf-8")
        pf = screen_scene_for_size_eval(
            api_key=api_key,
            model=str(args.model),
            image_b64=image_b64_cache,
            mime_type="image/png",
            primary_object_names=list(objects) if isinstance(objects, list) else [],
            timeout_s=int(args.timeout_s),
            temperature=0.1,
            verbose=bool(args.verbose),
        )
        if pf is not None:
            scene_prefilter.update(pf)
            scene_prefilter["api_failed"] = False
        return {"task_id": tid, "scene_prefilter": scene_prefilter, "error": None}


    def score_one_image(
        bench_i: int,
        row: Dict[str, Any],
        more_tasks_after: bool,
    ) -> Dict[str, Any]:
        """Load image, run all pair judges, return one output record."""
        tid = str(row.get("task_id", ""))
        img_path = os.path.join(args.image_dir, f"{tid}.png")
        img_bytes = read_image_bytes(img_path)
        objects = row.get("objects_included", [])
        
        # In Task2, we only have one pair per image: product and human part
        if len(objects) != 2:
            return {"task_id": tid, "error": "Task2 requires exactly 2 objects"}
            
        obj0 = objects[0]
        obj1 = objects[1]
        if "human" in obj0.lower():
            human_name, prod_name = obj0, obj1
        elif "human" in obj1.lower():
            human_name, prod_name = obj1, obj0
        else:
            # fallback
            prod_name, human_name = obj0, obj1
        
        product_scale = row.get("product_scale", {})
        prod_len = product_scale.get("typical_len_cm")
        
        if not prod_len:
            # Fallback if product_scale not present
            prod_len = kb.get(_normalize_name(prod_name), 0)
            
        gt_ratios = row.get("gt_ratios", {})
        ratio_key = f"{prod_name}_to_{human_name}"
        target_ratio = gt_ratios.get(ratio_key, {}).get("target_ratio")
        
        if not target_ratio:
            # Reverse key maybe?
            alt_key = f"{human_name}_to_{prod_name}"
            target_ratio_alt = gt_ratios.get(alt_key, {}).get("target_ratio")
            if target_ratio_alt:
                target_ratio = 1.0 / target_ratio_alt
        
        if prod_len and target_ratio:
            human_len = prod_len / target_ratio
        else:
            # Fallback to kb
            human_len = kb.get(_normalize_name(human_name), 170.0) # Assume 170.0 for full body
            if not prod_len:
                prod_len = human_len * (target_ratio or 1.0)
                
        obj_data = [
            {"raw_name": human_name, "len_cm": human_len},
            {"raw_name": prod_name, "len_cm": prod_len}
        ]
        
        # We only have 1 combo for Task2
        combos = [(obj_data[0], obj_data[1])]
        pair_workers = 1
        image_b64_cache: str = base64.b64encode(img_bytes).decode("utf-8")
        mime_type = "image/png"

        scene_prefilter: Dict[str, Any] = {
            "prefilter_enabled": not bool(args.disable_prefilter),
            "skip_size_correction": False,
            "duplicate_objects": None,
            "extra_unnamed_objects": None,
            "severe_generation_artifacts": None,
            "objects_individually_clear": None,
            "brief_reason": "",
            "prefilter_primary_labels": list(objects) if isinstance(objects, list) else [],
            "prefilter_model": str(args.model),
            "prefilter_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "api_failed": False if bool(args.disable_prefilter) else True,
        }
        if not args.disable_prefilter:
            pf = screen_scene_for_size_eval(
                api_key=api_key,
                model=str(args.model),
                image_b64=image_b64_cache,
                mime_type=mime_type,
                primary_object_names=list(objects) if isinstance(objects, list) else [],
                timeout_s=int(args.timeout_s),
                temperature=0.1,  # keep prefilter stable regardless of judge temperature
                verbose=bool(args.verbose),
            )
            if pf is not None:
                scene_prefilter.update(pf)
                scene_prefilter["api_failed"] = False
            # Fail-open: if prefilter API fails, continue scoring.
            if scene_prefilter.get("skip_size_correction") is True:
                return {
                    "task_id": tid,
                    "image_path": img_path,
                    "scene_prefilter": scene_prefilter,
                    "error": "prefilter_skip_size_correction",
                    "pairs": [],
                    "image_mean_abs_dev": None,
                    "meta": {
                        "gemini_model": args.model,
                        "pair_workers": 0,
                        "image_workers": image_workers,
                    },
                }

        scenario_str = str(row.get("scenario") or "")

        def score_combo(k: int, obj_a: Dict[str, Any], obj_b: Dict[str, Any]) -> Dict[str, Any]:

            prompt = build_task2_prompt(
                obj_a["raw_name"],
                obj_a["len_cm"],
                obj_b["raw_name"],
                obj_b["len_cm"],
                scenario=scenario_str,
                benchmark_prompt=str(row.get("prompt") or ""),
            )
            local_scores: List[int] = []
            all_results: List[Dict[str, Any]] = []
            last_result: Optional[Dict[str, Any]] = None
            n_samples = max(1, int(args.samples_per_pair))
            sleep_s = float(args.sleep)
            try:
                if args.verbose:
                    print(
                        f"   Pair [{k+1}/{len(combos)}] {tid}: {obj_a['raw_name']} vs {obj_b['raw_name']}",
                        flush=True,
                    )
                def _run_sample(s_idx: int) -> Dict[str, Any]:
                    return gemini_score_pair(
                        api_key=api_key,
                        model=args.model,
                        prompt=prompt,
                        image_bytes=None,
                        image_b64=image_b64_cache,
                        timeout_s=int(args.timeout_s),
                        temperature=float(args.temperature),
                        verbose=bool(args.verbose),
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=n_samples) as sample_ex:
                    sample_futures = [sample_ex.submit(_run_sample, s_idx) for s_idx in range(n_samples)]
                    for fut in concurrent.futures.as_completed(sample_futures):
                        judge_result = fut.result()
                        all_results.append(judge_result)
                        last_result = judge_result
                        if (
                            judge_result.get("both_objects_present", False)
                            and judge_result.get("size_score") is not None
                        ):
                            local_scores.append(int(judge_result["size_score"]))

                agg_score: Optional[int] = None
                if local_scores:
                    from statistics import median, multimode
                    modes = multimode(local_scores)
                    if len(modes) == 1:
                        agg_score = modes[0]
                    else:
                        agg_score = round(median(local_scores))

                if last_result is not None:
                    # Aggregate booleans using majority vote
                    bop_votes = [r.get("both_objects_present", False) for r in all_results]
                    dup_votes = [r.get("duplicate_objects", False) for r in all_results]
                    euo_votes = [r.get("extra_unnamed_objects", False) for r in all_results]
                    
                    agg_bop = sum(bop_votes) >= (len(bop_votes) / 2.0)
                    agg_dup = sum(dup_votes) >= (len(dup_votes) / 2.0)
                    agg_euo = sum(euo_votes) >= (len(euo_votes) / 2.0)
                    
                    rec: Dict[str, Any] = {
                        "object_a": {
                            "name": obj_a["raw_name"],
                            "typical_len_cm": obj_a["len_cm"],
                        },
                        "object_b": {
                            "name": obj_b["raw_name"],
                            "typical_len_cm": obj_b["len_cm"],
                        },
                        **last_result,
                    }
                    
                    # Override with aggregated values
                    rec["both_objects_present"] = agg_bop
                    if "duplicate_objects" in last_result:
                        rec["duplicate_objects"] = agg_dup
                    if "extra_unnamed_objects" in last_result:
                        rec["extra_unnamed_objects"] = agg_euo
                        
                    rec["all_size_scores"] = local_scores
                    
                    if agg_score is not None:
                        rec["size_score"] = agg_score
                    elif not agg_bop:
                        rec["size_score"] = None
                        
                    return rec
                return {
                    "object_a": {"name": obj_a["raw_name"]},
                    "object_b": {"name": obj_b["raw_name"]},
                    "error": "judge_failed: no successful response",
                }
            except Exception as e:
                return {
                    "object_a": {"name": obj_a["raw_name"]},
                    "object_b": {"name": obj_b["raw_name"]},
                    "error": f"judge_failed: {e}",
                }

        pairs_results: List[Dict[str, Any]] = []
        sleep_s = float(args.sleep)

        if pair_workers <= 1:
            for k, (obj_a, obj_b) in enumerate(combos):
                pairs_results.append(score_combo(k, obj_a, obj_b))
                if sleep_s > 0.0 and (k < len(combos) - 1 or more_tasks_after):
                    time.sleep(sleep_s)
        else:
            def _run(k: int, obj_a: Dict[str, Any], obj_b: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                return (k, score_combo(k, obj_a, obj_b))

            with concurrent.futures.ThreadPoolExecutor(max_workers=pair_workers) as ex:
                futures = [
                    ex.submit(_run, k, obj_a, obj_b)
                    for k, (obj_a, obj_b) in enumerate(combos)
                ]
                by_k: Dict[int, Dict[str, Any]] = {}
                for fut in concurrent.futures.as_completed(futures):
                    k, rec = fut.result()
                    by_k[k] = rec
            pairs_results = [by_k[k] for k in range(len(combos))]
            if sleep_s > 0.0 and more_tasks_after:
                time.sleep(sleep_s)

        ok_scores: List[int] = []
        for rec in pairs_results:
            if rec.get("error"):
                continue
            if rec.get("both_objects_present") and rec.get("size_score") is not None:
                ok_scores.append(int(rec["size_score"]))

        img_score = mean_abs_dev(ok_scores)
        return {
            "task_id": tid,
            "image_path": img_path,
            "scene_prefilter": scene_prefilter,
            "pairs": pairs_results,
            "image_mean_abs_dev": img_score,
            "meta": {
                "gemini_model": args.model,
                "pair_workers": pair_workers,
                "image_workers": image_workers,
            },
        }

    # Phase 2: Gemini (optionally parallel across images).
    if tasks_to_prefilter_only:
        print(
            f"  Phase 2a: backfilling scene_prefilter for {len(tasks_to_prefilter_only)} already-scored image(s)...",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=image_workers) as ex:
            futures = [ex.submit(run_prefilter_only, bench_i, row) for (bench_i, row) in tasks_to_prefilter_only]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                tid = str(res.get("task_id", ""))
                sp = res.get("scene_prefilter")
                err = res.get("error")
                with io_lock:
                    out_idx = tid_to_out_index.get(tid)
                    if out_idx is None:
                        continue
                    if isinstance(out_rows[out_idx], dict):
                        out_rows[out_idx]["scene_prefilter"] = sp
                        if err:
                            out_rows[out_idx].setdefault("prefilter_backfill_error", str(err))
                        persist()
                if err:
                    print(f"prefilter-only {tid}: {err}", flush=True)
                else:
                    print(f"prefilter-only backfilled {tid}", flush=True)

    n_tasks = len(tasks_to_score)
    if n_tasks == 0:
        print("No Task2 images to score (all skipped or filtered).", flush=True)
    elif image_workers <= 1:
        for j, (bench_i, row) in enumerate(tasks_to_score):
            more = j < n_tasks - 1
            out_rec = score_one_image(bench_i, row, more)
            tid = str(out_rec["task_id"])
            out_rows.append(out_rec)
            existing_tids.add(tid)
            persist()
            print(
                f"[{bench_i+1}/{len(rows)}] scored {tid} -> mean_abs_dev={out_rec['image_mean_abs_dev']}",
                flush=True,
            )
    else:
        def _run_image(j: int, bench_i: int, row: Dict[str, Any]) -> Dict[str, Any]:
            more = j < n_tasks - 1
            return score_one_image(bench_i, row, more)

        with concurrent.futures.ThreadPoolExecutor(max_workers=image_workers) as ex:
            futures = [
                ex.submit(_run_image, j, bench_i, row)
                for j, (bench_i, row) in enumerate(tasks_to_score)
            ]
            for fut in concurrent.futures.as_completed(futures):
                out_rec = fut.result()
                tid = str(out_rec["task_id"])
                with io_lock:
                    out_rows.append(out_rec)
                    existing_tids.add(tid)
                    persist()
                print(
                    f"scored {tid} -> mean_abs_dev={out_rec['image_mean_abs_dev']}",
                    flush=True,
                )

    print(f"Done. Saved: {args.out} ({len(out_rows)} total record(s))")


if __name__ == "__main__":
    main()
