import pandas as pd
import random
import json
import time
import itertools
import requests
import os
import argparse
import math
import base64
import re
from typing import List, Dict, Any, Optional, Tuple, Set


def _float_field(row: Dict[str, Any], key: str) -> float:
    try:
        v = float(row.get(key))
        if math.isnan(v):
            return float("nan")
        return v
    except (TypeError, ValueError):
        return float("nan")


def format_product_scale_block_for_task2(product: Dict[str, Any]) -> str:
    """
    Compact, evaluation-aligned physical priors from ABO-style CSV rows.
    Emits characteristic length plus every available axis (L/W/H) so shape/aspect is explicit, not only a single scalar.
    """
    typ = _float_field(product, "typical_len_cm")
    mn = _float_field(product, "min_len_cm")
    mx = _float_field(product, "max_len_cm")

    lines: List[str] = [
        "AUTHORITATIVE PRODUCT SCALE (from dataset metadata — the benchmark’s ground-truth size uses these; "
        "the generated scene must match this order of magnitude, not an improvised guess from the title alone):",
    ]
    if not math.isnan(typ):
        lines.append(f"- Primary scale cue (characteristic / longest span): ~{typ:.2f} cm.")

    axis_labels = (
        ("length", "length_cm"),
        ("width", "width_cm"),
        ("height", "height_cm"),
    )
    axis_parts: List[str] = []
    present_vals: List[float] = []
    for label, key in axis_labels:
        v = _float_field(product, key)
        if math.isnan(v):
            continue
        axis_parts.append(f"{label}={v:.2f} cm")
        present_vals.append(v)
    if axis_parts:
        lines.append(
            "- Axis-aligned bounding box from metadata (report each axis that exists; use together for bulk + aspect ratio): "
            + "; ".join(axis_parts) + "."
        )
        lines.append(
            "  Near-zero values along an axis usually mean very thin / sheet-like / flexible extent in that direction, not missing data."
        )
        if len(present_vals) >= 2:
            s = sorted(present_vals, reverse=True)
            lines.append(
                f"- Extent spread (largest→smallest axis in this row): "
                f"{', '.join(f'{x:.2f} cm' for x in s)} — use to infer elongation vs flat vs compact form."
            )

    if not math.isnan(mn) and not math.isnan(mx) and mx >= mn > 0:
        lines.append(f"- Metadata span for overall size (min–max of characteristic length): {mn:.2f}–{mx:.2f} cm.")
    if len(lines) == 1:
        lines.append(
            "- Numeric dimensions are incomplete; infer a plausible real-world size consistent with the cleaned description, "
            "but prefer conservative, catalog-realistic proportions."
        )
    return "\n    ".join(lines)

class GeminiAPI:
    def __init__(self, api_key=None, model_version="gemini-3.1-pro-preview"):
        """
        使用 requests 封装的 Gemini API 调用类。
        推荐使用 gemini-1.5-flash，速度快且逻辑能力强。
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set the environment variable or pass it directly.")
        self.model_version = model_version
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_version}:generateContent?key={self.api_key}"

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """纯文本生成接口"""
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(self.url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return None
                
            result = response.json()
            # 从 JSON 结构中提取生成的文本
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
            except (KeyError, IndexError):
                print("Unexpected API response structure.")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return None

    def generate_json_with_image(
        self,
        prompt: str,
        image_path: str,
        *,
        model_override: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 180,
    ) -> Optional[Dict[str, Any]]:
        """Vision + JSON. Uses responseMimeType application/json when supported."""
        model = model_override or self.model_version
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime = (
            "image/png"
            if ext == ".png"
            else "image/webp"
            if ext == ".webp"
            else "image/jpeg"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": mime, "data": b64}}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "responseMimeType": "application/json",
            },
        }
        req_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        try:
            response = requests.post(
                req_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as e:
            print(f"[visual_qc] request error: {e}")
            return None

        if response.status_code != 200:
            print(f"[visual_qc] HTTP {response.status_code}: {response.text[:300]}")
            return None
        try:
            result = response.json()
            text = ""
            for cand in result.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if "text" in part:
                        text += part["text"]
            text = (text or "").strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if text.startswith("```"):
                    text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text).strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return None
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[visual_qc] parse error: {e}")
            return None


# Task3 pulls Task1 rows only when benchmark num_objects==2 (_valid). QC should flag duplicate
# instances of the same category (two rabbits, two rings), not “wrong object count” otherwise.


def _task1_visual_qc_prompt(objects_included: Optional[List[str]]) -> str:
    """
    Task1: exactly two categories in the benchmark — expect one salient instance per category.
    Primary failure mode to catch: duplicate instances of the same category (model ignored prompt).
    """
    names = [str(x).strip() for x in (objects_included or []) if str(x).strip()]
    if len(names) >= 2:
        a, b = names[0], names[1]
        cat_line = (
            f"The prompt intended EXACTLY ONE instance of each category: “{a}” and “{b}” "
            f"(two different object types, one of each)."
        )
    else:
        cat_line = (
            "The scene should depict exactly two different object categories from the benchmark, "
            "with one salient instance per category."
        )

    return f"""Screen this image for a two-object size-ratio benchmark.

{cat_line}

PRIMARY check (be strict on this only):
- REJECT (acceptable=false) if you clearly see DUPLICATE instances of the SAME category when only one should appear — e.g. two rabbits, two identical rings side by side, two duplicate main objects of the same type. Symmetric pairs of small props or reflections are not duplicates unless they read as two full separate objects of that category.

DO NOT REJECT for:
- Exactly two distinct object types visible (even if small imperfections, mild blur, texture noise, or perspective).
- Extra background clutter, people, or scenery that are not a second copy of the listed category.
- “Three objects” in the sense of background + two main subjects — the benchmark only cares that there is not a duplicate of either main category.

SECONDARY (only if severe): extreme blur / fog so both categories are unidentifiable, or gross melted/unreadable synthesis.

When unsure about duplicates, prefer acceptable=true.

Return JSON only:
{{"acceptable": <boolean>, "reject_tags": [<short string>], "comment": <under 120 characters>}}
"""

TASK3_VISUAL_QC_PROMPT_T2 = """Screen this image for a PRODUCT + HUMAN BODY PART catalog-style size benchmark.

Set acceptable=true unless CLEARLY broken:
- REJECT if: strong fake portrait blur / heavy bokeh so the product or the body part is not identifiable for scale; OR the product or the human anchor is essentially missing/covered; OR gross synthesis failure (melted product, extra limbs, unreadable chaos).

DO NOT REJECT for: mild texture noise, gentle depth, fingers touching the product, or small catalog-style imperfections.

When unsure, choose acceptable=true.

Return JSON only:
{"acceptable": <boolean>, "reject_tags": [<short string>], "comment": <under 120 characters>}
"""


def _visual_qc_prompt_for_label(
    label: str,
    benchmark_row: Optional[Dict[str, Any]] = None,
) -> str:
    if (label or "").strip().upper().startswith("T2"):
        return TASK3_VISUAL_QC_PROMPT_T2
    objs: Optional[List[str]] = None
    if benchmark_row:
        oi = benchmark_row.get("objects_included")
        if isinstance(oi, list) and len(oi) >= 2:
            objs = [str(x) for x in oi[:2]]
    return _task1_visual_qc_prompt(objs)


def _gemini_visual_qc_passes(
    image_path: str,
    gemini: GeminiAPI,
    qc_model: str,
    task_label: str = "T1",
    benchmark_row: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    True = include image in Task3 pool. False = reject (severe issue).
    On API/parse failure, keep the image (True) with a warning to avoid dropping good data.
    """
    prompt = _visual_qc_prompt_for_label(task_label, benchmark_row=benchmark_row)
    js = gemini.generate_json_with_image(
        prompt,
        image_path,
        model_override=qc_model,
        temperature=0.05,
        timeout=180,
    )
    if js is None:
        print(f"      [visual_qc] API/parse failed for {image_path} — keeping image.", flush=True)
        return True
    ok = js.get("acceptable")
    if ok is True:
        return True
    if ok is False:
        tags = js.get("reject_tags") or []
        cmt = js.get("comment", "")
        print(f"      [visual_qc] reject: {image_path} tags={tags} comment={cmt!r}", flush=True)
        return False
    print(f"      [visual_qc] ambiguous acceptable={ok!r} for {os.path.basename(image_path)} — keeping.", flush=True)
    return True


def _select_pool_with_visual_qc(
    pool: List[Dict[str, Any]],
    image_root: str,
    n_select: int,
    gemini: GeminiAPI,
    qc_model: str,
    label: str,
    sleep_s: float = 0.2,
    benchmark_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Scan pool in order; keep entries that pass QC until n_select or pool exhausted."""
    if n_select <= 0:
        return []
    out: List[Dict[str, Any]] = []
    for entry in pool:
        if len(out) >= n_select:
            break
        tid = str(entry.get("task_id", "") or "").strip()
        path = _resolve_image_path(image_root, tid)
        if not os.path.isfile(path):
            print(f"   [{label}] skip {tid}: file not found ({path})", flush=True)
            continue
        bm_row = benchmark_map.get(tid) if benchmark_map else None
        if _gemini_visual_qc_passes(
            path, gemini, qc_model, task_label=label, benchmark_row=bm_row
        ):
            out.append(entry)
        if sleep_s > 0:
            time.sleep(float(sleep_s))
    return out


# ==========================================
# 2. 数据加载与预处理 (Bucketing)
# ==========================================
def load_data():
    print("Loading databases...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_common = pd.read_csv(os.path.join(script_dir, "authoritative_kb_3d_100.csv"))
    common_objects = df_common.to_dict('records')
    
    df_products = pd.read_csv(os.path.join(script_dir, "abo_local_sampled_1000_representative.csv"))
    products = df_products.to_dict('records')
    
    return common_objects, products


def is_human_body_category_name(category_name: str) -> bool:
    """True for authoritative_kb entries like 'human face', 'human hand', 'human foot' (Task2-style anchors)."""
    s = (category_name or "").strip()
    return bool(s) and s.lower().startswith("human ")


def filter_task1_common_objects_excluding_human_parts(
    common_objects: List[Dict],
) -> List[Dict]:
    """Task1 should not mix human anatomy categories; those are reserved for Task2."""
    return [o for o in common_objects if not is_human_body_category_name(str(o.get("category_name", "")))]


def get_combinations(common_objects: List[Dict], min_disparity: float = 4.0, max_disparity: float = 20.0):
    """
    Task 1 核心：按概率抽取 2-4 个物体，并进行物理悬殊比例的【上下限双重过滤】。
    For Task1, pass a pool already filtered with filter_task1_common_objects_excluding_human_parts.
    """
    rand = random.random()
    if rand < 0.5:
        num_objs = 2
    elif rand < 0.8:
        num_objs = 3
    else:
        num_objs = 4

    candidates = random.sample(common_objects, num_objs)
    lengths = [obj['typical_len_cm'] for obj in candidates]

    # 计算最大物体与最小物体的差距
    disparity = max(lengths) / min(lengths)

    # 【核心修改】：差距太小（<4倍）没挑战性，差距太大（>20倍）没法画，全部丢弃
    if disparity < min_disparity or disparity > max_disparity:
        return None

    return candidates


def generate_task1_prompt(gemini: GeminiAPI, objects: List[Dict], scenario_type: str = "same_plane") -> str:
    """Task 1: 生成罕见但物理上可行的场景，支持同一平面与自然深度双轨制"""
    obj_names = [obj['category_name'] for obj in objects]
    names_str = ", ".join(obj_names)

    # 根据 scenario_type 动态调整系统指令
    if scenario_type == "same_plane":
        depth_rule = "The objects MUST be placed roughly on the SAME DEPTH PLANE (e.g., sitting side-by-side on the ground) to avoid perspective distortion."
    else:
        # 新增的自然场景规则
        depth_rule = "The objects should be placed naturally within a 3D space, allowing for NATURAL DEPTH, occlusion, and perspective (e.g., one object slightly in front of another). The background MUST remain clean and uncluttered so the objects are clearly visible."

    system_instruction = f"""
    You are an expert prompt engineer for text-to-image models.
    I need to generate a photorealistic image containing the following objects: [{names_str}].
    
    CRITICAL RULES:
    1. Emphasize UNCOMMON JUXTAPOSITION. Do NOT try to force them into a mundane, everyday scene. Instead, place them in a neutral, vast environment (e.g., an empty massive concrete parking lot, a giant photography studio, a desert floor) where their extreme size contrast is the main focal point.
    2. {depth_rule}
    3. If it is physically impossible to place them together in reality (e.g., fitting a bus inside a microwave), output EXACTLY the word "REJECT".
    4. If physically possible to arrange, output a 1-2 sentence descriptive prompt emphasizing their size contrast. Keep it under 40 words.
    
    DO NOT append any extra constraints, I will add them programmatically.
    """

    text = gemini.generate_text(system_instruction)
    if text == "REJECT" or (text and "REJECT" in text):
        return None
    return text

def generate_task2_prompt(gemini: GeminiAPI, product: Dict) -> Dict[str, Any]:
    """Task 2: 商品与智能人体锚点交互"""
    prod_name = product['category_name']
    prod_len = float(product['typical_len_cm'])
    
    if prod_len < 30.0:
        anchor_name, anchor_len = "human hand", 19.3
    elif prod_len < 60.0:
        anchor_name, anchor_len = "human face/head", 24.0
    elif prod_len < 100.0:
        anchor_name, anchor_len = "human foot/leg", 26.5
    else:
        anchor_name, anchor_len = "full human body", 170.0  # 标准成年人身高

    scale_block = format_product_scale_block_for_task2(product)
        
    system_instruction = f"""
    {scale_block}

    Write a single prompt for a text-to-image model: a {anchor_name} interacting with the product described below.
    The anchor ({anchor_name}) is fixed for metric consistency; you must still describe a **natural, purpose-consistent** interaction—do not treat the body part as arbitrary scale furniture only.

    Raw listing title (for your cleanup only; do not paste SEO noise into the final prompt): '{prod_name}'.

    CRITICAL RULES:
    1. Clean up the product name. Remove SEO spam, redundant brands, and truncated boilerplate; keep a short, concrete noun phrase for the actual object (e.g., reduce a long mattress listing title to “a black metal bed frame”).
    2. Physical scale: you MUST respect the AUTHORITATIVE PRODUCT SCALE above, including characteristic length and every axis extent listed (length/width/height). Fold them into the image prompt in natural language so the generator gets real-world bulk and aspect ratio—e.g. a compact ~12×6×4 cm object vs a ~180×80×0.2 cm sheet—not only a single scalar like “~10 cm”; do not omit these cues or substitute a different order of magnitude.
    3. Interaction (usage + physics): Depict a photorealistic pose where the {anchor_name} engages the product in a way that matches **typical real-world use** implied by the cleaned object (e.g., hands typing or resting near a keyboard, holding a mug by the handle, face near a compact mirror, foot or leg next to a large appliance or furniture for scale, full body beside very large items). Avoid absurd or purely decorative arrangements that ignore what the product is for. Contact and posture must stay physically plausible given the stated dimensions.
    4. Visibility / identifiability (advertising-style clarity): compose like a premium catalog or hero shot. The product must remain easy to recognize—its overall shape and most of its visible silhouette should be unobstructed. Hands or skin may touch or support the item, but avoid layouts where fingers, palm, limbs, or accessories hide most of the product’s projected area (e.g., do not wrap a fist around a small device so that only a sliver shows). Mild partial occlusion consistent with normal use is fine; large-area occlusion that would block a clean product crop is not.
    5. Camera / perspective: favor a fairly orthographic or mild-perspective view (roughly front or three-quarter, not extreme wide-angle) so apparent 2D size ratios between the product and the {anchor_name} stay interpretable.
    Output ONLY the final image prompt text, nothing else.
    """
    
    text = gemini.generate_text(system_instruction)
    if not text:
        return None
        
    return {
        "prompt": text,
        "anchor_name": anchor_name,
        "anchor_len": anchor_len
    }

# ==========================================
# 4. Pipeline 执行引擎
# ==========================================


def build_benchmark(
    num_task1_same_plane=0,
    num_task1_natural=0,
    num_task2=300,
    output_file: str = "GenScale_Benchmark_v3_Task2.json",
):
    gemini = GeminiAPI(api_key=os.environ.get("GOOGLE_API_KEY"))
    common_objects, products = load_data()
    common_objects_t1 = filter_task1_common_objects_excluding_human_parts(common_objects)
    n_h = len(common_objects) - len(common_objects_t1)
    if n_h:
        print(
            f"Task1: excluded {n_h} human-body category row(s) from sampling pool "
            f"(e.g. human face/hand/foot); pool size {len(common_objects_t1)}."
        )
    if len(common_objects_t1) < 4:
        raise RuntimeError(
            "Need at least 4 non-human categories in authoritative_kb for Task1; "
            f"got {len(common_objects_t1)} after excluding human parts."
        )

    benchmark_dataset = []

    t1_global_count = 0
    if num_task1_same_plane + num_task1_natural > 0:
        print("\n🚀 Starting Task 1 Generation (Dual-Track)...")
    else:
        print("\n⏭️  Skipping Task 1 (0 items requested).")

    # 定义两种子任务的数量分配
    # scenario_id 与 GenScale_Benchmark_v3_Task1.json 一致：S2=同平面极端尺度，S1=自然景深
    t1_tasks = [
        {"type": "same_plane", "target_count": num_task1_same_plane,
            "scenario_id": "S2_Extreme_Contrast"},
        {"type": "natural_depth", "target_count": num_task1_natural,
            "scenario_id": "S1_Natural_Depth"},
    ]

    for task_config in t1_tasks:
        scenario_type = task_config["type"]
        target_count = task_config["target_count"]
        scenario_id = task_config["scenario_id"]
        if target_count <= 0:
            continue

        print(
            f"\n   -> Generating subset: {scenario_id} (Target: {target_count})")

        current_count = 0
        while current_count < target_count:
            candidates = get_combinations(
                common_objects_t1, min_disparity=4.0, max_disparity=20.0)
            if not candidates:
                continue

            scene_prompt = generate_task1_prompt(
                gemini, candidates, scenario_type=scenario_type)
            if not scene_prompt:
                continue

            # 【核心修改】：前置物理约束，并根据场景类型分配后缀
            if scenario_type == "same_plane":
                final_prompt = f"Strictly accurate real-world physical proportions, exact same depth plane. {scene_prompt} Photorealistic."
            else:
                final_prompt = f"Strictly accurate real-world physical proportions, natural depth and perspective. {scene_prompt} Photorealistic."

            gt_ratios = {}
            for objA, objB in itertools.combinations(candidates, 2):
                nameA, nameB = objA['category_name'], objB['category_name']
                ratio_typ = objA['typical_len_cm'] / objB['typical_len_cm']
                ratio_min = objA['min_len_cm'] / objB['max_len_cm']
                ratio_max = objA['max_len_cm'] / objB['min_len_cm']

                gt_ratios[f"{nameA}_to_{nameB}"] = {
                    "target_ratio": round(ratio_typ, 3),
                    "acceptable_range": [round(ratio_min, 3), round(ratio_max, 3)]
                }

            benchmark_dataset.append({
                "task_id": f"T1_{t1_global_count:04d}",
                "scenario": scenario_id,
                "num_objects": len(candidates),
                "objects_included": [c['category_name'] for c in candidates],
                "prompt": final_prompt,
                "gt_ratios": gt_ratios,
                "reference_image_path": None
            })
            current_count += 1
            t1_global_count += 1
            print(
                f"      ✅ Generated {current_count}/{target_count} ({scenario_id})")
            time.sleep(1.5)

    print("\n🚀 Starting Task 2 Generation...")
    t2_count = 0
    random.shuffle(products)
    
    for prod in products:
        if t2_count >= num_task2: break
            
        result = generate_task2_prompt(gemini, prod)
        if not result: continue
            
        final_prompt = (
            result["prompt"]
            + " Photorealistic, strictly accurate real-world physical proportions between the human body part and the product; "
            "catalog-style framing so the product remains largely identifiable and unobstructed."
        )
        
        prod_len = prod['typical_len_cm']
        prod_min, prod_max = prod['min_len_cm'], prod['max_len_cm']
        anchor_len = result['anchor_len']
        anchor_min, anchor_max = anchor_len * 0.95, anchor_len * 1.05
        
        ratio_typ = prod_len / anchor_len
        ratio_min = prod_min / anchor_max
        ratio_max = prod_max / anchor_min
        
        benchmark_dataset.append({
            "task_id": f"T2_{t2_count:04d}",
            "scenario": "S2_Human_Product_Anchor",
            "num_objects": 2,
            "objects_included": [prod['category_name'], result['anchor_name']],
            "prompt": final_prompt,
            "gt_ratios": {
                f"{prod['category_name']}_to_{result['anchor_name']}": {
                    "target_ratio": round(ratio_typ, 3),
                    "acceptable_range": [round(ratio_min, 3), round(ratio_max, 3)]
                }
            },
            "reference_image_path": prod.get('image_path', None),
        })
        t2_count += 1
        print(f"✅ T2: Generated {t2_count}/{num_task2}")
        time.sleep(1.5)

    out_path = output_file
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dataset, f, indent=4, ensure_ascii=False)
    print(f"\n🎉 Done. Wrote {len(benchmark_dataset)} benchmark row(s) to {out_path}")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_object_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _load_eval_details_list(path: str) -> List[Dict[str, Any]]:
    """evaluate_genscale evaluation_report.json uses {\"details\": [...]}; Gemini exports may be a bare list."""
    raw = _load_json(path)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        d = raw.get("details")
        if isinstance(d, list):
            return d
    return []


def _identity_from_eval_pair(pair: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    pk = pair.get("pair")
    if not pk or "_to_" not in str(pk):
        return None
    a, b = str(pk).split("_to_", 1)
    na = _normalize_object_name(a)
    nb = _normalize_object_name(b)
    if not na or not nb:
        return None
    return tuple(sorted([na, nb]))


def _identity_from_gemini_pair(gpair: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    oa = gpair.get("object_a") or {}
    ob = gpair.get("object_b") or {}
    na = _normalize_object_name(str(oa.get("name", "")))
    nb = _normalize_object_name(str(ob.get("name", "")))
    if not na or not nb:
        return None
    return tuple(sorted([na, nb]))


def _gemini_size_lookup_from_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, Tuple[str, str]], int]:
    """(task_id, sorted name identity) -> discrete size_score (e.g. 1–5)."""
    lookup: Dict[Tuple[str, Tuple[str, str]], int] = {}
    for row in rows:
        tid = str(row.get("task_id", "") or "").strip()
        if not tid:
            continue
        for gpair in row.get("pairs") or []:
            ident = _identity_from_gemini_pair(gpair)
            if ident is None:
                continue
            sc = gpair.get("size_score")
            if sc is None:
                continue
            try:
                lookup[(tid, ident)] = int(sc)
            except (TypeError, ValueError):
                continue
    return lookup


def _enrich_eval_pairs_with_gemini_scores(
    details: List[Dict[str, Any]],
    lookup: Dict[Tuple[str, Tuple[str, str]], int],
) -> None:
    for entry in details:
        tid = str(entry.get("task_id", "") or "").strip()
        for p in entry.get("pairs") or []:
            ident = _identity_from_eval_pair(p)
            if ident is None:
                continue
            key = (tid, ident)
            if key in lookup:
                p["gemini_size_score"] = lookup[key]


def _pair_discrete_size_score(pair: Dict[str, Any]) -> Optional[int]:
    """Prefer merged Gemini field; else inline size_score on the pair."""
    for k in ("gemini_size_score", "size_score"):
        v = pair.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return None


def _non3_pairs(
    entry: Dict[str, Any],
    perfect_score: int,
) -> List[Dict[str, Any]]:
    """Pairs usable for Task3 when selecting 'not perfectly scaled' (default: size_score != 3)."""
    out: List[Dict[str, Any]] = []
    for p in entry.get("pairs") or []:
        if "_to_" not in str(p.get("pair", "")):
            continue
        sc = _pair_discrete_size_score(p)
        if sc is None:
            continue
        if sc != perfect_score:
            out.append(p)
    return out


def _pick_pair_for_task3(
    entry: Dict[str, Any],
    task3_select_mode: str,
    rng: random.Random,
    perfect_score: int,
) -> Dict[str, Any]:
    if task3_select_mode == "eval_worst_first":
        return _pair_from_entry(entry)
    cands = _non3_pairs(entry, perfect_score)
    if cands:
        return rng.choice(cands)
    return _pair_from_entry(entry)


def _load_human_aggregated_task_ids(path: str) -> Set[str]:
    """
    All task_id strings present in aggregated human JSON (one row per image that survived
    human QC). Used to drop eval rows for images humans rejected / did not aggregate.
    """
    raw = _load_json(path)
    if isinstance(raw, dict):
        raw = raw.get("records") or raw.get("details") or []
    if not isinstance(raw, list):
        return set()
    out: Set[str] = set()
    for rec in raw:
        tid = str(rec.get("task_id") or "").strip()
        if tid:
            out.add(tid)
    return out


def _load_human_aggregated_by_task_id(path: str) -> Dict[str, Dict[str, float]]:
    """
    Load scripts/eval/human_scores/aggregated_human_scores.json (list of records).

    Returns task_id -> {"mean": m, "min": n} over pair size_score values (1..3 rubric).
    Lower scores = worse perceived size → same direction as ascending sort on eval image_avg_score
    when selecting "needs correction" cases for Task3.
    """
    raw = _load_json(path)
    if isinstance(raw, dict):
        raw = raw.get("records") or raw.get("details") or []
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for rec in raw:
        tid = str(rec.get("task_id") or "").strip()
        if not tid:
            continue
        pairs = rec.get("pairs") or []
        scores: List[float] = []
        for p in pairs:
            v = p.get("size_score")
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                scores.append(float(v))
        if not scores:
            continue
        out[tid] = {
            "mean": float(sum(scores) / len(scores)),
            "min": float(min(scores)),
        }
    return out


def _sort_key_task3(
    entry: Dict[str, Any],
    human_by_tid: Dict[str, Dict[str, float]],
    rank_by: str,
) -> Tuple[float, ...]:
    """
    Sort key tuple; lower = selected first (worse perceived / worse automatic score).
    """
    tid = str(entry.get("task_id", ""))
    eval_avg = float(entry.get("image_avg_score", 1.0))
    h = human_by_tid.get(tid)
    h_mean = h["mean"] if h else float("nan")
    h_min = h["min"] if h else float("nan")

    if rank_by == "eval_image_avg":
        return (eval_avg,)
    if rank_by == "human_mean":
        return (h_mean if h is not None else 1e9,)
    if rank_by == "human_min":
        return (h_min if h is not None else 1e9,)
    if rank_by == "human_then_eval":
        # Primary: human mean (missing human → deprioritize). Secondary: automatic eval.
        return (
            h_mean if h is not None else 1e9,
            eval_avg,
        )
    raise ValueError(f"Unknown rank_by: {rank_by}")


def _resolve_image_path(image_root: str, task_id: str) -> str:
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        p = os.path.join(image_root, f"{task_id}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(image_root, f"{task_id}.png")


def _pair_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    pairs = entry.get("pairs", [])
    if not pairs:
        return {}
    for pair in pairs:
        if "_to_" in str(pair.get("pair", "")):
            return pair
    return pairs[0]


def _compute_precise_plan(pair_key: str, generated_ratio: float, target_ratio: float) -> Dict[str, Any]:
    obj_a, obj_b = pair_key.split("_to_")
    # generated_ratio = size(A) / size(B)
    # STRICT RULE: keep GT larger object fixed, edit GT smaller object only.
    # If target_ratio >= 1, A is GT larger and B is GT smaller -> edit B.
    # If target_ratio < 1,  B is GT larger and A is GT smaller -> edit A.
    if target_ratio >= 1.0:
        larger = obj_a
        smaller = obj_b
        # target = A / (B * s) => s = generated / target
        scale_factor = generated_ratio / max(target_ratio, 1e-6)
    else:
        larger = obj_b
        smaller = obj_a
        # target = (A * s) / B => s = target / generated
        scale_factor = target_ratio / max(generated_ratio, 1e-6)
    return {
        "smaller_object": smaller,
        "larger_object": larger,
        "scale_factor": float(scale_factor),
        "direction": "enlarge" if scale_factor >= 1.0 else "shrink",
    }


# Skip Task3 rows when uncorrected ratio (from generated_ratio vs target_ratio only) is extreme — bad data.
TASK3_RAW_RATIO_MAX = 5.0
TASK3_RAW_RATIO_MIN = 0.2

# Manually flagged bad source images (same task_id strings as in eval JSON). Add IDs here as you find them.
TASK3_MANUAL_SKIP_TASK_IDS = frozenset({
    "T2_0070",
})

# Align with Task1/2 ``scenario`` field in merged benchmark JSON.
TASK3_SCENARIO_HARD = "S4_Hard_Auto_Discovery"
TASK3_SCENARIO_PRECISE = "S5_Precise_Scale_Instruction"


def _raw_correction_scale_factor(generated_ratio: float, target_ratio: float) -> float:
    """
    Uncorrected multiplicative factor needed to match target (same formula as _compute_precise_plan).
    Eval JSON may clamp scale_factor into tiers; this value is NOT clamped and is used only for QC.
    """
    if target_ratio >= 1.0:
        return generated_ratio / max(target_ratio, 1e-6)
    return target_ratio / max(generated_ratio, 1e-6)


def _plan_from_eval_or_compute(
    pair: Dict[str, Any],
    pair_key: str,
    generated_ratio: float,
    target_ratio: float,
) -> Optional[Dict[str, Any]]:
    """
    First check uncorrected ratio from generated_ratio / target_ratio (same math as _compute_precise_plan).
    If outside [TASK3_RAW_RATIO_MIN, TASK3_RAW_RATIO_MAX], skip (flawed data).
    Otherwise prefer tier-clamped scale_factor + objects from evaluate_genscale_gemini; else fall back
    to ratio-derived _compute_precise_plan.
    """
    raw_sf = _raw_correction_scale_factor(generated_ratio, target_ratio)
    if raw_sf > TASK3_RAW_RATIO_MAX or raw_sf < TASK3_RAW_RATIO_MIN:
        return None

    if pair.get("used_gemini_ratio_plan") and pair.get("scale_factor") is not None:
        try:
            sf = float(pair["scale_factor"])
        except (TypeError, ValueError):
            sf = None
        if sf is not None:
            et = pair.get("edit_target_object")
            ref = pair.get("reference_object")
            if et and ref:
                direction = str(
                    pair.get("scale_direction")
                    or ("enlarge" if sf >= 1.0 else "shrink")
                )
                return {
                    "scale_factor": sf,
                    "direction": direction,
                    "keep_unchanged": str(ref).strip(),
                    "edit_only": str(et).strip(),
                }

    plan = _compute_precise_plan(pair_key, generated_ratio, target_ratio)
    return {
        "scale_factor": float(plan["scale_factor"]),
        "direction": str(plan["direction"]),
        "keep_unchanged": str(plan["larger_object"]),
        "edit_only": str(plan["smaller_object"]),
    }


def _hard_prompt() -> str:
    return (
        "Check whether object size proportions in this image are unrealistic. "
        "If there is a proportion error, automatically correct only the necessary object size(s) to make the scene physically plausible. "
        "Preserve object identity, pose, viewpoint, composition, background, and lighting. "
        "If proportions are already correct, keep the image unchanged."
    )


def _task2_product_human_names_from_benchmark(bm: Dict[str, Any]) -> Tuple[str, str]:
    """Benchmark gt_ratios keys are ``{product}_to_{human_anchor}``."""
    gt = bm.get("gt_ratios") or {}
    for pk in gt.keys():
        s = str(pk)
        if "_to_" in s:
            a, b = s.split("_to_", 1)
            return a.strip(), b.strip()
    oi = bm.get("objects_included") or []
    if len(oi) >= 2:
        return str(oi[0]).strip(), str(oi[1]).strip()
    return "", ""


def _human_anchor_typical_len_cm(anchor_name: str) -> float:
    """Approximate characteristic linear size (cm) for Task2 anchor labels (same order of magnitude as Task2 generation)."""
    n = (anchor_name or "").lower()
    if "full" in n and "body" in n:
        return 170.0
    if "foot" in n or "leg" in n:
        return 26.5
    if "head" in n or "face" in n:
        return 24.0
    if "hand" in n:
        return 19.3
    return 19.3


def _task2_compact_product_ref_cm(ps: Dict[str, Any]) -> str:
    """One short line: typical length + optional L×W×H from benchmark product_scale."""
    typ = _float_field(ps, "typical_len_cm")
    if math.isnan(typ):
        return ""
    bits: List[str] = [f"~{typ:.2f} cm (characteristic)"]
    L, W, H = _float_field(ps, "length_cm"), _float_field(ps, "width_cm"), _float_field(ps, "height_cm")
    if not math.isnan(L) and not math.isnan(W) and not math.isnan(H):
        bits.append(f"L×W×H {L:.1f}×{W:.1f}×{H:.1f} cm")
    return ", ".join(bits)


def _hard_prompt_task2(bm: Dict[str, Any]) -> str:
    """Short Task3 hard prompt: same idea as _hard_prompt, plus named product + anchor reference lengths (cm)."""
    product_name, human_name = _task2_product_human_names_from_benchmark(bm)
    ps = bm.get("product_scale")
    if not product_name and isinstance(ps, dict):
        cn = str(ps.get("category_name") or "").strip()
        if cn:
            product_name = cn.split(",")[0].strip()
            if len(product_name) > 80:
                product_name = product_name[:77] + "..."
    pn = product_name or "the product"

    hcm = _human_anchor_typical_len_cm(human_name) if human_name else 19.3
    anchor_s = f'"{human_name}" ~{hcm:.1f} cm' if human_name else f"anchor ~{hcm:.1f} cm"

    if isinstance(ps, dict) and ps:
        prod_s = _task2_compact_product_ref_cm(ps)
    else:
        prod_s = ""
    if not prod_s:
        prod_s = "infer a catalog-realistic size from the scene"

    return (
        "Check whether object size proportions in this image are unrealistic. "
        f'Reference lengths — "{pn}": {prod_s}; human part: {anchor_s}. '
        f'If wrong relative to these references, rescale "{pn}" only; if already plausible, keep the image unchanged. '
        "Preserve object identity, pose, viewpoint, composition, background, and lighting."
    )


def _fallback_precise_prompt(plan: Dict[str, Any], pair_key: str, target_ratio: float) -> str:
    ku = plan.get("keep_unchanged") or plan.get("larger_object", "")
    eo = plan.get("edit_only") or plan.get("smaller_object", "")
    return (
        f"Strictly follow this edit instruction: keep '{ku}' unchanged, "
        f"and {plan['direction']} only '{eo}' by a factor of {plan['scale_factor']:.3f}. "
        f"Target size ratio '{pair_key}' is {target_ratio:.3f}. "
        "Do not change object identities, positions, background, lighting, or camera viewpoint."
    )


def _gemini_precise_prompt(
    gemini: GeminiAPI,
    plan: Dict[str, Any],
    pair_key: str,
    generated_ratio: float,
    target_ratio: float,
) -> str:
    ku = plan.get("keep_unchanged") or plan.get("larger_object", "")
    eo = plan.get("edit_only") or plan.get("smaller_object", "")
    instruction = f"""
Write one concise image-editing prompt for strict instruction following.

Known values:
- Pair key: {pair_key}
- Current generated ratio: {generated_ratio:.6f}
- Target ratio: {target_ratio:.6f}
- Keep unchanged: {ku}
- Edit only: {eo}
- Operation: {plan['direction']}
- Exact scale factor for the edited object: {plan['scale_factor']:.6f}

Requirements:
1. Mention the exact numeric scale factor.
2. Emphasize preserving identity/background/lighting/composition.
3. No extra edits beyond size correction.
Output only the final prompt text.
"""
    text = gemini.generate_text(instruction, temperature=0.1)
    if not text:
        return _fallback_precise_prompt(plan, pair_key, target_ratio)
    return text.strip()


def build_task3_dataset(
    task1_eval_json: str,
    task2_eval_json: str,
    benchmark_json: str,
    task1_image_root: str,
    task2_image_root: str,
    num_task1_select: int = 80,
    num_task2_select: int = 70,
    output_file: str = "GenScale_Benchmark_v2_Task3.json",
    use_gemini_precise_prompt: bool = True,
    gemini_model_version: str = "gemini-3.1-pro-preview",
    human_scores_json: Optional[str] = None,
    task3_rank_by: str = "eval_image_avg",
    task3_human_whitelist: bool = False,
    task1_gemini_json: Optional[str] = None,
    task2_gemini_json: Optional[str] = None,
    task3_select_mode: str = "gemini_random_not_3",
    task3_seed: int = 42,
    task3_perfect_score: int = 3,
    task3_visual_qc: bool = True,
    task3_qc_model: str = "gemini-2.5-flash",
    task3_qc_sleep_s: float = 0.25,
):
    benchmark = _load_json(benchmark_json)
    if not benchmark:
        v1_fallback = os.path.join(os.path.dirname(benchmark_json), "GenScale_Benchmark_v1.json")
        if os.path.abspath(benchmark_json).endswith("GenScale_Benchmark_v2.json") and os.path.exists(v1_fallback):
            print(f"⚠️ Benchmark file is empty: {benchmark_json}")
            print(f"⚠️ Fallback to: {v1_fallback}")
            benchmark_json = v1_fallback
            benchmark = _load_json(benchmark_json)
    if not benchmark:
        raise ValueError(
            f"Benchmark is empty: {benchmark_json}. "
            "Please provide a non-empty benchmark json via --benchmark_json."
        )
    benchmark_map = {x["task_id"]: x for x in benchmark}
    task1_details = _load_eval_details_list(task1_eval_json)
    task2_details = _load_eval_details_list(task2_eval_json)

    gemini = None
    if use_gemini_precise_prompt:
        gemini = GeminiAPI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            model_version=gemini_model_version,
        )

    def _valid(entry: Dict[str, Any], prefix: str) -> bool:
        tid = str(entry.get("task_id", ""))
        if not tid.startswith(prefix):
            return False
        bm = benchmark_map.get(tid)
        if bm is None:
            return False
        if prefix == "T1_" and int(bm.get("num_objects", 0)) != 2:
            return False
        pairs = entry.get("pairs", [])
        if not pairs:
            return False
        # evaluate_genscale sets missing explicitly; Gemini-only JSON has no ratio fields — reject those.
        for p in pairs:
            if p.get("missing"):
                return False
            if "generated_ratio" not in p or "target_ratio" not in p:
                return False
        return True

    task1_pool = [x for x in task1_details if _valid(x, "T1_")]
    task2_pool = [x for x in task2_details if _valid(x, "T2_")]
    missing_in_benchmark_t1 = sum(1 for x in task1_details if x.get("task_id") not in benchmark_map)
    missing_in_benchmark_t2 = sum(1 for x in task2_details if x.get("task_id") not in benchmark_map)

    if task3_human_whitelist and not human_scores_json:
        raise ValueError(
            "task3_human_whitelist=True requires --human_scores_json "
            "(aggregated human list used as allowlist of task_id)."
        )

    human_allow: Optional[Set[str]] = None
    if task3_human_whitelist and human_scores_json:
        human_allow = _load_human_aggregated_task_ids(human_scores_json)
        n1b, n2b = len(task1_pool), len(task2_pool)
        task1_pool = [x for x in task1_pool if str(x.get("task_id", "")) in human_allow]
        task2_pool = [x for x in task2_pool if str(x.get("task_id", "")) in human_allow]
        print(
            f"   Human aggregate allowlist: {len(human_allow)} task_id(s) in file; "
            f"T1 pool {n1b} → {len(task1_pool)}; T2 pool {n2b} → {len(task2_pool)} "
            f"(ranking still uses eval image_avg_score unless --task3_rank_by selects human).",
            flush=True,
        )

    human_by_tid: Dict[str, Dict[str, float]] = {}
    if human_scores_json:
        human_by_tid = _load_human_aggregated_by_task_id(human_scores_json)
    elif task3_rank_by != "eval_image_avg":
        raise ValueError(
            f"task3_rank_by={task3_rank_by!r} requires --human_scores_json "
            "(aggregated human scores by task_id)."
        )

    score_lookup: Dict[Tuple[str, Tuple[str, str]], int] = {}
    if task1_gemini_json:
        score_lookup.update(_gemini_size_lookup_from_rows(_load_eval_details_list(task1_gemini_json)))
    if task2_gemini_json:
        score_lookup.update(_gemini_size_lookup_from_rows(_load_eval_details_list(task2_gemini_json)))
    _enrich_eval_pairs_with_gemini_scores(task1_pool, score_lookup)
    _enrich_eval_pairs_with_gemini_scores(task2_pool, score_lookup)
    if score_lookup:
        print(
            f"   Merged Gemini size_score onto eval pairs ({len(score_lookup)} lookup key(s)).",
            flush=True,
        )

    rng = random.Random(int(task3_seed))

    def _pool_sort(pool: List[Dict[str, Any]]) -> None:
        pool.sort(key=lambda e: _sort_key_task3(e, human_by_tid, task3_rank_by))

    if task3_select_mode == "gemini_random_not_3":
        if task3_rank_by != "eval_image_avg":
            print(
                f"   Note: task3_select_mode=gemini_random_not_3 ignores --task3_rank_by ({task3_rank_by!r}).",
                flush=True,
            )
        ps = int(task3_perfect_score)
        t1_non3 = [e for e in task1_pool if _non3_pairs(e, ps)]
        t2_non3 = [e for e in task2_pool if _non3_pairs(e, ps)]
        print(
            f"   task3_select_mode=gemini_random_not_3: T1 {len(t1_non3)}/{len(task1_pool)} images "
            f"with ≥1 pair size_score≠{ps}; T2 {len(t2_non3)}/{len(task2_pool)}.",
            flush=True,
        )
        if num_task1_select > 0 and not task1_pool:
            raise ValueError(
                "Task1 pool is empty before size_score filtering. "
                "Use --task1_eval_json pointing to evaluate_genscale evaluation_report.json "
                "(must contain pairs with generated_ratio/target_ratio), not gemini_scores_*.json alone. "
                "Put Gemini rubric scores in --task1_gemini_json."
            )
        if num_task2_select > 0 and not task2_pool:
            raise ValueError(
                "Task2 pool is empty before size_score filtering. "
                "Use --task2_eval_json pointing to evaluate_genscale evaluation_report.json; "
                "put Gemini scores in --task2_gemini_json."
            )
        if num_task1_select > 0 and not t1_non3:
            raise ValueError(
                "No Task1 eval rows left after filtering to pairs with discrete size_score "
                f"≠ {ps}. Provide --task1_gemini_json (or merge size_score into eval JSON pairs) "
                "and ensure name matching vs evaluate_genscale pair keys."
            )
        if num_task2_select > 0 and not t2_non3:
            raise ValueError(
                "No Task2 eval rows left after filtering to pairs with discrete size_score "
                f"≠ {ps}. Provide --task2_gemini_json (or merge size_score into eval JSON pairs)."
            )
        task1_pool = t1_non3
        task2_pool = t2_non3
        rng.shuffle(task1_pool)
        rng.shuffle(task2_pool)
    else:
        _pool_sort(task1_pool)
        _pool_sort(task2_pool)

    if human_by_tid and task3_rank_by != "eval_image_avg" and task3_select_mode != "gemini_random_not_3":
        miss_h1 = sum(1 for x in task1_pool if str(x.get("task_id", "")) not in human_by_tid)
        miss_h2 = sum(1 for x in task2_pool if str(x.get("task_id", "")) not in human_by_tid)
        print(
            f"   Human score coverage: T1 pool {len(task1_pool) - miss_h1}/{len(task1_pool)} "
            f"with human rows; T2 pool {len(task2_pool) - miss_h2}/{len(task2_pool)} "
            f"(missing → deprioritized when ranking by human).",
            flush=True,
        )

    gemini_qc: Optional[GeminiAPI] = None
    if task3_visual_qc:
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            raise ValueError(
                "task3_visual_qc=True requires GOOGLE_API_KEY for Gemini image screening, "
                "or pass --no_task3_visual_qc to skip."
            )
        gemini_qc = GeminiAPI(api_key=os.environ.get("GOOGLE_API_KEY"), model_version=task3_qc_model)
        print(
            f"   Task3 visual QC: ON (model={task3_qc_model}). "
            f"Scanning pools until {num_task1_select} T1 + {num_task2_select} T2 pass.",
            flush=True,
        )
    else:
        print("   Task3 visual QC: OFF (using pool order only).", flush=True)

    if task3_visual_qc and gemini_qc is not None:
        selected_t1 = _select_pool_with_visual_qc(
            task1_pool,
            task1_image_root,
            num_task1_select,
            gemini_qc,
            task3_qc_model,
            "T1",
            sleep_s=task3_qc_sleep_s,
            benchmark_map=benchmark_map,
        )
        selected_t2 = _select_pool_with_visual_qc(
            task2_pool,
            task2_image_root,
            num_task2_select,
            gemini_qc,
            task3_qc_model,
            "T2",
            sleep_s=task3_qc_sleep_s,
            benchmark_map=benchmark_map,
        )
    else:
        selected_t1 = task1_pool[:num_task1_select]
        selected_t2 = task2_pool[:num_task2_select]
    if len(selected_t1) < num_task1_select:
        print(
            f"   ⚠️ T1: only {len(selected_t1)} image(s) available (requested {num_task1_select}).",
            flush=True,
        )
    if len(selected_t2) < num_task2_select:
        print(
            f"   ⚠️ T2: only {len(selected_t2)} image(s) available (requested {num_task2_select}); "
            f"pool size after filters was {len(task2_pool)}.",
            flush=True,
        )

    dataset = []
    idx = 0
    processed_images = 0
    total_images = len(selected_t1) + len(selected_t2)
    skipped_extreme_scale = 0
    skipped_manual = 0

    def _append_entry(entry: Dict[str, Any], source_task_type: str, image_root: str):
        nonlocal idx, processed_images, skipped_extreme_scale, skipped_manual
        tid = entry["task_id"]
        if tid in TASK3_MANUAL_SKIP_TASK_IDS:
            skipped_manual += 1
            print(f"   [skip] {tid}: TASK3_MANUAL_SKIP_TASK_IDS", flush=True)
            return
        bm = benchmark_map[tid]
        pair = _pick_pair_for_task3(entry, task3_select_mode, rng, int(task3_perfect_score))
        pair_key = str(pair.get("pair", ""))
        if "_to_" not in pair_key:
            return

        generated_ratio = float(pair.get("generated_ratio"))
        target_ratio = float(pair.get("target_ratio"))
        plan = _plan_from_eval_or_compute(pair, pair_key, generated_ratio, target_ratio)
        if plan is None:
            skipped_extreme_scale += 1
            print(
                f"   [skip] {tid} {pair_key}: unusable scale "
                f"(raw ratio outside [{TASK3_RAW_RATIO_MIN}, {TASK3_RAW_RATIO_MAX}], "
                f"or missing eval plan after QC)",
                flush=True,
            )
            return

        image_path = _resolve_image_path(image_root, tid)
        if not os.path.exists(image_path):
            return

        gss = _pair_discrete_size_score(pair)
        edit_plan_out = {
            "edit_target_object": plan["edit_only"],
            "reference_object": plan["keep_unchanged"],
            "scale_factor": float(plan["scale_factor"]),
            "direction": plan["direction"],
        }
        if source_task_type == "T2":
            hard_p = _hard_prompt_task2(bm)
        else:
            hard_p = _hard_prompt()
        dataset.append({
            "task_id": f"T3_{idx:04d}",
            "scenario": TASK3_SCENARIO_HARD,
            "source_task_id": tid,
            "source_task_type": source_task_type,
            "prompt_type": "hard_auto_discovery",
            "image_path": image_path,
            "objects_included": bm["objects_included"],
            "gt_ratios": bm["gt_ratios"],
            "prompt": hard_p,
            "reference_image_path": bm.get("reference_image_path"),
            **({"source_size_score": gss} if gss is not None else {}),
        })
        idx += 1
        processed_images += 1
        if processed_images % 5 == 0 or processed_images == total_images:
            print(f"   Progress: {processed_images}/{total_images} images -> {len(dataset)} rows", flush=True)

        if gemini is None:
            prompt_precise = _fallback_precise_prompt(plan, pair_key, target_ratio)
        else:
            prompt_precise = _gemini_precise_prompt(
                gemini=gemini,
                plan=plan,
                pair_key=pair_key,
                generated_ratio=generated_ratio,
                target_ratio=target_ratio,
            )

        dataset.append({
            "task_id": f"T3_{idx:04d}",
            "scenario": TASK3_SCENARIO_PRECISE,
            "source_task_id": tid,
            "source_task_type": source_task_type,
            "prompt_type": "precise_scale_instruction",
            "image_path": image_path,
            "objects_included": bm["objects_included"],
            "gt_ratios": bm["gt_ratios"],
            "prompt": prompt_precise,
            "edit_plan": edit_plan_out,
            "pair_key": pair_key,
            "generated_ratio": round(generated_ratio, 6),
            "target_ratio": round(target_ratio, 6),
            "reference_image_path": bm.get("reference_image_path"),
            **({"source_size_score": gss} if gss is not None else {}),
        })
        idx += 1

    for e in selected_t1:
        _append_entry(e, "T1", task1_image_root)
    for e in selected_t2:
        _append_entry(e, "T2", task2_image_root)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("\n✅ Task3 dataset generated.")
    print(f"   - Visual QC: {'ON' if task3_visual_qc else 'OFF'}" + (f" ({task3_qc_model})" if task3_visual_qc else ""))
    print(f"   - Select mode: {task3_select_mode}" + (f" (seed={task3_seed}, perfect_score={task3_perfect_score})" if task3_select_mode == "gemini_random_not_3" else ""))
    if task3_select_mode == "eval_worst_first":
        print(f"   - Rank by: {task3_rank_by}" + (f" (human file: {human_scores_json})" if human_scores_json else ""))
    if task3_human_whitelist:
        print("   - Human aggregate allowlist: ON (eval/Gemini ranking only among those task_ids)")
    print(f"   - Selected from T1: {len(selected_t1)} / requested {num_task1_select}")
    print(f"   - Selected from T2: {len(selected_t2)} / requested {num_task2_select}")
    print(f"   - T1 eval entries: {len(task1_details)} (missing benchmark ids: {missing_in_benchmark_t1})")
    print(f"   - T2 eval entries: {len(task2_details)} (missing benchmark ids: {missing_in_benchmark_t2})")
    print(f"   - Total rows (2 prompts per image): {len(dataset)}")
    if skipped_extreme_scale:
        print(
            f"   - Skipped source images (bad/missing scale): {skipped_extreme_scale} "
            f"(raw ratio not in [{TASK3_RAW_RATIO_MIN}, {TASK3_RAW_RATIO_MAX}], or no usable plan)",
            flush=True,
        )
    if skipped_manual:
        print(
            f"   - Skipped source images (manual list): {skipped_manual} "
            f"(TASK3_MANUAL_SKIP_TASK_IDS in generate_benchmark_dataset.py)",
            flush=True,
        )
    print(f"   - Output: {output_file}")


if __name__ == "__main__":
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="task3", choices=["task12", "task3"])
    parser.add_argument("--num_task1_same_plane", type=int, default=12)
    parser.add_argument("--num_task1_natural", type=int, default=11)
    parser.add_argument("--num_task2", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="GenScale_Benchmark_v3_Task1_with_human_parts.json",
        help="Output JSON path for --mode task12 (relative paths are under scripts/).",
    )
    parser.add_argument(
        "--task1_eval_json",
        type=str,
        default=os.path.join(
            _SCRIPT_DIR,
            "eval/task1/eval_reports/Flux_2_Fal/evaluation_report_gemini_ratio_T1_v3_FLUX.json",
        ),
        help="evaluate_genscale (or gemini_ratio) JSON: {\"details\": [...]} with pairs[].generated_ratio/target_ratio/pair/missing.",
    )
    parser.add_argument(
        "--task2_eval_json",
        type=str,
        default=os.path.join(
            _SCRIPT_DIR,
            "eval/task2/eval_reports/Qwen_Image_Edit_2511_1024/evaluation_report_gemini_ratio_T2_v3_Qwen.json",
        ),
        help="Same as --task1_eval_json for Task2.",
    )
    parser.add_argument(
        "--benchmark_json",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "GenScale_Benchmark_v3_Task12_merged_with_product_scales.json"),
    )
    parser.add_argument(
        "--task1_image_root",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "eval/task1/genscale_eval_images_v3_Task1/FLUX_2_Fal"),
    )
    parser.add_argument(
        "--task2_image_root",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "eval/task2/genscale_eval_images_T2_v3/Qwen_Image_Edit_2511_1024"),
    )
    parser.add_argument(
        "--task3_num_from_t1",
        type=int,
        default=80,
        help="After size_score filter + optional visual QC, take this many Task1 images (default 80+70=150 total).",
    )
    parser.add_argument(
        "--task3_num_from_t2",
        type=int,
        default=80,
        help="After size_score filter + optional visual QC, take this many Task2 images.",
    )
    parser.add_argument("--task3_output", type=str, default="scripts/GenScale_Benchmark_v3_Task3.json")
    parser.add_argument(
        "--task1_gemini_json",
        type=str,
        default=os.path.join(
            _SCRIPT_DIR,
            "eval/task1/eval_reports/Flux_2_Fal/gemini_scores_T1_FLUX_2_Fal_flash_v4_5samples.json",
        ),
        help="Gemini VLM judge output (pairs[].size_score 1–5). Merged onto eval pairs for gemini_random_not_3.",
    )
    parser.add_argument(
        "--task2_gemini_json",
        type=str,
        default=os.path.join(
            _SCRIPT_DIR,
            "eval/task2/eval_reports/Qwen_Image_Edit_2511_1024/gemini_scores_T2_Qwen_Image_Edit_2511_1024_flash_v4_5samples.json",
        ),
        help="Same as --task1_gemini_json for Task2.",
    )
    parser.add_argument(
        "--task3_select_mode",
        type=str,
        default="gemini_random_not_3",
        choices=["gemini_random_not_3", "eval_worst_first"],
        help="gemini_random_not_3: among images with ≥1 pair where size_score≠--task3_perfect_score, "
        "shuffle (--task3_seed) then take top-N (avoids picking only the worst-looking failures). "
        "eval_worst_first: sort by image_avg_score ascending (legacy).",
    )
    parser.add_argument(
        "--task3_seed",
        type=int,
        default=42,
        help="RNG seed for gemini_random_not_3 shuffles and per-image pair choice.",
    )
    parser.add_argument(
        "--task3_perfect_score",
        type=int,
        default=3,
        help="Rubric value treated as 'no edit needed' (default 3 on 1–5 scale). Task3 pool keeps pairs with score≠this.",
    )
    parser.add_argument(
        "--human_scores_json",
        type=str,
        default=None,
        help="Aggregated human JSON (list of {task_id, ...}). "
        "Required when --task3_rank_by is not eval_image_avg. "
        "With --task3_human_whitelist, only task_id present in this file are kept (drops human-rejected images); "
        "ranking stays --task3_rank_by (default: eval/Gemini image_avg_score).",
    )
    parser.add_argument(
        "--task3_human_whitelist",
        action="store_true",
        help="Intersect eval pools with task_id set from --human_scores_json. "
        "Use with default --task3_rank_by eval_image_avg to rank by Gemini/eval while excluding human-invalid images. "
        "If the pool is smaller than --task3_num_from_t1/--task3_num_from_t2, all available images are used (see warnings).",
    )
    parser.add_argument(
        "--task3_rank_by",
        type=str,
        default="eval_image_avg",
        choices=[
            "eval_image_avg",
            "human_mean",
            "human_min",
            "human_then_eval",
        ],
        help="How to rank valid eval rows before taking top-N for Task3. "
        "eval_image_avg: ascending image_avg_score from evaluate_genscale (default; lower=worse). "
        "human_mean / human_min: ascending human size_score (lower=worse). "
        "human_then_eval: human mean first, then eval score as tiebreak.",
    )
    parser.add_argument("--disable_gemini_precise_prompt", action="store_true")
    parser.add_argument("--gemini_model_version", type=str,
                        default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--no_task3_visual_qc",
        action="store_true",
        help="Skip Gemini image screening (strong blur, wrong object count, severe artifacts).",
    )
    parser.add_argument(
        "--task3_qc_model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model id for Task3 visual QC (image+JSON).",
    )
    parser.add_argument(
        "--task3_qc_sleep_s",
        type=float,
        default=0.0,
        help="Pause between QC API calls to reduce rate-limit issues.",
    )
    args = parser.parse_args()

    if args.mode == "task12":
        build_benchmark(
            num_task1_same_plane=args.num_task1_same_plane,
            num_task1_natural=args.num_task1_natural,
            num_task2=args.num_task2,
            output_file=args.output,
        )
    else:
        build_task3_dataset(
            task1_eval_json=args.task1_eval_json,
            task2_eval_json=args.task2_eval_json,
            benchmark_json=args.benchmark_json,
            task1_image_root=args.task1_image_root,
            task2_image_root=args.task2_image_root,
            num_task1_select=args.task3_num_from_t1,
            num_task2_select=args.task3_num_from_t2,
            output_file=args.task3_output,
            use_gemini_precise_prompt=(not args.disable_gemini_precise_prompt),
            gemini_model_version=args.gemini_model_version,
            human_scores_json=args.human_scores_json,
            task3_rank_by=args.task3_rank_by,
            task3_human_whitelist=args.task3_human_whitelist,
            task1_gemini_json=args.task1_gemini_json,
            task2_gemini_json=args.task2_gemini_json,
            task3_select_mode=args.task3_select_mode,
            task3_seed=args.task3_seed,
            task3_perfect_score=args.task3_perfect_score,
            task3_visual_qc=(not args.no_task3_visual_qc),
            task3_qc_model=args.task3_qc_model,
            task3_qc_sleep_s=args.task3_qc_sleep_s,
        )