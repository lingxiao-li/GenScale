#!/usr/bin/env python3
"""
Gemini VLM-as-a-Judge evaluation for GenScale Task3, v5.

Task3 rows live in `GenScale_Benchmark_v3_final.json` and declare provenance via:
  - source_task_type: "T1" | "T2"
  - source_task_id: e.g. "T1_0123" / "T2_0045"
  - prompt_type: subfolder name under the model image root (same as other T3 eval scripts)

We do NOT duplicate the Gemini scoring logic here. Instead, we:
  1) Split Task3 rows by provenance
  2) Build tiny benchmark JSONs containing ONLY the referenced source rows (T1_/T2_),
     pulled from the same final list JSON as the Task3 rows (`GenScale_Benchmark_v3_final.json`).
  3) Stage flat image dirs with synthetic filenames so each distinct T3 output can be scored
     even when multiple T3 rows share the same source_task_id (different edits / prompt_type)
  4) Run the existing v5 scorers as subprocesses:
       - task1: scripts/eval/task1/evaluate_genscale_task1_gemini_v5.py
       - task2: scripts/eval/task2/evaluate_genscale_task2_gemini_v5.py
  5) Post-process outputs: remap synthetic task_ids back to real T3_* ids and fix image_path

Outputs are written to --out (single merged JSON list), plus optional per-branch temp outputs under --work_dir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


def _repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


DEFAULT_FINAL_BENCHMARK = os.path.join(os.path.dirname(_repo_root()), "GenScale_Benchmark_v3_final_anonymous.json")


def _encode_synthetic_source_task_id(source_tid: str, t3_tid: str) -> str:
    """
    Must be a valid filename stem and unlikely to collide with real benchmark ids.
    Example: T1_0123 + T3_0042 -> T1_0123__t3_T3_0042
    """
    # Task ids like T3_0123 are already filename-safe; keep them literal for easy debugging.
    return f"{source_tid}__t3_{t3_tid}"


def _decode_synthetic_source_task_id(synthetic_tid: str) -> Tuple[str, str]:
    if "__t3_" not in synthetic_tid:
        raise ValueError(f"not a synthetic tid: {synthetic_tid!r}")
    src, rest = synthetic_tid.split("__t3_", 1)
    # rest is the original T3 task id (e.g. T3_0042)
    return src, rest


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def index_rows_by_task_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        tid = str(r.get("task_id", ""))
        if tid and tid not in m:
            m[tid] = r
    return m


def build_source_indexes_from_final_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    t1_rows = [x for x in rows if str(x.get("task_id", "")).startswith("T1_")]
    t2_rows = [x for x in rows if str(x.get("task_id", "")).startswith("T2_")]
    return index_rows_by_task_id(t1_rows), index_rows_by_task_id(t2_rows)


def extract_t3_rows_from_list(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [x for x in data if str(x.get("task_id", "")).startswith("T3_")]
    out.sort(key=lambda x: str(x.get("task_id", "")))
    return out


def load_t3_rows(t3_benchmark: str) -> List[Dict[str, Any]]:
    data = load_json(t3_benchmark)
    if not isinstance(data, list):
        raise SystemExit(f"ERROR: expected list JSON in {t3_benchmark}")
    return extract_t3_rows_from_list(data)


def resolve_t3_image_path(image_dir: str, t3_row: Dict[str, Any]) -> str:
    tid = str(t3_row.get("task_id", ""))
    ptype = str(t3_row.get("prompt_type", "") or "").strip()
    candidates = []
    if ptype:
        candidates.append(os.path.join(image_dir, ptype, f"{tid}.png"))
    candidates.append(os.path.join(image_dir, f"{tid}.png"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(" / ".join(candidates))


def link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        try:
            if os.path.islink(dst) or os.path.isfile(dst):
                os.remove(dst)
        except IsADirectoryError:
            shutil.rmtree(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


@dataclass
class BranchPlan:
    name: str
    scorer_py: str
    bench_out_path: str
    img_stage_dir: str
    score_out_path: str


def build_branch_plan(
    *,
    name: str,
    scorer_py: str,
    t3_rows: List[Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
    image_dir: str,
    work_dir: str,
) -> BranchPlan:
    os.makedirs(work_dir, exist_ok=True)
    bench_out_path = os.path.join(work_dir, f"_tmp_benchmark_{name}.json")
    img_stage_dir = os.path.join(work_dir, f"_tmp_images_{name}")
    score_out_path = os.path.join(work_dir, f"_tmp_scores_{name}.json")

    if os.path.isdir(img_stage_dir):
        shutil.rmtree(img_stage_dir)
    os.makedirs(img_stage_dir, exist_ok=True)

    bench_rows: List[Dict[str, Any]] = []
    t3_by_synthetic: Dict[str, Dict[str, Any]] = {}

    for t3 in t3_rows:
        t3_tid = str(t3.get("task_id", ""))
        src_type = str(t3.get("source_task_type", "") or "").strip().upper()
        src_tid = str(t3.get("source_task_id", "") or "").strip()
        if src_type != name.upper():
            raise SystemExit(f"internal error: branch {name} got wrong source_type for {t3_tid}")
        if not src_tid or src_tid not in source_index:
            raise SystemExit(
                f"ERROR: missing source row {src_tid!r} in source benchmark index "
                f"(needed by {t3_tid}; branch={name})"
            )

        base_row = json.loads(json.dumps(source_index[src_tid]))  # deep copy
        syn = _encode_synthetic_source_task_id(src_tid, t3_tid)
        base_row["task_id"] = syn
        bench_rows.append(base_row)
        t3_by_synthetic[syn] = t3

        img_src = resolve_t3_image_path(image_dir, t3)
        img_dst = os.path.join(img_stage_dir, f"{syn}.png")
        link_or_copy(img_src, img_dst)

    dump_json(bench_out_path, bench_rows)
    # Persist mapping for debugging / postprocess
    mapping_for_json: Dict[str, Any] = {k: {"task_id": str(v.get("task_id", "")),
                                            "source_task_id": str(v.get("source_task_id", "")),
                                            "source_task_type": str(v.get("source_task_type", "")),
                                            "prompt_type": str(v.get("prompt_type", ""))}
                                       for k, v in t3_by_synthetic.items()}
    dump_json(os.path.join(work_dir, f"_tmp_mapping_{name}.json"), mapping_for_json)

    return BranchPlan(
        name=name,
        scorer_py=scorer_py,
        bench_out_path=bench_out_path,
        img_stage_dir=img_stage_dir,
        score_out_path=score_out_path,
    )


def run_scorer_subprocess(
    *,
    python_exe: str,
    scorer_py: str,
    benchmark_path: str,
    kb_csv: str,
    image_dir: str,
    out_path: str,
    passthrough_args: List[str],
) -> None:
    # Provide sensible per-branch defaults without requiring users to remember subtle differences
    # between Task1 vs Task2 v5 scorers (notably --pair-workers).
    tail: List[str] = []
    joined = " ".join(passthrough_args)
    if "--pair-workers" not in joined and "-pair-workers" not in joined:
        base = os.path.basename(scorer_py)
        if "task1" in base:
            tail.extend(["--pair-workers", "6"])
        elif "task2" in base:
            tail.extend(["--pair-workers", "1"])
    cmd = [
        python_exe,
        "-u",
        scorer_py,
        "--benchmark",
        benchmark_path,
        "--kb_csv",
        kb_csv,
        "--image_dir",
        image_dir,
        "--out",
        out_path,
        *passthrough_args,
        *tail,
    ]
    print("\n==> Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def postprocess_output_json(
    out_path: str,
    mapping_path: str,
    final_image_dir: str,
) -> List[Dict[str, Any]]:
    mapping: Dict[str, Any] = load_json(mapping_path)
    if not isinstance(mapping, dict):
        raise SystemExit(f"ERROR: mapping json is not a dict: {mapping_path}")

    rows = load_json(out_path)
    if not isinstance(rows, list):
        raise SystemExit(f"ERROR: scorer output is not a list: {out_path}")

    fixed: List[Dict[str, Any]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        syn = str(rec.get("task_id", "") or "")
        if syn not in mapping:
            fixed.append(rec)
            continue
        t3 = mapping[syn]
        t3_tid = str(t3.get("task_id", ""))
        ptype = str(t3.get("prompt_type", "") or "")

        new_rec = dict(rec)
        new_rec["task_id"] = t3_tid
        new_rec["source_task_id"] = str(t3.get("source_task_id", ""))
        new_rec["source_task_type"] = str(t3.get("source_task_type", ""))
        new_rec["prompt_type"] = ptype

        # Prefer the on-disk path under the user-provided image_dir/prompt_type/
        try:
            new_rec["image_path"] = resolve_t3_image_path(final_image_dir, t3)
        except Exception:
            # leave as-is if cannot resolve
            pass

        fixed.append(new_rec)

    dump_json(out_path, fixed)
    return fixed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--t3_benchmark",
        default=DEFAULT_FINAL_BENCHMARK,
        help="GenScale v3 final benchmark JSON (list): T1_*, T2_*, and T3_* rows in one file.",
    )
    p.add_argument(
        "--image_dir",
        required=True,
        help="Task3 model image root, e.g. .../genscale_eval_images_T3_v3/OpenAI_GPT_Image_2",
    )
    p.add_argument("--out", required=True, help="Final merged Task3 scores JSON path.")
    p.add_argument(
        "--work_dir",
        default="",
        help="Temp working directory (benchmark slices, staged images, per-branch score json). "
        "Default: a directory under /tmp.",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable to run v5 scorers.")
    p.add_argument("--kb_csv", default=os.path.join(_repo_root(), "scripts", "authoritative_kb_3d_100.csv"))

    # Optional: restrict which T3 rows to score (by index in sorted T3 list)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)

    # Everything after `--` (or any unknown flags) is forwarded to BOTH scorers.
    p.add_argument(
        "scorer_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to task1/task2 v5 scorers (e.g. --model ... --samples_per_pair 5). "
        "Tip: use `--` before flags, e.g. `-- --model gemini-3.1-pro-preview`",
    )
    args = p.parse_args()

    passthrough = list(args.scorer_args or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    final_path = str(args.t3_benchmark)
    final_rows = load_json(final_path)
    if not isinstance(final_rows, list):
        raise SystemExit(f"ERROR: expected list JSON in {final_path}")

    t3_rows_all = extract_t3_rows_from_list(final_rows)
    n_all = len(t3_rows_all)
    start = 0 if args.start is None else int(args.start)
    end = n_all if args.end is None else int(args.end)
    if start < 0 or end < start or end > n_all:
        raise SystemExit(f"ERROR: invalid --start/--end: start={start} end={end} (have {n_all} T3 rows)")
    t3_rows = t3_rows_all[start:end]
    print(f"Task3 rows: {len(t3_rows)} (slice [{start}:{end}] of {n_all})", flush=True)

    t1_index, t2_index = build_source_indexes_from_final_rows(final_rows)

    t3_t1 = [r for r in t3_rows if str(r.get("source_task_type", "")).strip().upper() == "T1"]
    t3_t2 = [r for r in t3_rows if str(r.get("source_task_type", "")).strip().upper() == "T2"]
    unknown = [r for r in t3_rows if str(r.get("source_task_type", "")).strip().upper() not in ("T1", "T2")]
    if unknown:
        bad = {str(r.get("task_id")) for r in unknown}
        raise SystemExit(f"ERROR: unsupported source_task_type for Task3 rows: {sorted(bad)[:20]}")

    needed_t1 = sorted({str(r.get("source_task_id", "") or "").strip() for r in t3_t1 if str(r.get("source_task_id", "") or "").strip()})
    needed_t2 = sorted({str(r.get("source_task_id", "") or "").strip() for r in t3_t2 if str(r.get("source_task_id", "") or "").strip()})
    missing_t1 = [tid for tid in needed_t1 if tid not in t1_index]
    missing_t2 = [tid for tid in needed_t2 if tid not in t2_index]
    if missing_t1:
        raise SystemExit(
            "ERROR: missing T1 source rows in --t3_benchmark (final JSON must include every referenced T1_* row).\n"
            f"- missing ({len(missing_t1)}): {missing_t1[:20]}{' ...' if len(missing_t1) > 20 else ''}"
        )
    if missing_t2:
        raise SystemExit(
            "ERROR: missing T2 source rows in --t3_benchmark (final JSON must include every referenced T2_* row).\n"
            f"- missing ({len(missing_t2)}): {missing_t2[:20]}{' ...' if len(missing_t2) > 20 else ''}"
        )

    work_dir = (args.work_dir or "").strip()
    if not work_dir:
        work_dir = tempfile.mkdtemp(prefix="genscale_t3_gemini_v5_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"Work dir: {work_dir}", flush=True)

    scorer_t1 = os.path.join(_repo_root(), "scripts", "eval", "task1", "evaluate_genscale_task1_gemini_v5.py")
    scorer_t2 = os.path.join(_repo_root(), "scripts", "eval", "task2", "evaluate_genscale_task2_gemini_v5.py")

    merged: List[Dict[str, Any]] = []

    if t3_t1:
        plan = build_branch_plan(
            name="t1",
            scorer_py=scorer_t1,
            t3_rows=t3_t1,
            source_index=t1_index,
            image_dir=args.image_dir,
            work_dir=work_dir,
        )
        run_scorer_subprocess(
            python_exe=args.python,
            scorer_py=plan.scorer_py,
            benchmark_path=plan.bench_out_path,
            kb_csv=args.kb_csv,
            image_dir=plan.img_stage_dir,
            out_path=plan.score_out_path,
            passthrough_args=passthrough,
        )
        mapping_path = os.path.join(work_dir, "_tmp_mapping_t1.json")
        merged.extend(
            postprocess_output_json(plan.score_out_path, mapping_path, args.image_dir)
        )

    if t3_t2:
        plan = build_branch_plan(
            name="t2",
            scorer_py=scorer_t2,
            t3_rows=t3_t2,
            source_index=t2_index,
            image_dir=args.image_dir,
            work_dir=work_dir,
        )
        run_scorer_subprocess(
            python_exe=args.python,
            scorer_py=plan.scorer_py,
            benchmark_path=plan.bench_out_path,
            kb_csv=args.kb_csv,
            image_dir=plan.img_stage_dir,
            out_path=plan.score_out_path,
            passthrough_args=passthrough,
        )
        mapping_path = os.path.join(work_dir, "_tmp_mapping_t2.json")
        merged.extend(
            postprocess_output_json(plan.score_out_path, mapping_path, args.image_dir)
        )

    # Stable ordering for downstream tools
    merged.sort(key=lambda r: str((r or {}).get("task_id", "")))
    dump_json(args.out, merged)
    print(f"\n[+] Wrote merged Task3 scores: {args.out} ({len(merged)} records)", flush=True)


if __name__ == "__main__":
    main()
