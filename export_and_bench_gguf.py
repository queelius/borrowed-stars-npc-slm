#!/usr/bin/env python3
"""
#CLAB cpus=32 ram=48 est=25m priority=normal

Convert Shootout v2 Borrowed Stars HF checkpoints -> Q4_K_M GGUFs via llama.cpp,
measure exact disk size (MB), real CPU prefill/decode tok/s at 4 & 8 game-safe threads
via `llama-bench`, and run live GBNF-constrained Borrowed Stars NPC/Narrator generations!
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import prepare

LLAMA_DIR = Path("/project/alex_phd/npc-text-gen/llama.cpp")
LLAMA_BENCH = LLAMA_DIR / "build/bin/llama-bench"
LLAMA_CLI = LLAMA_DIR / "build/bin/llama-cli"
LLAMA_COMPLETION = LLAMA_DIR / "build/bin/llama-completion"
LLAMA_QUANTIZE = LLAMA_DIR / "build/bin/llama-quantize"
CONVERT_PY = LLAMA_DIR / "convert_hf_to_gguf.py"
GRAMMAR_V1_PATH = Path(__file__).resolve().parent / "npc_grammar.gbnf"
GRAMMAR_V2_PATH = Path(__file__).resolve().parent / "npc_grammar_v2.gbnf"

SYSTEM_PROMPT_V1 = (
    "You are the Borrowed Stars in-game dialogue & Interactive Narrator engine. "
    "Reply ONLY with a single valid JSON object matching "
    "{\"mood\": <calm|wary|friendly|hostile|fearful|amused|cryptic>, "
    "\"action\": <NONE|GREET|TRADE|WARN|DRAW_WEAPON|FLEE|SHARE_RUMOR|OFFER_QUEST>, "
    "\"line\": <concise in-character response <= 240 chars>}.\n"
)

SYSTEM_PROMPT_V2 = (
    "You are the Borrowed Stars NPC & Interactive Narrator engine. "
    "Emit ONLY a single valid JSON object {\"mood\", \"line\", \"calls\"} with 0..2 authorized tool calls from [ALLOWED_TOOLS]. "
    "Never use non-canon names or break character.\n"
)


def extract_last_json_object(raw: str) -> str:
    """Extract the final top-level JSON object from llama-completion/llama-cli stdout."""
    text = raw
    for marker in ("[End thinking]", "JSON:\n", "JSON:"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    # Scan backwards from the last '}' to find its matching '{'
    end = text.rfind("}")
    if end == -1:
        return text.strip().splitlines()[-1] if text.strip() else ""
    depth = 0
    in_str = False
    for i in range(end, -1, -1):
        ch = text[i]
        if in_str:
            if ch == '"' and not (i > 0 and text[i - 1] == "\\"):
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i : end + 1].strip()
    return text.strip()


def parse_llama_bench_json(raw: str) -> Dict[str, Any]:
    rows = json.loads(raw)
    out: Dict[str, Any] = {}
    for r in rows:
        n_prompt = r.get("n_prompt", 0)
        n_gen = r.get("n_gen", 0)
        t = r.get("n_threads", 0)
        ts = round(float(r.get("avg_ts", 0.0)), 2)
        if n_prompt > 0 and n_gen == 0:
            out[f"pp{n_prompt}_t{t}_tok_s"] = ts
            out[f"ttft_pp{n_prompt}_t{t}_ms"] = round(1000.0 * n_prompt / max(ts, 1e-6), 1)
        elif n_gen > 0 and n_prompt == 0:
            out[f"tg{n_gen}_t{t}_tok_s"] = ts
    return out


def convert_and_bench(
    slug: str,
    ckpt_dir: Path,
    gguf_dir: Path,
    quant: str,
    contract: str = "v1",
    eval_v2_path: Path | None = None,
) -> Dict[str, Any]:
    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16_path = gguf_dir / f"{slug}_f16.gguf"
    q_path = gguf_dir / f"{slug}_{quant}.gguf"

    print(f"\n==================== GGUF + CPU BENCH: {slug} ({quant}) ====================", flush=True)
    if not q_path.exists():
        if not f16_path.exists():
            print(f"Converting {ckpt_dir} -> {f16_path} ...", flush=True)
            subprocess.run(
                [sys.executable, str(CONVERT_PY), str(ckpt_dir), "--outfile", str(f16_path), "--outtype", "f16"],
                check=True,
            )
        print(f"Quantizing {f16_path} -> {q_path} ({quant}) ...", flush=True)
        subprocess.run([str(LLAMA_QUANTIZE), str(f16_path), str(q_path), quant], check=True)
        f16_path.unlink(missing_ok=True)

    size_mb = round(q_path.stat().st_size / (1024 * 1024), 2)
    print(f"GGUF size on disk: {size_mb} MB", flush=True)

    bench_cmd = [
        str(LLAMA_BENCH),
        "-m", str(q_path),
        "-p", "128",
        "-n", "64",
        "-t", "4,8",
        "-r", "3",
        "-o", "json",
    ]
    bproc = subprocess.run(bench_cmd, capture_output=True, text=True, check=True)
    perf = parse_llama_bench_json(bproc.stdout)
    print("BENCH:", json.dumps(perf), flush=True)

    grammar_path = GRAMMAR_V2_PATH if contract == "v2" else GRAMMAR_V1_PATH
    sys_prompt = SYSTEM_PROMPT_V2 if contract == "v2" else SYSTEM_PROMPT_V1

    samples = []
    if contract == "v2" and eval_v2_path and eval_v2_path.exists():
        import validate_v2
        all_rows = [json.loads(line) for line in eval_v2_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Pick 1 give_item row, 1 steer_dialogue_node row, 1 refusal row, 1 pure-dialogue row
        def _tools_in(row):
            try:
                return [c.get("tool") for c in json.loads(row["target"]).get("calls", [])]
            except Exception:
                return []

        picked = []
        for pred_fn in (
            lambda r: "give_item" in _tools_in(r),
            lambda r: "steer_dialogue_node" in _tools_in(r),
            lambda r: str(r.get("category", "")).startswith("refusal_"),
            lambda r: not _tools_in(r) and not str(r.get("category", "")).startswith("refusal_"),
        ):
            for i, row in enumerate(all_rows):
                if pred_fn(row) and i not in [p[0] for p in picked]:
                    picked.append((i, row))
                    break
        if not picked:
            picked = [(i, all_rows[i]) for i in range(min(3, len(all_rows)))]
    else:
        eval_rows = prepare.load_eval_data()
        picked = [(idx, eval_rows[idx]) for idx in (0, 13) if idx < len(eval_rows)]

    for idx, row in picked:
        prompt = sys_prompt + row["prompt"]
        use_comp = LLAMA_COMPLETION.exists()
        cmd = [
            str(LLAMA_COMPLETION if use_comp else LLAMA_CLI),
            "-m", str(q_path),
            "-t", "8",
            "-n", "128" if contract == "v2" else "80",
            "--temp", "0.2",
            "--grammar-file", str(grammar_path),
            "-p", prompt,
            "--no-display-prompt",
        ]
        if not use_comp:
            cmd.append("--single-turn")
            if "qwen" in slug.lower():
                cmd.append("--no-jinja")
        t0 = time.time()
        cproc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        wall_ms = round((time.time() - t0) * 1000.0, 1)
        pred = extract_last_json_object(cproc.stdout.strip())
        if contract == "v2":
            import validate_v2
            chk = validate_v2.validate_v2_response(pred, row["prompt"])
            valid = bool(chk["valid"])
            errs = [] if valid else [chk.get("reason", "invalid")]
        else:
            chk = prepare.validate_npc_json(pred)
            valid = bool(chk["valid"])
            errs = [] if valid else [chk.get("reason", "invalid")]
        samples.append(
            {
                "idx": idx,
                "subset": row.get("subset") or row.get("category"),
                "wall_ms": wall_ms,
                "schema_valid": valid,
                "errors": errs,
                "pred": pred,
            }
        )
        print(f"  [GBNF-{contract} idx={idx} {wall_ms}ms valid={valid}] {pred}", flush=True)

    return {
        "slug": slug,
        "quant": quant,
        "contract": contract,
        "gguf_path": str(q_path),
        "gguf_mb": size_mb,
        "perf": perf,
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/project/alex_phd/npc-text-gen/shootout_v2/checkpoints")
    ap.add_argument("--gguf-dir", default="/project/alex_phd/npc-text-gen/gguf_v2")
    ap.add_argument("--contract", choices=["v1", "v2"], default="v1")
    ap.add_argument("--eval-v2", default="/project/alex_phd/npc-text-gen/data/bs_tools_v2_eval.jsonl")
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "google__gemma-3-1b-it:Q4_K_M",
            "google__gemma-3-1b-it:Q3_K_M",
            "Qwen__Qwen3.5-0.8B:Q4_K_M",
            "LiquidAI__LFM2-700M:Q4_K_M",
            "LiquidAI__LFM2.5-350M:Q4_K_M",
            "google__gemma-3-270m-it:Q4_K_M",
        ],
    )
    ap.add_argument("--out-json", default="results/gguf_cpu_bench_v2.json")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    gguf_dir = Path(args.gguf_dir)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_v2_path = Path(args.eval_v2) if args.eval_v2 else None

    results: List[Dict[str, Any]] = []
    for spec in args.models:
        slug, quant = spec.split(":", 1)
        ckpt_dir = ckpt_root / slug
        if not ckpt_dir.is_dir():
            print(f"[SKIP] Missing checkpoint {ckpt_dir}", flush=True)
            continue
        try:
            res = convert_and_bench(
                slug, ckpt_dir, gguf_dir, quant, contract=args.contract, eval_v2_path=eval_v2_path
            )
            results.append(res)
        except Exception as exc:
            print(f"[ERROR] {spec}: {type(exc).__name__}: {exc}", flush=True)
            results.append({"slug": slug, "quant": quant, "error": f"{type(exc).__name__}: {exc}"})
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n================ CPU GGUF BENCHMARK SUMMARY ================")
    for r in results:
        if "error" in r:
            print(f"FAIL  {r['slug']:30} {r['quant']:7} {r['error']}")
        else:
            p = r["perf"]
            print(
                f"OK    {r['slug']:30} {r['quant']:7} {r['gguf_mb']:6.1f}MB  "
                f"4t: {p.get('tg64_t4_tok_s',0):6.1f} tok/s (TTFT {p.get('ttft_pp128_t4_ms',0):5.0f}ms) | "
                f"8t: {p.get('tg64_t8_tok_s',0):6.1f} tok/s (TTFT {p.get('ttft_pp128_t8_ms',0):5.0f}ms)"
            )


if __name__ == "__main__":
    main()
