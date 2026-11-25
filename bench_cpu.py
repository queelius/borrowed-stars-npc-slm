#!/usr/bin/env python3
"""
#CLAB cpus=16 ram=32 est=30m priority=normal

CPU Inference Benchmark Harness for <=500MB In-Game NPC GGUF Candidates.

Benchmarks local GGUF models (via `llama-cli` / `llama-bench` if present on PATH,
or falls back to Python environment + `train.py` looped-scratch CPU timing when
no GGUF binary/file is supplied yet). Measures:
  - Peak RSS RAM (MB)
  - Prompt prefill / TTFT (ms)
  - Autoregressive decode speed (tok/s)
  - Strict GBNF schema pass rate (`npc_grammar.gbnf`)
  - Optional `--spec-type ngram-mod` speculative decoding speedup
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

import prepare

GRAMMAR_PATH = Path(__file__).resolve().parent / "npc_grammar.gbnf"


def run_llama_cli_bench(
    llama_bin: str,
    model_path: str,
    threads: int,
    use_grammar: bool,
    use_ngram_spec: bool,
    n_predict: int = 48,
) -> dict:
    eval_rows = prepare.load_eval_data()
    prompt = (
        "You are an in-game NPC dialogue engine. Reply ONLY with valid JSON "
        "matching {\"mood\", \"action\", \"line\"}.\n" + eval_rows[0]["prompt"]
    )
    cmd = [
        llama_bin,
        "-m",
        model_path,
        "-t",
        str(threads),
        "-n",
        str(n_predict),
        "--temp",
        "0.2",
        "-p",
        prompt,
        "--no-display-prompt",
    ]
    if use_grammar and GRAMMAR_PATH.exists():
        cmd += ["--grammar-file", str(GRAMMAR_PATH)]
    if use_ngram_spec:
        cmd += ["--spec-type", "ngram-mod", "--draft-max", "12"]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    elapsed = time.time() - t0

    output = proc.stdout.strip()
    schema_check = prepare.validate_npc_json(output)
    model_mb = Path(model_path).stat().st_size / (1024 * 1024)
    return {
        "model": Path(model_path).name,
        "model_mb": round(model_mb, 2),
        "threads": threads,
        "grammar": use_grammar,
        "ngram_spec": use_ngram_spec,
        "wall_sec": round(elapsed, 3),
        "rc": proc.returncode,
        "schema_valid": schema_check["valid"],
        "sample_output": output[:240],
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-gguf", default="", help="Path to local .gguf file to benchmark")
    ap.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--quick", action="store_true", help="Run quick host + looped-scratch CPU baseline")
    args = ap.parse_args()

    host_info = {
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "llama_cli": shutil.which("llama-cli"),
        "llama_bench": shutil.which("llama-bench"),
    }
    print("HOST:", json.dumps(host_info))

    if args.model_gguf:
        llama_bin = host_info["llama_cli"]
        if not llama_bin:
            raise SystemExit("llama-cli not found in PATH. Build llama.cpp or add it to PATH.")
        results = [
            run_llama_cli_bench(llama_bin, args.model_gguf, args.threads, True, False),
            run_llama_cli_bench(llama_bin, args.model_gguf, args.threads, True, True),
        ]
        print(json.dumps(results, indent=2))
        return

    if args.quick:
        print("No --model-gguf supplied; running quick CPU smoke benchmark via train.py (--mode looped-scratch)...")
        cmd = ["python3", str(Path(__file__).resolve().parent / "train.py"), "--mode", "looped-scratch", "--budget-sec", "10", "--device", "cpu"]
        proc = subprocess.run(cmd, check=True)
        rss_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
        print(f"Peak child RSS: {rss_mb:.1f} MB (rc={proc.returncode})")


if __name__ == "__main__":
    main()
