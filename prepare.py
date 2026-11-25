#!/usr/bin/env python3
"""
IMMUTABLE EVALUATION HARNESS (Karpathy `autoresearch` pattern).

DO NOT MODIFY THIS FILE during autonomous research loops.
Provides:
  - Fixed train/eval dataset loading (`data/world_sample.jsonl`, `data/eval_npc.jsonl`)
  - Tokenizer-agnostic `val_bpb` (bits-per-byte) computation
  - Hard gates: artifact size <= 500 MB, schema validity >= 0.95
  - Structured `::clab::metric` stdout emitters for live clab tracking
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = REPO_ROOT / "data" / "bs_train.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "bs_eval.jsonl"
LORE_CHUNKS_PATH = REPO_ROOT / "data" / "bs_lore_chunks.jsonl"
DOMAIN_TERMS_PATH = REPO_ROOT / "data" / "bs_domain_terms.json"

MAX_ARTIFACT_MB = 500.0
MIN_SCHEMA_VALID_RATE = 0.95
VALID_MOODS = {"calm", "wary", "friendly", "hostile", "fearful", "amused", "cryptic"}
VALID_ACTIONS = {
    "NONE",
    "GREET",
    "TRADE",
    "WARN",
    "DRAW_WEAPON",
    "FLEE",
    "SHARE_RUMOR",
    "OFFER_QUEST",
}


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_train_data() -> List[Dict[str, str]]:
    return load_jsonl(TRAIN_PATH)


def load_eval_data() -> List[Dict[str, str]]:
    return load_jsonl(EVAL_PATH)


def load_lore_chunks() -> List[Dict[str, str]]:
    return load_jsonl(LORE_CHUNKS_PATH) if LORE_CHUNKS_PATH.exists() else []


def validate_npc_json(text: str) -> Dict[str, Any]:
    """Check whether a generated NPC reply strictly satisfies the game schema."""
    try:
        obj = json.loads(text.strip())
    except Exception as exc:
        return {"valid": False, "reason": f"json_error: {exc}"}
    if not isinstance(obj, dict):
        return {"valid": False, "reason": "not_a_dict"}
    if set(obj.keys()) != {"mood", "action", "line"}:
        return {"valid": False, "reason": f"bad_keys: {sorted(obj.keys())}"}
    if obj["mood"] not in VALID_MOODS:
        return {"valid": False, "reason": f"bad_mood: {obj['mood']}"}
    if obj["action"] not in VALID_ACTIONS:
        return {"valid": False, "reason": f"bad_action: {obj['action']}"}
    line = obj["line"]
    if not isinstance(line, str) or not (1 <= len(line) <= 240):
        return {"valid": False, "reason": "bad_line_len"}
    return {"valid": True, "reason": "ok"}


def compute_val_bpb(
    eval_rows: Iterable[Dict[str, str]],
    target_nll_nats_fn: Callable[[str, str], float],
) -> Dict[str, float]:
    """
    Compute tokenizer-agnostic validation bits-per-byte (val_bpb) strictly on
    the target NPC responses given their prompts.

    `target_nll_nats_fn(prompt, target)` must return the total negative log
    likelihood (in nats, base e) of `target` conditioned on `prompt`.
    """
    total_nats = 0.0
    total_bytes = 0
    n_examples = 0

    for row in eval_rows:
        prompt = row["prompt"]
        target = row["target"]
        nll_nats = float(target_nll_nats_fn(prompt, target))
        if not math.isfinite(nll_nats) or nll_nats < 0:
            raise ValueError(f"Invalid NLL value: {nll_nats}")
        total_nats += nll_nats
        total_bytes += len(target.encode("utf-8"))
        n_examples += 1

    if total_bytes == 0:
        raise ValueError("Eval set has zero target bytes")

    val_bpb = (total_nats / math.log(2.0)) / total_bytes
    return {
        "val_bpb": val_bpb,
        "total_bytes": float(total_bytes),
        "n_examples": float(n_examples),
    }


def evaluate_and_report(
    val_bpb: float,
    artifact_mb: float,
    decode_tok_s: float | None = None,
    generations: List[str] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply hard gates and print structured metrics (including ::clab::metric)."""
    schema_rate = 1.0
    if generations is not None and len(generations) > 0:
        valid_count = sum(1 for g in generations if validate_npc_json(g)["valid"])
        schema_rate = valid_count / len(generations)

    size_ok = artifact_mb <= MAX_ARTIFACT_MB
    schema_ok = schema_rate >= MIN_SCHEMA_VALID_RATE
    bpb_ok = math.isfinite(val_bpb)
    pass_gate = bool(size_ok and schema_ok and bpb_ok)

    report: Dict[str, Any] = {
        "val_bpb": round(val_bpb, 6),
        "artifact_mb": round(artifact_mb, 3),
        "schema_valid_rate": round(schema_rate, 4),
        "decode_tok_s": round(decode_tok_s, 2) if decode_tok_s is not None else None,
        "pass_gate": pass_gate,
    }
    if extra:
        report.update(extra)

    metric_parts = [
        f"val_bpb={report['val_bpb']:.6f}",
        f"artifact_mb={report['artifact_mb']:.3f}",
        f"schema_rate={report['schema_valid_rate']:.4f}",
        f"pass_gate={int(pass_gate)}",
    ]
    if decode_tok_s is not None:
        metric_parts.append(f"decode_tok_s={decode_tok_s:.2f}")
    print("::clab::metric " + " ".join(metric_parts))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train_rows = load_train_data()
    eval_rows = load_eval_data()
    print(f"Loaded {len(train_rows)} train examples and {len(eval_rows)} eval examples.")
    invalid = [
        (i, validate_npc_json(r["target"]))
        for i, r in enumerate(train_rows + eval_rows)
        if not validate_npc_json(r["target"])["valid"]
    ]
    if invalid:
        raise SystemExit(f"Dataset schema errors found: {invalid}")
    print("All dataset targets pass strict NPC JSON schema.")
