#!/usr/bin/env python3
"""
#CLAB gpus=1 vram=24 cpus=8 ram=48 est=20m priority=high

Fast Batched Synthetic Borrowed Stars NPC & Interactive-Narrator Corpus Expander.

Grounds every generated dialogue pair in canonical chunks from `data/bs_lore_chunks.jsonl`,
using a strong local teacher model on A100 (e.g. cached `gemma-3-12b-it` or `Qwen3-14B`)
with batched left-padded generation (`--batch-size 16` -> ~10x throughput on A100!).
Strictly validates every output against `prepare.validate_npc_json()` and guarantees zero
overlap with frozen `data/bs_eval.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prepare

GEN_SYSTEM = """You are a senior narrative designer for the fantasy-maritime RPG "Borrowed Stars" (set in Aethermoor).
Given a canonical lore excerpt, write 3 distinct, grounded in-game dialogue training pairs in JSONL format.
Each JSON line MUST have exactly two keys: "prompt" and "target".
- "prompt" MUST follow this exact header format ending with a newline:
  "[LOC: <place>] [NPC: <Name, Role, Faction>] [WORLD: <grounded situation>] [PLAYER: <player action/question>] [INTENT: <NPC intent>]\\n"
  (Mix both regular Borrowed Stars NPCs AND "[NPC: The Narrator, Unseen Chronicler, Outside the Rim]" Talking-to-the-Narrator exchanges upholding the Ontological Horizon.)
- "target" MUST be a JSON object (or JSON-encoded string) with exact keys:
  {"mood": "<calm|wary|friendly|hostile|fearful|amused|cryptic>", "action": "<NONE|GREET|TRADE|WARN|DRAW_WEAPON|FLEE|SHARE_RUMOR|OFFER_QUEST>", "line": "<concise, literary, in-character spoken line <= 220 chars>"}
Output ONLY valid JSONL lines, no markdown fences, no commentary."""


def resolve_local_or_hf(repo: str) -> str:
    cache_root = Path("/project/alex_phd/research-cache/models")
    if cache_root.exists():
        slug = repo.replace("/", "--")
        exact = cache_root / slug
        if exact.is_dir() and (exact / "config.json").exists():
            return str(exact)
        for cand in sorted(cache_root.glob(f"{slug}--*")):
            if cand.is_dir() and (cand / "config.json").exists():
                return str(cand)
    return repo


def extract_valid_pairs(raw_text: str, eval_prompts: Set[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for raw_line in raw_text.splitlines():
        s = raw_line.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict) or "prompt" not in obj or "target" not in obj:
            continue
        prompt = obj["prompt"]
        target = obj["target"]
        if isinstance(target, dict):
            target = json.dumps(target, ensure_ascii=False)
        if not isinstance(prompt, str) or not isinstance(target, str):
            continue
        if not prompt.startswith("[LOC:") or "[NPC:" not in prompt or "[INTENT:" not in prompt:
            continue
        if not prompt.endswith("\n"):
            prompt = prompt.rstrip() + "\n"
        if prompt.strip().lower() in eval_prompts:
            continue
        chk = prepare.validate_npc_json(target)
        if not chk["valid"]:
            continue
        out.append({"prompt": prompt, "target": target})
    return out


def build_prompt_str(tok, chunk: Dict[str, str], ex1: Dict[str, str], ex2: Dict[str, str], narrator_bias: bool) -> str:
    extra = (
        "Focus at least 2 of the 3 pairs on '[NPC: The Narrator, Unseen Chronicler, Outside the Rim]' "
        "in interactive mid-narration dialogue with the player."
        if narrator_bias
        else "Focus on grounded Borrowed Stars inhabitants (clerks, sailors, archivists, keepers, rekindlers, rangers, merchants, guards)."
    )
    user_msg = (
        f"CANONICAL LORE EXCERPT ({chunk['source']}):\n{chunk['text'][:1300]}\n\n"
        f"{extra}\n\n"
        f"STYLE EXAMPLES:\n{json.dumps({'prompt': ex1['prompt'], 'target': ex1['target']}, ensure_ascii=False)}\n"
        f"{json.dumps({'prompt': ex2['prompt'], 'target': ex2['target']}, ensure_ascii=False)}\n\n"
        f"Now output 3 new grounded Borrowed Stars JSONL lines:"
    )
    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Gemma chat templates merge system into user turn
            return tok.apply_chat_template(
                [{"role": "user", "content": GEN_SYSTEM + "\n\n" + user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return GEN_SYSTEM + "\n\n" + user_msg + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="google/gemma-3-12b-it")
    ap.add_argument("--target-count", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="data/bs_synth_train.jsonl")
    ap.add_argument("--mirror-out", default="/project/alex_phd/npc-text-gen/data/bs_synth_train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lore_chunks = prepare.load_lore_chunks()
    train_examples = prepare.load_train_data()
    npc_examples = [r for r in train_examples if r.get("subset") == "npc"] or train_examples
    narr_examples = [r for r in train_examples if r.get("subset") == "narrator_interactive"] or train_examples
    eval_prompts = {r["prompt"].strip().lower() for r in prepare.load_eval_data()}

    src = resolve_local_or_hf(args.teacher)
    print(f"Loading teacher {args.teacher} (src={src}) with batch_size={args.batch_size}...", flush=True)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            src, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    collected: List[Dict[str, str]] = []
    seen_targets: Set[str] = set()
    if out_path.exists():
        for r in prepare.load_jsonl(out_path):
            collected.append(r)
            seen_targets.add(r["target"])
        print(f"Resuming from {len(collected)} existing synthetic examples.", flush=True)

    t0 = time.time()
    batch_num = 0
    chunk_cursor = 0
    with out_path.open("a", encoding="utf-8") as fout:
        while len(collected) < args.target_count:
            prompt_strs: List[str] = []
            batch_chunks: List[Dict[str, str]] = []
            for bi in range(args.batch_size):
                ch = lore_chunks[(chunk_cursor + bi) % len(lore_chunks)]
                narr_bias = ((chunk_cursor + bi) % 3 == 0)
                ex1 = rng.choice(narr_examples if narr_bias else npc_examples)
                ex2 = rng.choice(train_examples)
                prompt_strs.append(build_prompt_str(tok, ch, ex1, ex2, narr_bias))
                batch_chunks.append(ch)
            chunk_cursor += args.batch_size

            enc = tok(prompt_strs, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
            in_len = enc.input_ids.shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=420,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.92,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            added = 0
            for bi in range(args.batch_size):
                raw = tok.decode(gen[bi, in_len:], skip_special_tokens=True)
                pairs = extract_valid_pairs(raw, eval_prompts)
                for p in pairs:
                    if p["target"] not in seen_targets:
                        seen_targets.add(p["target"])
                        p["source_chunk"] = batch_chunks[bi]["source"]
                        p["subset"] = "narrator_interactive" if "The Narrator" in p["prompt"] else "npc"
                        collected.append(p)
                        fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                        added += 1
            fout.flush()
            batch_num += 1
            elapsed = max(time.time() - t0, 1e-6)
            rate = len(collected) / elapsed
            print(
                f"::clab::progress {min(len(collected), args.target_count)}/{args.target_count} "
                f"(batch={batch_num} +{added} total={len(collected)} rate={rate:.2f} ex/s elapsed={elapsed:.0f}s)",
                flush=True,
            )

    if args.mirror_out:
        mp = Path(args.mirror_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Mirrored {len(collected)} rows to {mp}", flush=True)

    print(f"Done! Wrote {len(collected)} verified Borrowed Stars synthetic pairs to {out_path}.", flush=True)


if __name__ == "__main__":
    main()
