#!/usr/bin/env python3
"""
Publish the 3 Borrowed Stars Stage 2 (v2.1, 23-tool) Q4_K_M GGUF models to Hugging Face
with principled per-architecture repos, accurate upstream licenses, base_model tree links,
complete benchmark model cards, npc_grammar_v2.gbnf, and a unified HF Collection.

Usage:
  uv run --no-project --python 3.12 --with huggingface_hub python publish_hf.py
  uv run --no-project --python 3.12 --with huggingface_hub python publish_hf.py --private-dataset
  uv run --no-project --python 3.12 --with huggingface_hub python publish_hf.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
GRAMMAR_PATH = ROOT / "npc_grammar_v2.gbnf"
SPEC_PATH = ROOT / "data" / "borrowed_stars_tools_spec.json"
SPEC_MD_PATH = ROOT / "data" / "BORROWED_STARS_TOOLS_SPEC.md"

REPOS: List[Dict[str, Any]] = [
    {
        "repo_slug": "borrowed-stars-npc-qwen3.5-0.8b-32k-GGUF",
        "local_file": "Qwen__Qwen3.5-0.8B__32k_Q4_K_M.gguf",
        "remote_file": "borrowed-stars-npc-qwen3.5-0.8b-32k-Q4_K_M.gguf",
        "tier": "Default / Best Overall (Apache 2.0)",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "license": "apache-2.0",
        "license_note": "Apache 2.0 (inherited from `Qwen/Qwen3.5-0.8B`)",
        "size_mb": "327.9 MB",
        "params": "531.7M (32K English + Aethermoor pruned from 752.4M / 248K vocab)",
        "val_bpb": "0.4908",
        "schema_rate": "91.7%",
        "tool_f1": "0.747 (P=0.84 / R=0.67)",
        "refusal_rate": "96.7% (29/30)",
        "chitchat_zero": "97.7%",
        "kill_leak": "0.00% (0/120)",
        "laptop_speed": "66–68 tok/s decode, 378 t/s prefill, ~1.5–1.7 s full turn (Ryzen 5 7430U, 6 CPU threads)",
        "server_flags": "--no-jinja",
        "extra_note": (
            "**Important (`llama.cpp`):** Pass `--no-jinja` (or call `/completion` / `/v1/completions`) so "
            "Qwen3.5's default thinking template does not prepend `<think>` tokens before the GBNF JSON root."
        ),
    },
    {
        "repo_slug": "borrowed-stars-npc-lfm2.5-350m-GGUF",
        "local_file": "LiquidAI__LFM2.5-350M_Q4_K_M.gguf",
        "remote_file": "borrowed-stars-npc-lfm2.5-350m-Q4_K_M.gguf",
        "tier": "Ultra-Fast / Low-Spec Laptop",
        "base_model": "LiquidAI/LFM2.5-350M",
        "license": "other",
        "license_name": "lfm1.0",
        "license_link": "https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE",
        "license_note": "LFM Open License v1.0 (inherited from `LiquidAI/LFM2.5-350M`)",
        "size_mb": "218.7 MB",
        "params": "354M (hybrid short-conv + GQA)",
        "val_bpb": "0.4944",
        "schema_rate": "92.5%",
        "tool_f1": "0.683 (P=0.78 / R=0.61)",
        "refusal_rate": "90.0% (27/30)",
        "chitchat_zero": "100.0% (44/44)",
        "kill_leak": "0.00% (0/120)",
        "laptop_speed": "123–125 tok/s decode, 827 t/s prefill, ~0.88–0.94 s full turn (Ryzen 5 7430U, 6 CPU threads)",
        "server_flags": "",
        "extra_note": (
            "Fastest CPU tier (`>120 tok/s` on a regular laptop CPU, `127 ms` TTFT on server CPU). "
            "Requires a recent `llama.cpp` build supporting the `lfm2` architecture."
        ),
    },
    {
        "repo_slug": "borrowed-stars-npc-gemma3-1b-32k-GGUF",
        "local_file": "google__gemma-3-1b-it__32k_Q4_K_M.gguf",
        "remote_file": "borrowed-stars-npc-gemma3-1b-32k-Q4_K_M.gguf",
        "tier": "Max Literary Prose",
        "base_model": "google/gemma-3-1b-it",
        "license": "gemma",
        "license_note": "Gemma Terms of Use (inherited from `google/gemma-3-1b-it`)",
        "size_mb": "495.5 MB",
        "params": "735.6M (32K native SentencePiece pruned from 999.9M / 262K vocab; -280 MB, +77% 4t CPU speed)",
        "val_bpb": "0.4107 (#1 calibration)",
        "schema_rate": "90.8%",
        "tool_f1": "0.716 (P=0.83 / R=0.63)",
        "refusal_rate": "86.7% (26/30)",
        "chitchat_zero": "100.0% (44/44)",
        "kill_leak": "0.00% (0/120)",
        "laptop_speed": "48–51 tok/s decode, 172 t/s prefill, ~2.6–3.0 s full turn (Ryzen 5 7430U, 6 CPU threads)",
        "server_flags": "",
        "extra_note": (
            "Pruned from 262,144 to 32,768 tokens via native SentencePiece protobuf slicing so spaces decode cleanly "
            "in `llama.cpp` while cutting disk size from `776.5 MB` down to `495.5 MB`."
        ),
    },
]


def render_model_card(user: str, cfg: Dict[str, Any]) -> str:
    repo_id = f"{user}/{cfg['repo_slug']}"
    lic_lines = [f"license: {cfg['license']}"]
    if cfg.get("license_name"):
        lic_lines.append(f"license_name: {cfg['license_name']}")
    if cfg.get("license_link"):
        lic_lines.append(f"license_link: {cfg['license_link']}")
    lic_yaml = "\n".join(lic_lines)

    flags = f" {cfg['server_flags']}" if cfg["server_flags"] else ""
    return f"""---
{lic_yaml}
base_model: {cfg['base_model']}
language:
  - en
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama-cpp
  - rpg
  - npc
  - tool-calling
  - gbnf
  - borrowed-stars
  - vocab-pruned
---

# {cfg['repo_slug']}

**Tier:** {cfg['tier']}
**Game:** *Borrowed Stars* (Anima RPG Engine — offline CPU NPC & Interactive Narrator model, v2.1 23-tool contract)

This is a fine-tuned, quantized `Q4_K_M` GGUF model trained to run **entirely on CPU (zero GPU contention)** inside an offline RPG, emitting single-shot structured JSON with in-character dialogue and `0..2` validated in-game tool calls:

```json
{{"mood":"friendly","line":"Aye, a rough patch. Here, take this salt loaf; it'll keep you from the wind.","calls":[{{"tool":"give_item","args":{{"item_id":"salt_loaf","qty":1}}}}]}}
```

## Benchmarks (Frozen 120-row 23-Tool v2.1 Eval + Real Laptop CPU)

| Metric | Value |
| :--- | :--- |
| **GGUF Size (`Q4_K_M`)** | `{cfg['size_mb']}` |
| **Parameters** | {cfg['params']} |
| **Held-out `val_bpb`** | `{cfg['val_bpb']}` |
| **Unconstrained JSON Schema** *(before GBNF)* | `{cfg['schema_rate']}` |
| **23-Tool F1** *(before GBNF)* | `{cfg['tool_f1']}` |
| **Anti-Exploit Refusal Pass Rate** *(6 refusal classes)* | `{cfg['refusal_rate']}` |
| **Pure Chitchat `calls: []` Discipline** | `{cfg['chitchat_zero']}` |
| **Retired-Name Kill-List / OOC Leaks** | `{cfg['kill_leak']}` |
| **Laptop CPU Speed** | {cfg['laptop_speed']} |
| **Live `llama.cpp` + `npc_grammar_v2.gbnf`** | `4/4` valid (`give_item`, `steer_dialogue_node`, refusal, chitchat) |

{cfg['extra_note']}

## Quick Start

### 1. Run with `llama-server` (native GBNF + OpenAI `/v1/*`)
```bash
llama-server -hf {repo_id}:Q4_K_M -t 6 -c 2048 --port 8091{flags}
```

### 2. Pull with `muse`
```bash
muse pull hf://{repo_id}@q4_k_m
```

### 3. Prompt Format (Raw Completion — `/completion` or `/v1/completions`)
Prepend the 1-line system prompt, followed by the compact state header (`[LOC]`, `[NPC]`, `[VOICE]`, `[NPC_INV]`, `[PLAYER]`, `[QUESTS]`, `[FACTS]`, optional `[BIO]`/`[RECENT]`/`[PEER_CHAT]`/`[TREE]`, `[ALLOWED_TOOLS]`, `[HIST]`, `[SAY]`, `JSON:`):

```text
You are the Borrowed Stars NPC & Interactive Narrator engine. Emit ONLY a single valid JSON object {{"mood", "line", "calls"}} with 0..2 authorized tool calls from [ALLOWED_TOOLS]. Never use non-canon names or break character.
[LOC] Windward Harbor · Market Square (windward_harbor) | morning | light_rain
[NPC] Maren | role=baker | mood=friendly | trust=2 | doing=tending_oven | carrying=salt_loaf
[VOICE] warm, practical, concise
[NPC_INV] salt_loaf(2), traveler_rations(1)
[PLAYER] gold=42 | bounty=0 | items=trade_writ(1) | hp=18/60 (wounded)
[QUESTS] active=act1_the_uncounted_star | offerable=none
[FACTS]
- The Tern sails across the Crown Sea once the tide turns.
[ALLOWED_TOOLS] give_item(salt_loaf,traveler_rations), adjust_trust, emote
[HIST] none
[SAY] Could you spare a salt loaf for the road?
JSON:
```

Pass the included `npc_grammar_v2.gbnf` (or a per-turn dynamically narrowed subset) in `grammar` to guarantee 100% schema and item-ID compliance.

## License
{cfg['license_note']}.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="", help="HF username or org (defaults to authenticated whoami)")
    ap.add_argument("--tag", default="v2.1", help="Git tag to attach after upload (default v2.1)")
    ap.add_argument("--private-dataset", action="store_true", help="Also back up training/eval JSONL into a private HF dataset repo")
    ap.add_argument("--dry-run", action="store_true", help="Validate files and print planned uploads without pushing")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()
    user = args.namespace or me["name"]
    print(f"Authenticated as: {me['name']} (namespace={user}, pro={me.get('isPro')})")

    created_repos: List[str] = []
    for cfg in REPOS:
        gguf_path = MODELS_DIR / cfg["local_file"]
        if not gguf_path.is_file():
            raise FileNotFoundError(f"Missing local GGUF: {gguf_path}")
        repo_id = f"{user}/{cfg['repo_slug']}"
        card = render_model_card(user, cfg)
        print(f"\n==> {repo_id} ({gguf_path.stat().st_size / (1024*1024):.1f} MB)")
        if args.dry_run:
            print(f"  [dry-run] Would upload {gguf_path.name} -> {cfg['remote_file']}, npc_grammar_v2.gbnf, README.md")
            created_repos.append(repo_id)
            continue

        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Update {args.tag} model card",
        )
        if GRAMMAR_PATH.is_file():
            api.upload_file(
                path_or_fileobj=str(GRAMMAR_PATH),
                path_in_repo="npc_grammar_v2.gbnf",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add v2.1 23-tool GBNF grammar ({args.tag})",
            )
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=cfg["remote_file"],
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {args.tag} Q4_K_M GGUF ({cfg['size_mb']})",
        )
        if args.tag:
            try:
                api.create_tag(repo_id=repo_id, tag=args.tag, repo_type="model", exist_ok=True)
            except Exception as exc:
                print(f"  [warn] tag {args.tag}: {exc}")
        print(f"  Uploaded: https://huggingface.co/{repo_id}")
        created_repos.append(repo_id)

    # Create or update unified HF Collection
    if not args.dry_run:
        col = api.create_collection(
            title="Borrowed Stars — Offline CPU NPC Models",
            namespace=user,
            description=(
                "Sub-500 MB Q4_K_M GGUF NPC & Narrator models for Borrowed Stars: "
                "23 in-game tools, dynamic GBNF, 0% lore leaks, 48-125 tok/s on CPU."
            ),
            private=False,
            exists_ok=True,
        )
        for repo_id in created_repos:
            try:
                api.add_collection_item(
                    collection_slug=col.slug,
                    item_id=repo_id,
                    item_type="model",
                    exists_ok=True,
                )
            except Exception as exc:
                print(f"  [warn] collection item {repo_id}: {exc}")
        print(f"\nCollection: https://huggingface.co/collections/{col.slug}")

    if args.private_dataset:
        ds_id = f"{user}/borrowed-stars-npc-tools-v2.1"
        print(f"\n==> Private dataset backup: {ds_id}")
        if not args.dry_run:
            api.create_repo(repo_id=ds_id, repo_type="dataset", exist_ok=True, private=True)
            for rel in ("data/bs_tools_v2_train.jsonl", "data/bs_tools_v2_eval.jsonl", "data/borrowed_stars_tools_spec.json", "data/BORROWED_STARS_TOOLS_SPEC.md"):
                p = ROOT / rel
                if p.is_file():
                    api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=ds_id, repo_type="dataset")
            print(f"  Uploaded private dataset: https://huggingface.co/datasets/{ds_id}")


if __name__ == "__main__":
    main()
