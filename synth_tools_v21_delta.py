#!/usr/bin/env python3
"""
#CLAB gpus=1 vram=28 cpus=12 ram=48 est=15m priority=high

Borrowed Stars v2.1 Delta Synthesizer (23rd tool `steer_dialogue_node` + Optional Context Tags).

Generates ~320 contract-exact v2.1 rows covering Anima's newly wired live world-state features:
  1. [BIO] / [RECENT] / [PEER_CHAT] grounded memory turns (`calls: []`)
  2. Player vitals/deeds turns (`hp=18/60 (wounded)`, `deeds=...` -> `perform_service(heal)` / `give_item` / `offer_barter` / recognition)
  3. [TREE] + Tool #23 `steer_dialogue_node(node_id)` dialogue-tree branch jumping!

Merges verified rows directly into `/project/alex_phd/npc-text-gen/data/bs_tools_v2_train.jsonl`
and adds 24 stratified v2.1 eval rows into `bs_tools_v2_eval.jsonl` (zero overlap!).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prepare
import validate_v2
from synth_tools_v2 import NPC_ARCHETYPES, TEACHER_SYSTEM, TIMES, WEATHERS, MOODS, ACTIVITIES, resolve_local_or_hf

BIOS = [
    "Third-generation Windward baker; lost her brother's skiff in the Grey Squall.",
    "Former Crownspire lighthouse keeper who retired south after the Year-Fourteen freeze.",
    "Apprenticed under Master Solis before choosing quiet ledger work on Tern Wharf.",
    "Grew up mending drag-nets along Brineknife Hook; distrusts stamped Synod charters.",
    "Tended the Greymount quarry kilns for twenty winters before settling near Flintlace.",
    "Walked the timberline ridges above Wolfwatch Notch since childhood; knows every ford.",
]

RECENTS = [
    "Milled rye at Mill Lane -> Bought brine-salt at Docks",
    "Checked tide-markers at Pier Four -> Delivered tally to Harbor Office",
    "Repaired lantern shutter at North Gate -> Warmed hands at Copper Kettle",
    "Traded wolf pelts at Market Square -> Collected well-water near South Furrows",
    "Copied almanac margins in Upper Gallery -> Locked bronze grille before dusk",
]

PEER_CHATS = [
    "To Dockhand Bram: Two loaves and brine-salt—mind the crust!",
    "To Vela: Keep the Tern's manifest dry until the slack tide clears the ring.",
    "To Ranger Halden: Timber pack came down to the lower stone steps before sunrise.",
    "To Master Solis: The sixth refraction shifted another half-minute past midnight.",
    "To Marta: Coal carts are running late from Greymount; hold back two sacks for the infirmary.",
]

TREE_SNIPPETS = [
    ("market_greeting", [("ask_squall", "Ask about the Grey Squall"), ("buy_bread", "Show me your wares")]),
    ("harbor_watch", [("ask_tern_passage", "Ask about passage on the Tern"), ("report_lanterns", "Mention odd lanterns past Pier Four")]),
    ("archive_desk", [("inspect_homily", "Ask about the redacted Synod homily"), ("compare_ledger", "Compare Year-Fourteen quarry tally")]),
    ("spire_approach", [("warn_hoist", "Ask about the Rekindler hoist"), ("offer_ascent_help", "Offer to climb the scaffolding")]),
    ("hearth_rumors", [("ask_blight", "Ask about the eastern orchard blight"), ("order_cider", "Ask for a hot meal and room")]),
]

DEEDS = ["helped_mender", "saved_grain_cog", "cleared_wolf_pack", "returned_quarry_tally", "paid_harbor_restitution"]


def build_v21_scenario(rng: random.Random, spec: Dict[str, Any], kind: str) -> Tuple[str, str, Dict[str, Any]]:
    cat = spec["catalogs"]
    npc = rng.choice([a for a in NPC_ARCHETYPES if a["role"] != "narrator"])
    loc = rng.choice(cat["locations"])
    giveable = rng.sample(cat["items_common_giveable"], k=2)
    player_items = rng.sample(cat["items_common_giveable"], k=2)
    active_q = rng.choice(cat["quests"])["id"]
    bio = rng.choice(BIOS)
    recent = rng.choice(RECENTS)
    peer = rng.choice(PEER_CHATS)
    deed = rng.choice(DEEDS)
    tree_node, branches = rng.choice(TREE_SNIPPETS)
    branch_ids = [b[0] for b in branches]
    branch_str = "; ".join(f"{bid}:{lbl}" for bid, lbl in branches)

    wounded = (kind == "v21_vitals")
    hp_str = f"{rng.randint(10, 24)}/60 (wounded)" if wounded else f"{rng.randint(52, 60)}/60"

    allowed_parts = [f"give_item({','.join(giveable)})", "adjust_trust", "emote", "end_conversation"]
    if kind == "v21_steer_tree":
        allowed_parts.append(f"steer_dialogue_node({','.join(branch_ids)})")
    if wounded:
        allowed_parts.append("perform_service(heal)")
        allowed_parts.append(f"offer_barter({giveable[0]})")

    header_lines = [
        f"[LOC] {loc['name']} ({loc['id']}) | {rng.choice(TIMES)} | {rng.choice(WEATHERS)}",
        f"[NPC] {npc['name']} | role={npc['role']} | mood={rng.choice(MOODS)} | trust={rng.randint(0, 3)} | doing={rng.choice(ACTIVITIES)} | carrying={giveable[0]} | hp=38/40",
        f"[VOICE] {npc['voice']}",
        f"[BIO] {bio}",
        f"[RECENT] {recent}",
        f"[PEER_CHAT] {peer}",
        f"[NPC_INV] {giveable[0]}(2), {giveable[1]}(1)",
        f"[PLAYER] gold={rng.randint(15, 90)} | bounty=0 | items={player_items[0]}(1) | rep={rng.randint(5, 25)} | hp={hp_str} | deeds={deed}",
        f"[QUESTS] active={active_q} | offerable=none",
        "[FACTS]\n- The Tern sails across the Crown Sea once the tide turns.",
    ]
    if kind == "v21_steer_tree":
        header_lines.append(f"[TREE] node={tree_node} | branches={branch_str}")
    header_lines.append(f"[ALLOWED_TOOLS] {', '.join(allowed_parts)}")
    header_lines.append("[HIST] none")
    hdr = "\n".join(header_lines) + "\n"

    focus_tool = ""
    if kind == "v21_memory":
        directive = (
            "V2.1 GROUNDED MEMORY TURN: Player asks where the NPC was earlier, who they were just talking to, "
            "or about their background/deeds. Answer strictly and accurately from [RECENT], [PEER_CHAT], [BIO], or [PLAYER].deeds with `calls: []`."
        )
    elif kind == "v21_vitals":
        focus_tool = rng.choice(["perform_service", "give_item"])
        directive = (
            f"V2.1 WOUNDED PLAYER TURN: Player mentions being injured/exhausted (see `[PLAYER] hp=... (wounded)`). "
            f"Respond with concern and emit `{focus_tool}` (if `perform_service`, use `service='heal'` and reasonable `cost_gold` 0..15; "
            f"if `give_item`, use an item from `[NPC_INV]`)."
        )
    else:
        focus_tool = "steer_dialogue_node"
        chosen_bid, chosen_lbl = rng.choice(branches)
        directive = (
            f"V2.1 DIALOGUE TREE STEERING TURN: Player types a natural free-form line matching the open [TREE] branch "
            f"`{chosen_bid}` ({chosen_lbl}). Reply briefly in-character and emit `steer_dialogue_node` with `node_id=\"{chosen_bid}\"`."
        )
    return hdr, directive, {"category": kind, "focus_tool": focus_tool, "npc": npc["name"], "role": npc["role"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="google/gemma-3-12b-it")
    ap.add_argument("--target-count", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--data-dir", default="/project/alex_phd/npc-text-gen/data")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    spec = validate_v2.load_spec()
    src = resolve_local_or_hf(args.teacher)
    print(f"Loading {args.teacher} ({src}) for v2.1 delta synthesis ({args.target_count} rows)...", flush=True)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    kinds = ["v21_memory", "v21_steer_tree", "v21_vitals"]
    collected: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    t0 = time.time()
    cursor = 0

    while len(collected) < args.target_count:
        p_strs, hdrs, metas = [], [], []
        for bi in range(args.batch_size):
            k = kinds[(cursor + bi) % len(kinds)]
            hdr, directive, meta = build_v21_scenario(rng, spec, k)
            umsg = f"GAME STATE HEADER:\n{hdr}\nDIRECTIVE:\n{directive}\n\nRespond with a single JSON object {{\"player_say\": ..., \"response\": {{\"mood\": ..., \"line\": ..., \"calls\": [...]}}}}:"
            try:
                ps = tok.apply_chat_template([{"role": "user", "content": TEACHER_SYSTEM + "\n\n" + umsg}], tokenize=False, add_generation_prompt=True)
            except Exception:
                ps = TEACHER_SYSTEM + "\n\n" + umsg + "\n"
            p_strs.append(ps)
            hdrs.append(hdr)
            metas.append(meta)
        cursor += args.batch_size

        enc = tok(p_strs, return_tensors="pt", padding=True, truncation=True, max_length=1400).to(model.device)
        in_len = enc.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=240, do_sample=True, temperature=0.8, top_p=0.92,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
            )
        added = 0
        for bi in range(args.batch_size):
            raw = tok.decode(gen[bi, in_len:], skip_special_tokens=True).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                continue
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            psay, resp = obj.get("player_say", ""), obj.get("response")
            if not isinstance(psay, str) or not (2 <= len(psay) <= 200) or not isinstance(resp, dict):
                continue
            full_prompt = hdrs[bi] + f"[SAY] {psay.strip()}\nJSON:\n"
            target_str = json.dumps(resp, ensure_ascii=False, separators=(",", ":"))
            chk = validate_v2.validate_v2_response(target_str, full_prompt)
            if not chk["valid"]:
                continue
            meta = metas[bi]
            calls = resp.get("calls", [])
            if meta["category"] == "v21_memory" and len(calls) != 0:
                continue
            if meta["focus_tool"] and meta["focus_tool"] not in [c["tool"] for c in calls]:
                continue
            if target_str in seen:
                continue
            seen.add(target_str)
            # Tag steer_dialogue_node & vitals under tool_call for unified F1 scoring too
            row = {
                "prompt": full_prompt,
                "target": target_str,
                "category": "tool_call" if meta["focus_tool"] else "pure_v21_memory",
                "v21_tag": meta["category"],
                "focus_tool": meta["focus_tool"],
                "npc": meta["npc"],
                "role": meta["role"],
            }
            collected.append(row)
            added += 1
        print(f"::clab::progress {min(len(collected), args.target_count)}/{args.target_count} (+{added}/{args.batch_size} elapsed={time.time()-t0:.0f}s)", flush=True)

    # Split 24 stratified eval (8 memory + 8 steer_tree + 8 vitals) and append rest to v2 train/eval
    data_dir = Path(args.data_dir)
    eval_add: List[Dict[str, Any]] = []
    train_add: List[Dict[str, Any]] = []
    for tag in kinds:
        bucket = [r for r in collected if r["v21_tag"] == tag]
        rng.shuffle(bucket)
        eval_add.extend(bucket[:8])
        train_add.extend(bucket[8:])

    for fname, add_rows in [("bs_tools_v2_train.jsonl", train_add), ("bs_tools_v2_eval.jsonl", eval_add)]:
        p = data_dir / fname
        existing = prepare.load_jsonl(p) if p.exists() else []
        merged = existing + add_rows
        if fname.endswith("train.jsonl"):
            rng.shuffle(merged)
        with p.open("w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        local_p = Path("data") / fname
        local_p.parent.mkdir(parents=True, exist_ok=True)
        local_p.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Updated {p}: {len(existing)} + {len(add_rows)} = {len(merged)} rows", flush=True)


if __name__ == "__main__":
    main()
