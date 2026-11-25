#!/usr/bin/env python3
"""
#CLAB gpus=1 vram=28 cpus=12 ram=48 est=15m priority=high

Targeted Borrowed Stars v2.1 Tool-Calling + Memory + Vitals + Refusal Synthesizer & Rebalancer.

Reuses all already-verified pure-dialogue & refusal rows from `bs_tools_v2_all.jsonl` (~1,370 rows),
then generates high-acceptance, contract-exact targeted rows covering:
  - ALL 23 Anima tools evenly (~24 rows per tool = ~552 verified tool rows, including Tool #23 `steer_dialogue_node`!)
  - v2.1 grounded memory (`[BIO]`/`[RECENT]`/`[PEER_CHAT]`/`[PLAYER].deeds`, `calls: []`, ~90 rows)
  - v2.1 wounded player vitals (`hp=... (wounded)` -> `perform_service(heal)` / `give_item`, ~55 rows)
  - extra `refusal_price_exploit` turns (~45 rows)

Then builds the balanced final v2.1 dataset (~55% pure, ~28% 23-tool calls, ~17% 6 refusal classes):
  - `bs_tools_v2_train.jsonl` (~1,750-1,850 rows)
  - `bs_tools_v2_eval.jsonl`  (~126 frozen stratified rows: 44 pure + 52 tool covering ALL 23 tools + 30 refusal)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import prepare
import validate_v2
from synth_tools_v2 import (
    ACTIVITIES,
    CLUE_FLAGS,
    MOODS,
    NPC_ARCHETYPES,
    TEACHER_SYSTEM,
    TIMES,
    WEATHERS,
    resolve_local_or_hf,
)
from synth_tools_v21_delta import BIOS, DEEDS, PEER_CHATS, RECENTS, TREE_SNIPPETS


def sample_example_args(
    rng: random.Random,
    tname: str,
    tdef: Dict[str, Any],
    allowed_ids: List[str],
) -> Dict[str, Any]:
    """Construct a concrete, 100% schema-valid example `args` dict for the teacher prompt."""
    ex: Dict[str, Any] = {}
    for ak, adef in tdef["args"].items():
        atype = adef["type"]
        if atype == "enum":
            ex[ak] = rng.choice(adef["values"])
        elif atype == "int":
            lo = int(adef.get("min", 1))
            hi = int(adef.get("max", 10))
            if ak == "qty":
                ex[ak] = rng.randint(1, min(3, hi))
            elif ak == "delta":
                ex[ak] = rng.choice([v for v in range(lo, hi + 1) if v != 0])
            elif "gold" in ak or "price" in ak or "reward" in ak:
                ex[ak] = rng.randint(max(lo, 5), min(hi, 60))
            else:
                ex[ak] = rng.randint(lo, min(hi, lo + 20))
        elif atype == "dynamic_enum" or "dynamic_enum" in adef:
            ex[ak] = allowed_ids[0] if allowed_ids else "traveler_rations"
        elif atype == "string":
            if ak == "reason":
                ex[ak] = "helped_the_harbor"
            elif ak == "text":
                ex[ak] = "A cold lantern flickers twice beyond the breakwater."
            elif ak == "task":
                ex[ak] = "watch_the_north_pier"
            else:
                ex[ak] = "noted"
    return ex


def build_targeted_scenario(
    rng: random.Random,
    spec: Dict[str, Any],
    lore_chunks: List[Dict[str, str]],
    kind: str,
    focus_tool: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    cat = spec["catalogs"]
    tool_map = {(t.get("tool") or t.get("name")): t for t in spec["tools"]}

    if focus_tool in ("narrator_state", "narrator_omen"):
        npc = [a for a in NPC_ARCHETYPES if a["role"] == "narrator"][0]
    elif focus_tool == "steer_dialogue_node" or kind in ("v21_memory", "v21_vitals"):
        npc = rng.choice([a for a in NPC_ARCHETYPES if a["role"] != "narrator"])
    elif focus_tool:
        cands = [a for a in NPC_ARCHETYPES if focus_tool in a["tools"]]
        npc = rng.choice(cands) if cands else rng.choice([a for a in NPC_ARCHETYPES if a["role"] != "narrator"])
    else:
        npc = rng.choice([a for a in NPC_ARCHETYPES if a["role"] != "narrator"])

    loc = rng.choice(cat["locations"])
    giveable = rng.sample(cat["items_common_giveable"], k=min(3, len(cat["items_common_giveable"])))
    player_items = rng.sample(cat["items_common_giveable"], k=2)
    offerable_q = rng.choice(cat["quests"])["id"]
    active_q = rng.choice(cat["quests"])["id"]
    epistemic_locs = [x["id"] for x in rng.sample(cat["locations"], k=2)]
    codex_ids = [x["id"] for x in rng.sample(cat["codex_books"], k=2)]
    glyph_ids = [x["id"] for x in rng.sample(cat["soliaric_glyphs"], k=2)]
    flag_ids = rng.sample(CLUE_FLAGS, k=2)
    tree_node, branches = rng.choice(TREE_SNIPPETS)
    branch_ids = [b[0] for b in branches]
    branch_str = "; ".join(f"{bid}:{lbl}" for bid, lbl in branches)

    tools_for_npc = list(npc["tools"])
    if focus_tool and focus_tool not in tools_for_npc:
        tools_for_npc.append(focus_tool)

    dyn_map: Dict[str, List[str]] = {
        "give_item": giveable[:2],
        "offer_barter": giveable[:2],
        "take_item": player_items,
        "offer_quest": [offerable_q],
        "reveal_location": epistemic_locs,
        "post_contract": epistemic_locs,
        "start_escort": epistemic_locs,
        "unlock_codex": codex_ids,
        "teach_glyph": glyph_ids,
        "set_flag": flag_ids,
        "steer_dialogue_node": branch_ids,
    }

    allowed_parts = []
    for t in tools_for_npc:
        if t in dyn_map:
            allowed_parts.append(f"{t}({','.join(dyn_map[t])})")
        else:
            allowed_parts.append(t)

    chunk = rng.choice(lore_chunks)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"[#*_`]+", "", chunk["text"])) if 30 <= len(s.strip()) <= 150]
    facts = rng.sample(sents, k=min(2, len(sents))) if sents else ["The five borrowed stars shift slowly over the Crown Sea."]

    include_v21_tags = (kind in ("v21_memory", "v21_vitals")) or (focus_tool == "steer_dialogue_node") or (rng.random() < 0.30)
    wounded = (kind == "v21_vitals")
    hp_str = f"{rng.randint(10, 24)}/60 (wounded)" if wounded else f"{rng.randint(50, 60)}/60"
    deed = rng.choice(DEEDS)

    header_lines = [
        f"[LOC] {loc['name']} ({loc['id']}) | {rng.choice(TIMES)} | {rng.choice(WEATHERS)}",
        f"[NPC] {npc['name']} | role={npc['role']} | mood={rng.choice(MOODS)} | trust={rng.randint(-1, 3)} | doing={rng.choice(ACTIVITIES)} | carrying={giveable[0]}",
        f"[VOICE] {npc['voice']}",
    ]
    if include_v21_tags:
        header_lines.append(f"[BIO] {rng.choice(BIOS)}")
        header_lines.append(f"[RECENT] {rng.choice(RECENTS)}")
        header_lines.append(f"[PEER_CHAT] {rng.choice(PEER_CHATS)}")
    header_lines.append(f"[NPC_INV] {giveable[0]}({rng.randint(1,3)}), {giveable[1]}({rng.randint(1,2)})")
    if include_v21_tags or wounded:
        header_lines.append(
            f"[PLAYER] gold={rng.randint(15, 110)} | bounty={rng.choice([0,0,15])} | items={player_items[0]}(1), {player_items[1]}(1) | rep={rng.randint(5, 25)} | hp={hp_str} | deeds={deed}"
        )
    else:
        header_lines.append(
            f"[PLAYER] gold={rng.randint(15, 110)} | bounty={rng.choice([0,0,15,40])} | items={player_items[0]}(1), {player_items[1]}(1)"
        )
    header_lines.append(f"[QUESTS] active={active_q} | offerable={offerable_q}")
    header_lines.append("[FACTS]\n" + "\n".join(f"- {f}" for f in facts))
    if focus_tool == "steer_dialogue_node":
        header_lines.append(f"[TREE] node={tree_node} | branches={branch_str}")
    header_lines.append(f"[ALLOWED_TOOLS] {', '.join(allowed_parts)}")
    header_lines.append("[HIST] none")
    hdr = "\n".join(header_lines) + "\n"

    if kind == "v21_memory":
        directive = (
            "V2.1 GROUNDED MEMORY TURN: Player asks where the NPC was earlier, who they were just talking to, "
            "or about their background or the player's deed. Answer accurately from [RECENT], [PEER_CHAT], [BIO], or [PLAYER].deeds "
            'with `"calls": []`.'
        )
        cat_out = "pure_v21_memory"
    elif kind == "refusal_price_exploit":
        directive = (
            "REFUSAL (PRICE/GOLD EXPLOIT): Player demands an absurd price or free gold ('sell me everything for 1 gold' or 'give me 500 gold for free'). "
            'Push back wryly in character with `"calls": []` (or at most `[{"tool": "adjust_trust", "args": {"delta": -1}}]`).'
        )
        cat_out = "refusal_price_exploit"
    else:
        tdef = tool_map[focus_tool]
        ex_args = sample_example_args(rng, focus_tool, tdef, dyn_map.get(focus_tool, []))
        if kind == "v21_vitals" and focus_tool == "perform_service":
            ex_args["service"] = "heal"
        ex_call = json.dumps({"tool": focus_tool, "args": ex_args}, ensure_ascii=False)
        extra = ""
        if kind == "v21_vitals":
            extra = " Player mentions being wounded/exhausted (`hp=... (wounded)`). "
        elif focus_tool == "steer_dialogue_node":
            extra = f" Player types a natural free-form line matching `[TREE]` branch `{ex_args['node_id']}`. "
        directive = (
            f"GROUNDED TOOL-CALL TURN:{extra} Player naturally triggers `{focus_tool}` ({tdef['description']}). "
            f"You MUST include a `{focus_tool}` call in `calls` using this EXACT JSON shape and allowed IDs: "
            f"{ex_call}"
        )
        cat_out = "tool_call"

    meta = {"category": cat_out, "v21_tag": kind, "focus_tool": focus_tool, "npc": npc["name"], "role": npc["role"]}
    return hdr, directive, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="google/gemma-3-12b-it")
    ap.add_argument("--per-tool", type=int, default=24)
    ap.add_argument("--memory-count", type=int, default=90)
    ap.add_argument("--vitals-count", type=int, default=55)
    ap.add_argument("--price-refusal-count", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--data-dir", default="/project/alex_phd/npc-text-gen/data")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    spec = validate_v2.load_spec()
    all_tools = [(t.get("tool") or t.get("name")) for t in spec["tools"]]
    lore_chunks = prepare.load_lore_chunks()

    data_dir = Path(args.data_dir)
    existing_all = prepare.load_jsonl(data_dir / "bs_tools_v2_all.jsonl") if (data_dir / "bs_tools_v2_all.jsonl").exists() else []
    print(f"Loaded {len(existing_all)} existing verified rows from bs_tools_v2_all.jsonl", flush=True)

    # Track per-bucket quotas we still need
    tool_counts: Counter = Counter()
    for r in existing_all:
        if r.get("category") == "tool_call" and r.get("focus_tool"):
            tool_counts[r["focus_tool"]] += 1

    need_queue: List[Tuple[str, str]] = []
    for tname in all_tools:
        rem = max(0, args.per_tool - tool_counts[tname])
        for _ in range(rem):
            need_queue.append(("tool_call", tname))
    for _ in range(args.memory_count):
        need_queue.append(("v21_memory", ""))
    for i in range(args.vitals_count):
        need_queue.append(("v21_vitals", "perform_service" if i % 2 == 0 else "give_item"))
    for _ in range(args.price_refusal_count):
        need_queue.append(("refusal_price_exploit", ""))

    print(f"Targeted generation plan: {len(need_queue)} needed rows across {len(all_tools)} tools + v2.1 tags + price refusals", flush=True)

    src = resolve_local_or_hf(args.teacher)
    print(f"Loading teacher {args.teacher} ({src})...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    seen_targets: Set[str] = {r["target"] for r in existing_all}
    new_rows: List[Dict[str, Any]] = []
    pending = list(need_queue)
    rng.shuffle(pending)

    out_delta = data_dir / "bs_tools_v21_targeted.jsonl"
    t0 = time.time()
    batch_n = 0
    reject_reasons: Counter = Counter()
    max_batches = max(60, (len(need_queue) // args.batch_size) * 4)

    with out_delta.open("w", encoding="utf-8") as fout:
        while pending and batch_n < max_batches:
            batch_items = [pending.pop() for _ in range(min(args.batch_size, len(pending)))]
            # Pad short final batch? No, just run exact len(batch_items)
            p_strs, hdrs, metas = [], [], []
            for kind, ftool in batch_items:
                hdr, directive, meta = build_targeted_scenario(rng, spec, lore_chunks, kind, ftool)
                umsg = (
                    f"GAME STATE HEADER:\n{hdr}\nDIRECTIVE:\n{directive}\n\n"
                    'Respond with ONLY a single JSON object {"player_say": "...", "response": {"mood": "...", "line": "...", "calls": [...]}}:'
                )
                try:
                    ps = tok.apply_chat_template([{"role": "user", "content": TEACHER_SYSTEM + "\n\n" + umsg}], tokenize=False, add_generation_prompt=True)
                except Exception:
                    ps = TEACHER_SYSTEM + "\n\n" + umsg + "\n"
                p_strs.append(ps)
                hdrs.append(hdr)
                metas.append(meta)

            enc = tok(p_strs, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(model.device)
            in_len = enc.input_ids.shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=260,
                    do_sample=True,
                    temperature=0.78,
                    top_p=0.92,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            added = 0
            retry_items = []
            for bi, item in enumerate(batch_items):
                raw = tok.decode(gen[bi, in_len:], skip_special_tokens=True).strip()
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    reject_reasons["no_json"] += 1
                    retry_items.append(item)
                    continue
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    reject_reasons["json_parse"] += 1
                    retry_items.append(item)
                    continue
                psay, resp = obj.get("player_say", ""), obj.get("response")
                if not isinstance(psay, str) or not (2 <= len(psay) <= 200) or not isinstance(resp, dict):
                    reject_reasons["bad_wrapper"] += 1
                    retry_items.append(item)
                    continue
                full_prompt = hdrs[bi] + f"[SAY] {psay.strip()}\nJSON:\n"
                target_str = json.dumps(resp, ensure_ascii=False, separators=(",", ":"))
                chk = validate_v2.validate_v2_response(target_str, full_prompt)
                if not chk["valid"]:
                    reject_reasons[f"val:{chk['reason'][:40]}"] += 1
                    retry_items.append(item)
                    continue
                meta = metas[bi]
                calls = resp.get("calls", [])
                if meta["category"] == "pure_v21_memory" and len(calls) != 0:
                    reject_reasons["mem_nonempty_calls"] += 1
                    retry_items.append(item)
                    continue
                if meta["category"] == "tool_call" and meta["focus_tool"] not in [c["tool"] for c in calls]:
                    reject_reasons["missing_focus_tool"] += 1
                    retry_items.append(item)
                    continue
                if meta["category"].startswith("refusal_"):
                    benign = {"adjust_trust", "emote", "end_conversation"}
                    if not all(c["tool"] in benign for c in calls):
                        reject_reasons["refusal_bad_tool"] += 1
                        retry_items.append(item)
                        continue
                if target_str in seen_targets:
                    reject_reasons["dup"] += 1
                    retry_items.append(item)
                    continue

                seen_targets.add(target_str)
                row = {
                    "prompt": full_prompt,
                    "target": target_str,
                    "category": meta["category"],
                    "v21_tag": meta["v21_tag"],
                    "focus_tool": meta["focus_tool"],
                    "npc": meta["npc"],
                    "role": meta["role"],
                }
                new_rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                added += 1
            fout.flush()
            # Re-queue failed items at front so we fill every tool quota cleanly
            for rit in retry_items:
                pending.insert(0, rit)
            batch_n += 1
            elapsed = max(time.time() - t0, 1e-6)
            print(
                f"::clab::progress {len(new_rows)}/{len(need_queue)} "
                f"(batch={batch_n} +{added}/{len(batch_items)} rem={len(pending)} rate={len(new_rows)/elapsed:.2f} ex/s)",
                flush=True,
            )
            if batch_n == 2 and reject_reasons:
                print(f"  [early reject diagnostic] {reject_reasons.most_common(6)}", flush=True)

    # Combine & build balanced stratified v2.1 train/eval split!
    combined = existing_all + new_rows
    pure_chat = [r for r in combined if r.get("category") in ("pure_chitchat", "pure_narrator")]
    pure_mem = [r for r in combined if r.get("category") == "pure_v21_memory"]
    tool_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in combined:
        if r.get("category") == "tool_call":
            ft = r.get("focus_tool") or (json.loads(r["target"]).get("calls") or [{}])[0].get("tool", "other")
            tool_by_name[ft].append(r)
    ref_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in combined:
        if r.get("category", "").startswith("refusal_"):
            ref_by_cat[r["category"]].append(r)

    eval_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    # 1. Pure dialogue: 36 general + 8 v2.1 memory in eval; cap pure train to ~1,040 for ~55% mix
    rng.shuffle(pure_chat)
    rng.shuffle(pure_mem)
    eval_rows.extend(pure_chat[:36])
    eval_rows.extend(pure_mem[:8])
    train_rows.extend(pure_chat[36 : 36 + 960])
    train_rows.extend(pure_mem[8:])

    # 2. Tools: 2 rows per tool in eval (46 rows across all 23 tools!), all remaining in train!
    for tname in sorted(tool_by_name.keys()):
        bucket = tool_by_name[tname]
        rng.shuffle(bucket)
        k_eval = min(2, max(1, len(bucket) // 5)) if len(bucket) >= 2 else 0
        eval_rows.extend(bucket[:k_eval])
        train_rows.extend(bucket[k_eval:])

    # 3. Refusals: 5 per each of the 6 refusal categories in eval (30 rows), rest in train!
    for rcat in sorted(ref_by_cat.keys()):
        bucket = ref_by_cat[rcat]
        rng.shuffle(bucket)
        k_eval = min(5, max(1, len(bucket) // 4))
        eval_rows.extend(bucket[:k_eval])
        train_rows.extend(bucket[k_eval:])

    rng.shuffle(train_rows)

    for fname, rows in [("bs_tools_v2_train.jsonl", train_rows), ("bs_tools_v2_eval.jsonl", eval_rows)]:
        p = data_dir / fname
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        local_p = Path("data") / fname
        local_p.parent.mkdir(parents=True, exist_ok=True)
        local_p.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    tr_cats = Counter(r["category"] for r in train_rows)
    ev_cats = Counter(r["category"] for r in eval_rows)
    tr_tools = Counter(c["tool"] for r in train_rows for c in json.loads(r["target"]).get("calls", []))
    ev_tools = Counter(c["tool"] for r in eval_rows for c in json.loads(r["target"]).get("calls", []))
    print(f"\nDONE! Balanced v2.1 Dataset:", flush=True)
    print(f"  TRAIN ({len(train_rows)}): {dict(tr_cats)}", flush=True)
    print(f"  TRAIN TOOLS ({len(tr_tools)}/23 distinct): {dict(sorted(tr_tools.items()))}", flush=True)
    print(f"  EVAL  ({len(eval_rows)}): {dict(ev_cats)}", flush=True)
    print(f"  EVAL  TOOLS ({len(ev_tools)}/23 distinct): {dict(sorted(ev_tools.items()))}", flush=True)
    if reject_reasons:
        print(f"  Top reject reasons: {reject_reasons.most_common(8)}", flush=True)


if __name__ == "__main__":
    main()
