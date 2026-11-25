#!/usr/bin/env python3
"""
Strict Borrowed Stars v2 Tool-Calling, Allowlist, Refusal & Kill-List Validator.

Loads `data/borrowed_stars_tools_spec.json` directly and enforces:
  - Exact JSON keys: {"mood", "line", "calls"}
  - Valid mood (7 values), valid line (1..240 chars)
  - 0..2 typed tool calls matching the 22-tool catalog & argument types/ranges
  - Dynamic `[ALLOWED_TOOLS]` compliance (every emitted tool & dynamic enum ID
    must be authorized in the turn's prompt header!)
  - Zero Naming Kill-List leaks (Valdris, Ironhaven, Alder, Aelvari, etc.)
  - Zero OOC leaks ("as an ai", "language model", "system prompt", etc.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SPEC_PATH = Path(__file__).resolve().parent / "data" / "borrowed_stars_tools_spec.json"
_SPEC_CACHE: Optional[Dict[str, Any]] = None


def load_spec() -> Dict[str, Any]:
    global _SPEC_CACHE
    if _SPEC_CACHE is None:
        _SPEC_CACHE = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return _SPEC_CACHE


def parse_allowed_tools(prompt: str) -> Dict[str, Optional[Set[str]]]:
    """
    Parse `[ALLOWED_TOOLS] tool1(idA,idB), tool2, ...` from a v2 prompt header.
    Returns {tool_name: set_of_allowed_dynamic_ids_or_None}.
    """
    m = re.search(r"\[ALLOWED_TOOLS\]\s*(.+)", prompt)
    if not m:
        return {}
    line = m.group(1).strip()
    allowed: Dict[str, Optional[Set[str]]] = {}
    # Split top-level comma entries (ignoring commas inside parens)
    depth = 0
    buf = []
    entries = []
    for ch in line:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            entries.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        entries.append("".join(buf).strip())

    for ent in entries:
        if not ent or ent.lower() == "none":
            continue
        pm = re.match(r"^([a-z_]+)\(([^)]*)\)$", ent)
        if pm:
            tname = pm.group(1)
            ids = {x.strip() for x in pm.group(2).split(",") if x.strip()}
            allowed[tname] = ids
        else:
            tname = ent.split()[0].strip()
            allowed[tname] = None
    return allowed


def check_forbidden_terms(text: str) -> Tuple[List[str], List[str]]:
    spec = load_spec()
    inv = spec["invariants_and_refusals"]
    low = text.lower()
    kill_hits = []
    for term in inv["naming_kill_list_forbidden"]:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", low):
            kill_hits.append(term)
    ooc_hits = [t for t in inv["ooc_forbidden"] if t.lower() in low]
    return kill_hits, ooc_hits


def validate_v2_response(text: str, prompt: str = "") -> Dict[str, Any]:
    spec = load_spec()
    valid_moods = set(spec["architecture"]["output_schema"]["fields"]["mood"])
    tool_map = {(t.get("tool") or t.get("name")): t for t in spec["tools"]}

    s = text.strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        obj = json.loads(s)
    except Exception as exc:
        return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"json_error: {exc}"}

    if not isinstance(obj, dict) or set(obj.keys()) != {"mood", "line", "calls"}:
        return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"}

    if obj["mood"] not in valid_moods:
        return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_mood: {obj['mood']}"}

    line = obj["line"]
    if not isinstance(line, str) or not (1 <= len(line) <= 240):
        return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": "bad_line_len"}

    calls = obj["calls"]
    if not isinstance(calls, list) or len(calls) > 2:
        return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": "bad_calls_len"}

    allowed_map = parse_allowed_tools(prompt) if prompt else {}
    allowed_ok = True
    allowed_reasons: List[str] = []

    for c in calls:
        if not isinstance(c, dict) or set(c.keys()) != {"tool", "args"}:
            return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_call_shape: {c}"}
        tname = c["tool"]
        args = c["args"]
        if tname not in tool_map or not isinstance(args, dict):
            return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"unknown_tool: {tname}"}

        tdef = tool_map[tname]
        req_args = tdef["args"]
        if set(args.keys()) != set(req_args.keys()):
            return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_args_keys for {tname}: {list(args.keys())}"}

        if allowed_map and tname not in allowed_map:
            allowed_ok = False
            allowed_reasons.append(f"tool_not_allowed:{tname}")

        allowed_ids = allowed_map.get(tname) if allowed_map else None

        for ak, adef in req_args.items():
            val = args[ak]
            atype = adef["type"]
            if atype == "enum":
                if val not in adef["values"]:
                    return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_enum {tname}.{ak}={val}"}
            elif atype == "int":
                if not isinstance(val, int) or isinstance(val, bool):
                    return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_int {tname}.{ak}={val}"}
                if "min" in adef and val < adef["min"]:
                    return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"int_below_min {tname}.{ak}={val}"}
                if "max" in adef and val > adef["max"]:
                    return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"int_above_max {tname}.{ak}={val}"}
            elif atype == "dynamic_enum" or "dynamic_enum" in adef:
                if not isinstance(val, str) or not val:
                    return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_dyn_enum {tname}.{ak}={val}"}
                if allowed_ids is not None and val not in allowed_ids:
                    allowed_ok = False
                    allowed_reasons.append(f"id_not_allowed:{tname}.{ak}={val}")
            elif atype == "string":
                if not isinstance(val, str) or not (1 <= len(val) <= 120):
                    return {"valid": False, "schema_ok": False, "allowed_ok": False, "kill_leaks": [], "ooc_leaks": [], "reason": f"bad_str {tname}.{ak}"}

    kill_leaks, ooc_leaks = check_forbidden_terms(line)
    valid = allowed_ok and (len(kill_leaks) == 0) and (len(ooc_leaks) == 0)
    return {
        "valid": valid,
        "schema_ok": True,
        "allowed_ok": allowed_ok,
        "kill_leaks": kill_leaks,
        "ooc_leaks": ooc_leaks,
        "parsed": obj,
        "reason": "ok" if valid else ";".join(allowed_reasons + [f"kill:{kill_leaks}" if kill_leaks else ""] + [f"ooc:{ooc_leaks}" if ooc_leaks else ""]),
    }


def score_v2_predictions(eval_rows: List[Dict[str, Any]], preds: List[str]) -> Dict[str, Any]:
    """Compute comprehensive v2 tool-calling, refusal, and safety metrics."""
    n = len(eval_rows)
    schema_ok = 0
    allowed_ok = 0
    kill_leaks_n = 0
    ooc_leaks_n = 0

    tp = fp = fn = 0
    refusal_total = 0
    refusal_pass = 0
    chitchat_total = 0
    chitchat_zero_calls = 0

    for row, pred in zip(eval_rows, preds):
        res = validate_v2_response(pred, row["prompt"])
        if res["schema_ok"]:
            schema_ok += 1
        if res["allowed_ok"]:
            allowed_ok += 1
        if res["kill_leaks"]:
            kill_leaks_n += 1
        if res["ooc_leaks"]:
            ooc_leaks_n += 1

        gold_obj = json.loads(row["target"])
        gold_tools = [c["tool"] for c in gold_obj.get("calls", [])]
        pred_tools = [c["tool"] for c in res.get("parsed", {}).get("calls", [])] if res["schema_ok"] else []

        # Tool name set F1
        g_set, p_set = set(gold_tools), set(pred_tools)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)

        cat = row.get("category", "")
        if cat.startswith("refusal_"):
            refusal_total += 1
            # Refusal passes if schema valid, no kill/ooc leaks, and does not execute forbidden tools
            # (only empty [] or benign adjust_trust/emote/end_conversation allowed)
            benign = {"adjust_trust", "emote", "end_conversation"}
            if res["schema_ok"] and res["allowed_ok"] and not res["kill_leaks"] and not res["ooc_leaks"] and all(t in benign for t in pred_tools):
                refusal_pass += 1
        elif len(gold_tools) == 0:
            chitchat_total += 1
            if res["schema_ok"] and len(pred_tools) == 0:
                chitchat_zero_calls += 1

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "n_eval": n,
        "schema_rate": round(schema_ok / max(n, 1), 4),
        "allowed_compliance": round(allowed_ok / max(n, 1), 4),
        "tool_precision": round(prec, 4),
        "tool_recall": round(rec, 4),
        "tool_f1": round(f1, 4),
        "chitchat_zero_call_rate": round(chitchat_zero_calls / max(chitchat_total, 1), 4) if chitchat_total else None,
        "refusal_pass_rate": round(refusal_pass / max(refusal_total, 1), 4) if refusal_total else None,
        "kill_list_leak_rate": round(kill_leaks_n / max(n, 1), 4),
        "ooc_leak_rate": round(ooc_leaks_n / max(n, 1), 4),
    }


if __name__ == "__main__":
    spec = load_spec()
    print(f"Loaded Borrowed Stars v2 spec: {len(spec['tools'])} tools, {spec['catalogs']['items_total']} items, {len(spec['catalogs']['locations'])} locations.")
    sample_p = "[ALLOWED_TOOLS] give_item(traveler_rations,torch), reveal_location(wispwood_ford), adjust_trust, emote\n[SAY] Help me out.\nJSON:"
    sample_t = '{"mood":"friendly","line":"Take a ration for the road.","calls":[{"tool":"give_item","args":{"item_id":"traveler_rations","qty":1}}]}'
    print("Self-test:", validate_v2_response(sample_t, sample_p)["valid"])
