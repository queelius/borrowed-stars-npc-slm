#!/usr/bin/env python3
"""
#CLAB gpus=1 vram=32 cpus=12 ram=48 est=60m priority=high

Stage 2 Tool-Calling SFT + On-Policy Distillation (OPD) + 32K Vocab Pruning + GGUF Q4_K_M Export.

Stacks Borrowed Stars v2 tool-calling (`{"mood", "line", "calls"}`, 22 tools, refusals,
0% Naming Kill-List leaks) directly on top of our Stage 1 Borrowed Stars merged checkpoints:
  - google/gemma-3-1b-it   (Stage 1 #1 Quality Champ -> ~428 MB at 32K vocab)
  - Qwen/Qwen3.5-0.8B      (Stage 1 #1 Apache 2.0 Champ -> ~310 MB at 32K vocab)
  - LiquidAI/LFM2-700M     (Stage 1 Mid-Size Hybrid -> ~437 MB)
  - LiquidAI/LFM2.5-350M   (Stage 1 Ultra-Tiny Speed Champ -> ~210 MB)

Evaluates every model on frozen 100-row `bs_tools_v2_eval.jsonl` (`val_bpb`, `schema_rate`,
`allowed_compliance`, `tool_f1`, `chitchat_zero_call_rate`, `refusal_pass_rate`,
`kill_list_leak_rate`), saves final merged checkpoints, and exports ready-to-play Q4_K_M GGUFs!
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import prepare
import validate_v2
from shootout import detect_lora_targets, load_slm, resolve_local_or_hf

V2_SYSTEM = (
    "You are the Borrowed Stars NPC & Interactive Narrator engine. "
    "Emit ONLY a single valid JSON object {\"mood\", \"line\", \"calls\"} with 0..2 authorized tool calls from [ALLOWED_TOOLS]. "
    "Never use non-canon names or break character.\n"
)


@torch.no_grad()
def eval_v2(model, tok, eval_rows: List[Dict[str, Any]], device: torch.device, batch_gen: int = 8) -> Dict[str, Any]:
    model.eval()

    def nll_fn(prompt: str, target: str) -> float:
        p_ids = tok(V2_SYSTEM + prompt, add_special_tokens=True).input_ids
        t_ids = tok(target, add_special_tokens=False).input_ids
        if not t_ids:
            return 0.0
        inp = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
        lg = model(input_ids=inp).logits[0, len(p_ids) - 1 : len(p_ids) + len(t_ids) - 1, :].float()
        return float(F.cross_entropy(lg, torch.tensor(t_ids, dtype=torch.long, device=device), reduction="sum").item())

    bpb = prepare.compute_val_bpb(eval_rows, nll_fn)["val_bpb"]

    # Batched greedy generation across all 100 eval prompts for fast full scoring
    orig_pad_side = tok.padding_side
    tok.padding_side = "left"
    preds: List[str] = []
    for i in range(0, len(eval_rows), batch_gen):
        chunk = eval_rows[i : i + batch_gen]
        texts = [V2_SYSTEM + r["prompt"] for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(device)
        in_len = enc.input_ids.shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        for bi in range(len(chunk)):
            out_txt = tok.decode(gen[bi, in_len:], skip_special_tokens=True).strip()
            first = out_txt.splitlines()[0].strip() if out_txt else ""
            preds.append(first)
    tok.padding_side = orig_pad_side

    scores = validate_v2.score_v2_predictions(eval_rows, preds)
    scores["val_bpb"] = round(bpb, 5)
    # Grab 6 diverse sample predictions (2 pure, 2 tool, 2 refusal)
    samples = []
    seen_cats = {"pure": 0, "tool": 0, "refusal": 0}
    for row, pred in zip(eval_rows, preds):
        c = row.get("category", "")
        bucket = "refusal" if c.startswith("refusal_") else ("tool" if c == "tool_call" else "pure")
        if seen_cats[bucket] < 2:
            seen_cats[bucket] += 1
            samples.append({"category": c, "npc": row.get("npc"), "gold": row["target"], "pred": pred})
    scores["samples"] = samples
    return scores


def train_one(
    slug: str,
    ckpt_dir: Path,
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    out_dir: Path,
    epochs: int,
    lr: float,
    lora_r: int,
    teacher_repo: str,
    opd_steps: int,
    export_gguf: bool,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"\n==================== STAGE 2 TOOLS: {slug} ====================", flush=True)

    tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_slm(str(ckpt_dir), torch.bfloat16, device)

    targets = detect_lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(r=lora_r, lora_alpha=lora_r * 2, target_modules=targets, lora_dropout=0.05, task_type="CAUSAL_LM"),
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    grad_accum = 8
    total_steps = epochs * len(train_rows)
    step = 0
    best_bpb = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.Random(100 + ep).shuffle(order)
        ep_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for idx_i, row_i in enumerate(order):
            row = train_rows[row_i]
            p_ids = tok(V2_SYSTEM + row["prompt"], add_special_tokens=True).input_ids
            t_ids = tok(row["target"] + (tok.eos_token or ""), add_special_tokens=False).input_ids
            if not t_ids:
                continue
            inp = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
            lg = model(input_ids=inp).logits[0, len(p_ids) - 1 : len(p_ids) + len(t_ids) - 1, :].float()
            loss = F.cross_entropy(lg, torch.tensor(t_ids, dtype=torch.long, device=device), label_smoothing=0.05)
            (loss / grad_accum).backward()
            ep_loss += float(loss.item())
            step += 1
            if step % grad_accum == 0 or idx_i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                prog = step / max(total_steps, 1)
                lr_now = lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
                opt.step()
                opt.zero_grad(set_to_none=True)

        # Quick val_bpb check after each epoch
        model.eval()
        with torch.no_grad():
            def _nll(p: str, t: str) -> float:
                pi = tok(V2_SYSTEM + p, add_special_tokens=True).input_ids
                ti = tok(t, add_special_tokens=False).input_ids
                inp = torch.tensor([pi + ti], dtype=torch.long, device=device)
                l = model(input_ids=inp).logits[0, len(pi) - 1 : len(pi) + len(ti) - 1, :].float()
                return float(F.cross_entropy(l, torch.tensor(ti, dtype=torch.long, device=device), reduction="sum").item())
            ep_bpb = prepare.compute_val_bpb(eval_rows, _nll)["val_bpb"]
        print(f"  [TOOLS SFT ep {ep+1}/{epochs}] loss={ep_loss/max(len(train_rows),1):.4f} val_bpb={ep_bpb:.5f}", flush=True)
        if ep_bpb < best_bpb:
            best_bpb = ep_bpb
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print(f"  [BEST-EPOCH] Restored best LoRA state (val_bpb={best_bpb:.5f})", flush=True)

    pre_opd_metrics = eval_v2(model, tok, eval_rows, device)
    print(
        f"[PRE-OPD EVAL {slug}] bpb={pre_opd_metrics['val_bpb']} schema={pre_opd_metrics['schema_rate']} "
        f"allowed={pre_opd_metrics['allowed_compliance']} tool_f1={pre_opd_metrics['tool_f1']} "
        f"refusal={pre_opd_metrics['refusal_pass_rate']} chitchat0={pre_opd_metrics['chitchat_zero_call_rate']} "
        f"kill_leak={pre_opd_metrics['kill_list_leak_rate']}",
        flush=True,
    )
    metrics = pre_opd_metrics

    # Optional white-box On-Policy Distillation (OPD) with same-family teacher + regression guard
    if teacher_repo and opd_steps > 0:
        pre_opd_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
        t_src = resolve_local_or_hf(teacher_repo)
        print(f"  [OPD] Loading same-family teacher {teacher_repo} ({t_src}) for {opd_steps} on-policy steps...", flush=True)
        teacher = load_slm(t_src, torch.bfloat16, device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        opd_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
        rng_opd = random.Random(999)
        for st in range(1, opd_steps + 1):
            row = rng_opd.choice(train_rows)
            p_ids = tok(V2_SYSTEM + row["prompt"], return_tensors="pt").input_ids.to(device)
            model.eval()
            with torch.no_grad():
                gen_ids = model.generate(
                    p_ids, max_new_tokens=96, do_sample=True, temperature=0.7,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
                )
            model.train()
            if gen_ids.shape[1] <= p_ids.shape[1] + 2:
                continue
            rstart = p_ids.shape[1] - 1
            s_lg = model(gen_ids).logits[:, rstart:-1, :].float()
            with torch.no_grad():
                t_lg = teacher(gen_ids).logits[:, rstart:-1, :].float()
            if s_lg.shape[-1] != t_lg.shape[-1]:
                print("  [OPD] Skipping due to vocab dim mismatch", flush=True)
                break
            s_lp = F.log_softmax(s_lg, dim=-1)
            t_lp = F.log_softmax(t_lg, dim=-1)
            rev_kl = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1).mean()
            # Strong SFT anchor to preserve exact JSON/tool calibration
            t_ids = tok(row["target"] + (tok.eos_token or ""), add_special_tokens=False).input_ids
            inp = torch.tensor([p_ids[0].tolist() + t_ids], dtype=torch.long, device=device)
            a_lg = model(input_ids=inp).logits[0, p_ids.shape[1] - 1 : p_ids.shape[1] + len(t_ids) - 1, :].float()
            sft_anchor = F.cross_entropy(a_lg, torch.tensor(t_ids, dtype=torch.long, device=device), label_smoothing=0.05)
            loss = 0.25 * rev_kl + 1.0 * sft_anchor
            opd_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opd_opt.step()
            if st % 25 == 0:
                print(f"    [OPD {st}/{opd_steps}] rev_kl={rev_kl.item():.4f} anchor={sft_anchor.item():.4f}", flush=True)
        del teacher
        torch.cuda.empty_cache()

        post_opd_metrics = eval_v2(model, tok, eval_rows, device)
        print(
            f"[POST-OPD EVAL {slug}] bpb={post_opd_metrics['val_bpb']} schema={post_opd_metrics['schema_rate']} "
            f"allowed={post_opd_metrics['allowed_compliance']} tool_f1={post_opd_metrics['tool_f1']} "
            f"refusal={post_opd_metrics['refusal_pass_rate']} chitchat0={post_opd_metrics['chitchat_zero_call_rate']} "
            f"kill_leak={post_opd_metrics['kill_list_leak_rate']}",
            flush=True,
        )
        # Keep OPD weights only if tool_f1 + refusal improved (or stayed equal with better val_bpb)
        pre_score = (pre_opd_metrics["tool_f1"] + (pre_opd_metrics["refusal_pass_rate"] or 0.0), -pre_opd_metrics["val_bpb"])
        post_score = (post_opd_metrics["tool_f1"] + (post_opd_metrics["refusal_pass_rate"] or 0.0), -post_opd_metrics["val_bpb"])
        if post_score >= pre_score:
            print("  [OPD] Kept post-OPD weights (improved/matched eval metrics).", flush=True)
            metrics = post_opd_metrics
            metrics["opd_kept"] = True
        else:
            print("  [OPD] Reverted to pre-OPD best-SFT weights (better eval metrics).", flush=True)
            model.load_state_dict(pre_opd_state, strict=False)
            metrics = pre_opd_metrics
            metrics["opd_kept"] = False
            metrics["post_opd_val_bpb"] = post_opd_metrics["val_bpb"]
            metrics["post_opd_tool_f1"] = post_opd_metrics["tool_f1"]

    print(
        f"[V2 FINAL {slug}] bpb={metrics['val_bpb']} schema={metrics['schema_rate']} "
        f"allowed={metrics['allowed_compliance']} tool_f1={metrics['tool_f1']} "
        f"refusal={metrics['refusal_pass_rate']} chitchat0={metrics['chitchat_zero_call_rate']} "
        f"kill_leak={metrics['kill_list_leak_rate']}",
        flush=True,
    )

    merged_dir = out_dir / "checkpoints" / slug
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)

    gguf_mb = None
    if export_gguf:
        llama_dir = Path("/project/alex_phd/npc-text-gen/llama.cpp")
        gguf_dir = out_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        f16 = gguf_dir / f"{slug}_tools_v2_f16.gguf"
        q4 = gguf_dir / f"{slug}_tools_v2_Q4_K_M.gguf"
        try:
            subprocess.run([sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), str(merged_dir), "--outfile", str(f16), "--outtype", "f16"], check=True)
            subprocess.run([str(llama_dir / "build/bin/llama-quantize"), str(f16), str(q4), "Q4_K_M"], check=True)
            f16.unlink(missing_ok=True)
            gguf_mb = round(q4.stat().st_size / (1024 * 1024), 2)
            print(f"  [GGUF] Exported {q4} ({gguf_mb} MB)", flush=True)
        except Exception as exc:
            print(f"  [GGUF WARN] {exc}", flush=True)

    del model, merged
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "slug": slug,
        "teacher": teacher_repo,
        "train_sec": round(time.time() - t0, 1),
        "gguf_q4_mb": gguf_mb,
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", default="/project/alex_phd/npc-text-gen/shootout_v2/checkpoints")
    ap.add_argument("--data-dir", default="/project/alex_phd/npc-text-gen/data")
    ap.add_argument("--out-dir", default="/project/alex_phd/npc-text-gen/tools_v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--opd-steps", type=int, default=75)
    ap.add_argument("--export-gguf", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="Optional subset of model slugs to run (merges into existing results JSON)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_rows = prepare.load_jsonl(data_dir / "bs_tools_v2_train.jsonl")
    eval_rows = prepare.load_jsonl(data_dir / "bs_tools_v2_eval.jsonl")
    print(f"Loaded Stage 2 v2 tool dataset: train={len(train_rows)}, eval={len(eval_rows)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "tools_v2_results.json"

    # Top 4 Pareto Champions from Shootout v2 (with same-family OPD teachers where available!)
    candidates = [
        ("google__gemma-3-1b-it", "google/gemma-3-12b-it"),
        ("Qwen__Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"),
        ("LiquidAI__LFM2-700M", ""),
        ("LiquidAI__LFM2.5-350M", ""),
    ]
    if args.only:
        only_set = set(args.only)
        candidates = [(s, t) for s, t in candidates if s in only_set]

    existing_by_slug: Dict[str, Any] = {}
    if args.only and report_path.exists():
        for r in json.loads(report_path.read_text(encoding="utf-8")):
            existing_by_slug[r["slug"]] = r

    results = []
    for slug, teacher in candidates:
        ckpt = Path(args.stage1_dir) / slug
        if not ckpt.is_dir():
            print(f"[SKIP] Missing Stage 1 checkpoint {ckpt}", flush=True)
            continue
        try:
            res = train_one(
                slug=slug,
                ckpt_dir=ckpt,
                train_rows=train_rows,
                eval_rows=eval_rows,
                out_dir=out_dir,
                epochs=args.epochs,
                lr=args.lr,
                lora_r=args.lora_r,
                teacher_repo=teacher,
                opd_steps=args.opd_steps if teacher else 0,
                export_gguf=args.export_gguf,
            )
            results.append(res)
            existing_by_slug[slug] = res
        except Exception as exc:
            print(f"[ERROR] {slug}: {type(exc).__name__}: {exc}", flush=True)
            err_res = {"slug": slug, "error": f"{type(exc).__name__}: {exc}"}
            results.append(err_res)
            existing_by_slug[slug] = err_res
        merged_out = list(existing_by_slug.values()) if args.only else results
        report_path.write_text(json.dumps(merged_out, indent=2), encoding="utf-8")

    print("\n================ STAGE 2 TOOL-CALLING SUMMARY ================")
    for r in results:
        if "error" in r:
            print(f"FAIL  {r['slug']:30} {r['error']}")
        else:
            m = r["metrics"]
            print(
                f"OK    {r['slug']:30} bpb={m['val_bpb']:.4f} schema={m['schema_rate']:.2f} "
                f"allow={m['allowed_compliance']:.2f} toolF1={m['tool_f1']:.2f} "
                f"refusal={m['refusal_pass_rate']:.2f} killLeak={m['kill_list_leak_rate']:.2f} "
                f"gguf={r['gguf_q4_mb']}MB"
            )


if __name__ == "__main__":
    main()
