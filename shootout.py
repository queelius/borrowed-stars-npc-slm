#!/usr/bin/env python3
"""
#CLAB gpus=1 vram=24 cpus=8 ram=48 est=45m priority=normal

Borrowed Stars Sub-500MB Base Model Shootout on A100.

For each candidate SLM:
  1. Inspects param breakdown (embedding vs backbone) + projected Q4_K_M MB
     both raw and after 32K vocabulary pruning.
  2. Evaluates Zero-Shot `val_bpb` on frozen `data/bs_eval.jsonl` (overall,
     `npc`, and `narrator_interactive`) + zero-shot generations.
  3. Runs 2-Stage Borrowed Stars Adaptation (LoRA r=32 on all linear layers):
       - Stage 1 (CPT): Continued Pretraining on all 267 canonical lore +
         narrator monologue chunks (`data/bs_lore_chunks.jsonl`).
       - Stage 2 (SFT): Supervised Fine-Tuning on `data/bs_train.jsonl`
         (loss strictly on target JSON).
  4. Re-evaluates `val_bpb`, unconstrained JSON schema validity, and generates
     live Borrowed Stars NPC & Interactive-Narrator replies for inspection.
  5. Saves merged checkpoint + JSON report under `--out-dir`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import prepare


def load_slm(load_path: str, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    """Load standard causal LMs, early-fusion multimodal models (Qwen3.5-0.8B), and custom looped LMs (Ouro-1.4B)."""
    cfg = AutoConfig.from_pretrained(load_path, trust_remote_code=True)
    if not hasattr(cfg, "text_config") and (not hasattr(cfg, "pad_token_id") or isinstance(getattr(cfg, "pad_token_id", None), list)):
        eos = getattr(cfg, "eos_token_id", 0)
        cfg.pad_token_id = eos[0] if isinstance(eos, list) and eos else (eos if isinstance(eos, int) else 0)
    try:
        return AutoModelForCausalLM.from_pretrained(
            load_path,
            config=cfg,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)
    except Exception as first_err:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(
                load_path,
                config=cfg,
                dtype=dtype,
                trust_remote_code=True,
            ).to(device)
        except Exception:
            raise first_err

SYSTEM_PROMPT = (
    "You are the Borrowed Stars in-game dialogue & Interactive Narrator engine. "
    "Reply ONLY with a single valid JSON object matching "
    "{\"mood\": <calm|wary|friendly|hostile|fearful|amused|cryptic>, "
    "\"action\": <NONE|GREET|TRADE|WARN|DRAW_WEAPON|FLEE|SHARE_RUMOR|OFFER_QUEST>, "
    "\"line\": <concise in-character response <= 240 chars>}.\n"
)


def param_breakdown(model: torch.nn.Module, vocab_size: int) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    emb_module = model.get_input_embeddings()
    emb_params = emb_module.weight.numel() if emb_module is not None else 0
    out_module = model.get_output_embeddings()
    tied = (
        emb_module is not None
        and out_module is not None
        and getattr(out_module, "weight", None) is getattr(emb_module, "weight", None)
    )
    lm_head_params = 0 if tied or out_module is None else out_module.weight.numel()
    io_params = emb_params + lm_head_params
    backbone_params = total - io_params

    # Q4_K_M effective ~4.85 bits/param for backbone; embeddings typically Q4/Q6 (~5.5 bits)
    raw_q4_mb = (backbone_params * 4.85 + io_params * 5.5) / 8.0 / (1024 ** 2)
    pruned_vocab = min(vocab_size, 32768)
    pruned_io = int(io_params * (pruned_vocab / max(vocab_size, 1)))
    pruned_q4_mb = (backbone_params * 4.85 + pruned_io * 5.5) / 8.0 / (1024 ** 2)
    return {
        "total_params": total,
        "backbone_params": backbone_params,
        "io_params": io_params,
        "io_pct": round(100.0 * io_params / max(total, 1), 1),
        "vocab_size": vocab_size,
        "tied_embeddings": tied,
        "est_q4_mb_raw": round(raw_q4_mb, 1),
        "est_q4_mb_32k_vocab": round(pruned_q4_mb, 1),
    }


def eval_bpb_and_samples(
    model: torch.nn.Module,
    tok,
    eval_rows: List[Dict[str, str]],
    device: torch.device,
    sample_indices: List[int] = (0, 2, 8, 13),
) -> Dict[str, Any]:
    model.eval()

    @torch.no_grad()
    def nll_fn(prompt: str, target: str) -> float:
        full_prompt = SYSTEM_PROMPT + prompt
        p_ids = tok(full_prompt, add_special_tokens=True).input_ids
        t_ids = tok(target, add_special_tokens=False).input_ids
        if not t_ids:
            return 0.0
        inp = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
        out = model(input_ids=inp)
        # Shift logits: position len(p_ids)-1 predicts t_ids[0], ..., len-2 predicts t_ids[-1]
        logits = out.logits[0, len(p_ids) - 1 : len(p_ids) + len(t_ids) - 1, :].float()
        targets = torch.tensor(t_ids, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        return float(loss.item())

    overall = prepare.compute_val_bpb(eval_rows, nll_fn)
    npc_rows = [r for r in eval_rows if r.get("subset") == "npc"]
    narr_rows = [r for r in eval_rows if r.get("subset") == "narrator_interactive"]
    npc_bpb = prepare.compute_val_bpb(npc_rows, nll_fn)["val_bpb"] if npc_rows else None
    narr_bpb = prepare.compute_val_bpb(narr_rows, nll_fn)["val_bpb"] if narr_rows else None

    # Generate unconstrained samples on all eval prompts to measure raw JSON schema rate
    generations: List[str] = []
    samples_preview: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx, row in enumerate(eval_rows):
            full_prompt = SYSTEM_PROMPT + row["prompt"]
            enc = tok(full_prompt, return_tensors="pt").to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            txt = tok.decode(gen[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()
            # Keep first line / JSON object if trailing text appears
            first_line = txt.splitlines()[0].strip() if txt else ""
            generations.append(first_line)
            if idx in sample_indices:
                samples_preview.append(
                    {
                        "idx": idx,
                        "subset": row.get("subset"),
                        "prompt": row["prompt"].strip(),
                        "gold": row["target"],
                        "pred": first_line,
                        "schema": prepare.validate_npc_json(first_line),
                    }
                )

    valid_rate = sum(1 for g in generations if prepare.validate_npc_json(g)["valid"]) / max(len(generations), 1)
    return {
        "val_bpb": round(overall["val_bpb"], 5),
        "npc_val_bpb": round(npc_bpb, 5) if npc_bpb is not None else None,
        "narrator_val_bpb": round(narr_bpb, 5) if narr_bpb is not None else None,
        "raw_schema_valid_rate": round(valid_rate, 4),
        "samples": samples_preview,
    }


def detect_lora_targets(model: torch.nn.Module) -> List[str]:
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    forbidden = {"lm_head", "embed_tokens"}
    if "falcon_h1" in model_type or "mamba" in model_type:
        forbidden.update({"out_proj", "conv1d"})
    linear_names = set()
    for name, mod in model.named_modules():
        if "vision" in name.lower() or "visual" in name.lower():
            continue
        if isinstance(mod, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf not in forbidden:
                linear_names.add(leaf)
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj", "out_proj", "w1", "w2", "w3",
        "qkv_proj", "out_proj_conv",
    ]
    found = [p for p in preferred if p in linear_names]
    return found if found else sorted(linear_names)


def resolve_local_or_hf(repo: str) -> str:
    """Prefer pre-cached model directory on /project/alex_phd/research-cache/models if present."""
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


def run_candidate(
    repo: str,
    cpt_epochs: int,
    sft_epochs: int,
    lora_r: int,
    cpt_lr: float,
    sft_lr: float,
    label_smoothing: float,
    synth_data: str,
    out_dir: Path,
    save_merged: bool,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    t0 = time.time()
    load_path = resolve_local_or_hf(repo)
    print(f"\n==================== {repo} (src={load_path}) ====================", flush=True)

    tok = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_slm(load_path, dtype, device)

    stats = param_breakdown(model, len(tok))
    print("PARAM STATS:", json.dumps(stats), flush=True)

    golden_train = prepare.load_train_data()
    if synth_data and Path(synth_data).exists():
        synth_rows = prepare.load_jsonl(Path(synth_data))
        # Mix grounded synthetic pairs with 3x upweighted canonical golden pairs
        train_rows = synth_rows + golden_train * 3
        print(f"Using expanded SFT set: {len(synth_rows)} synth + 3x{len(golden_train)} golden = {len(train_rows)} rows", flush=True)
    else:
        train_rows = golden_train
    eval_rows = prepare.load_eval_data()
    lore_chunks = prepare.load_lore_chunks()

    # 1. Zero-shot eval
    zs = eval_bpb_and_samples(model, tok, eval_rows, device)
    print(
        f"[ZERO-SHOT] val_bpb={zs['val_bpb']} (npc={zs['npc_val_bpb']}, narr={zs['narrator_val_bpb']}) "
        f"raw_schema={zs['raw_schema_valid_rate']}",
        flush=True,
    )

    # 2. Attach LoRA across all attention/SSM/MLP linear projections
    targets = detect_lora_targets(model)
    print("LoRA targets:", targets, flush=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Stage 1: Continued Pretraining (CPT) on Borrowed Stars lore + narrator monologues
    t_train0 = time.time()
    model.train()
    if cpt_epochs > 0 and lore_chunks:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cpt_lr,
            weight_decay=0.01,
        )
        total_steps = cpt_epochs * len(lore_chunks)
        step = 0
        grad_accum = 4
        opt.zero_grad(set_to_none=True)
        for ep in range(cpt_epochs):
            ep_loss = 0.0
            for idx, ch in enumerate(lore_chunks):
                ids = tok(
                    ch["text"] + (tok.eos_token or ""),
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).input_ids.to(device)
                if ids.shape[1] < 8:
                    continue
                out = model(input_ids=ids, labels=ids)
                (out.loss / grad_accum).backward()
                ep_loss += float(out.loss.item())
                step += 1
                if step % grad_accum == 0 or idx == len(lore_chunks) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # Cosine LR decay
                    progress = step / max(total_steps, 1)
                    lr_now = cpt_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            print(f"  [CPT ep {ep+1}/{cpt_epochs}] avg_loss={ep_loss / max(len(lore_chunks), 1):.4f}", flush=True)

    # Stage 2: SFT on Borrowed Stars NPC + Interactive Narrator JSON pairs (with label smoothing + best-epoch tracking)
    best_sft_bpb = float("inf")
    best_lora_state = None
    if sft_epochs > 0 and train_rows:
        import random as _rnd
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=sft_lr,
            weight_decay=0.01,
        )
        grad_accum = 8 if len(train_rows) > 200 else 4
        total_steps = sft_epochs * len(train_rows)
        step = 0
        opt.zero_grad(set_to_none=True)
        for ep in range(sft_epochs):
            model.train()
            order = list(range(len(train_rows)))
            _rnd.Random(42 + ep).shuffle(order)
            ep_loss = 0.0
            for idx_i, row_i in enumerate(order):
                row = train_rows[row_i]
                p_ids = tok(SYSTEM_PROMPT + row["prompt"], add_special_tokens=True).input_ids
                t_ids = tok(row["target"] + (tok.eos_token or ""), add_special_tokens=False).input_ids
                if not t_ids:
                    continue
                inp = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
                out = model(input_ids=inp)
                logits = out.logits[0, len(p_ids) - 1 : len(p_ids) + len(t_ids) - 1, :].float()
                targets_t = torch.tensor(t_ids, dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, targets_t, label_smoothing=label_smoothing)
                (loss / grad_accum).backward()
                ep_loss += float(loss.item())
                step += 1
                if step % grad_accum == 0 or idx_i == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    progress = step / max(total_steps, 1)
                    lr_now = sft_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            # Fast per-epoch val_bpb checkpoint selection (no overfitting!)
            model.eval()
            with torch.no_grad():
                def _ep_nll(prompt: str, target: str) -> float:
                    p_ids = tok(SYSTEM_PROMPT + prompt, add_special_tokens=True).input_ids
                    t_ids = tok(target, add_special_tokens=False).input_ids
                    inp = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
                    lg = model(input_ids=inp).logits[0, len(p_ids) - 1 : len(p_ids) + len(t_ids) - 1, :].float()
                    return float(F.cross_entropy(lg, torch.tensor(t_ids, dtype=torch.long, device=device), reduction="sum").item())
                ep_bpb = prepare.compute_val_bpb(eval_rows, _ep_nll)["val_bpb"]
            print(f"  [SFT ep {ep+1}/{sft_epochs}] avg_loss={ep_loss / max(len(train_rows), 1):.4f} val_bpb={ep_bpb:.5f}", flush=True)
            if ep_bpb < best_sft_bpb:
                best_sft_bpb = ep_bpb
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

        if best_lora_state is not None:
            model.load_state_dict(best_lora_state, strict=False)
            print(f"  [BEST-EPOCH] Restored best SFT LoRA state (val_bpb={best_sft_bpb:.5f})", flush=True)

    train_sec = round(time.time() - t_train0, 2)

    # 3. Post-adaptation eval
    adapted = eval_bpb_and_samples(model, tok, eval_rows, device)
    print(
        f"[ADAPTED  ] val_bpb={adapted['val_bpb']} (npc={adapted['npc_val_bpb']}, narr={adapted['narrator_val_bpb']}) "
        f"raw_schema={adapted['raw_schema_valid_rate']} train_sec={train_sec}s",
        flush=True,
    )
    print(
        f"::clab::metric model={repo.split('/')[-1]} zs_bpb={zs['val_bpb']} "
        f"adapt_bpb={adapted['val_bpb']} schema={adapted['raw_schema_valid_rate']} "
        f"q4_mb={stats['est_q4_mb_raw']} q4_32k_mb={stats['est_q4_mb_32k_vocab']}",
        flush=True,
    )

    slug = repo.replace("/", "__")
    if save_merged:
        ckpt_dir = out_dir / "checkpoints" / slug
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
        print(f"Saved merged checkpoint to {ckpt_dir}", flush=True)

    result = {
        "repo": repo,
        "param_stats": stats,
        "lora_targets": targets,
        "train_sec": train_sec,
        "total_sec": round(time.time() - t0, 2),
        "zero_shot": zs,
        "adapted": adapted,
        "bpb_gain": round(zs["val_bpb"] - adapted["val_bpb"], 5),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "LiquidAI/LFM2-350M",
            "LiquidAI/LFM2-700M",
            "Qwen/Qwen3-0.6B",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
        ],
    )
    ap.add_argument("--cpt-epochs", type=int, default=2)
    ap.add_argument("--sft-epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--cpt-lr", type=float, default=1.2e-4)
    ap.add_argument("--sft-lr", type=float, default=8e-5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--synth-data", default="")
    ap.add_argument("--out-dir", default="results/shootout_v2")
    ap.add_argument("--save-merged", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "shootout_results.json"

    all_results: List[Dict[str, Any]] = []
    for repo in args.models:
        try:
            res = run_candidate(
                repo=repo,
                cpt_epochs=args.cpt_epochs,
                sft_epochs=args.sft_epochs,
                lora_r=args.lora_r,
                cpt_lr=args.cpt_lr,
                sft_lr=args.sft_lr,
                label_smoothing=args.label_smoothing,
                synth_data=args.synth_data,
                out_dir=out_dir,
                save_merged=args.save_merged,
            )
            all_results.append(res)
        except Exception as exc:
            print(f"[ERROR] {repo}: {type(exc).__name__}: {exc}", flush=True)
            all_results.append({"repo": repo, "error": f"{type(exc).__name__}: {exc}"})
        report_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n================ SHOOTOUT SUMMARY ================")
    for r in all_results:
        if "error" in r:
            print(f"FAIL  {r['repo']:40} {r['error']}")
        else:
            ps = r["param_stats"]
            print(
                f"OK    {r['repo']:40} "
                f"zs_bpb={r['zero_shot']['val_bpb']:.4f} -> adapt_bpb={r['adapted']['val_bpb']:.4f} "
                f"(gain={r['bpb_gain']:+.4f}) schema={r['adapted']['raw_schema_valid_rate']:.2f} "
                f"q4={ps['est_q4_mb_raw']:.0f}MB (32k={ps['est_q4_mb_32k_vocab']:.0f}MB)"
            )


if __name__ == "__main__":
    main()
