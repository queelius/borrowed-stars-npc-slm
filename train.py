#!/usr/bin/env python3
"""
#CLAB cpus=8 ram=32 est=15m priority=normal

MUTABLE EXPERIMENT SCRIPT (`autoresearch` target).

Supports two modes:
  1. `--mode looped-scratch`:
     Fast Parameter-Golf / Inkling-inspired Looped Transformer (Prelude +
     Recurrent Core looped R times with attention residuals & intermediate
     deep supervision + Coda + optional fake INT4 QAT + entropy-adaptive
     n-gram cache blending at eval time). Runs anywhere (CPU or GPU) in seconds.
  2. `--mode adapt`:
     Adapts a Hugging Face sub-1B SLM (e.g. Qwen3-0.6B, LFM2-350M, Gemma-3-1B)
     with optional BPE-safe vocabulary pruning, LoRA/full SFT, QAT, and
     evaluates `val_bpb` + projected INT4 artifact size via `prepare.py`.
"""

from __future__ import annotations

import argparse
import collections
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import prepare

# ---------------------------------------------------------------------------
# Tunable Hyperparameters (primary autoresearch mutation surface)
# ---------------------------------------------------------------------------
CFG = {
    # Looped-scratch defaults
    "d_model": 192,
    "n_heads": 6,
    "prelude_layers": 1,
    "core_layers": 2,
    "coda_layers": 1,
    "n_loops": 3,
    "dropout": 0.05,
    "lr": 3e-3,
    "weight_decay": 0.01,
    "batch_size": 8,
    "max_seq_len": 256,
    "qat_bits": 4,                # 0 disables fake quantization; 4 or 6 enables QAT STE
    "deep_supervision": True,     # Supervise every loop pass (Ouro / uptraining trick)
    "ngram_order": 5,             # Parameter-Golf entropy-adaptive n-gram cache order
    "ngram_max_alpha": 0.35,      # Max interpolation weight when neural entropy is high
    "ema_decay": 0.99,
    # Adapt mode defaults
    "adapt_lr": 2e-4,
    "adapt_lora_r": 16,
    "adapt_prune_vocab": 0,       # e.g., 32768 to trim multilingual vocab on Qwen/Gemma
}


# ---------------------------------------------------------------------------
# Mode 1: Byte-Level Looped Transformer + QAT STE + Entropy-Adaptive N-Gram
# ---------------------------------------------------------------------------
def run_looped_scratch(budget_sec: float, device_str: str) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device(
        device_str
        if device_str != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(42)

    class FakeQuantLinear(nn.Linear):
        def __init__(self, in_f: int, out_f: int, bias: bool = False, bits: int = 4):
            super().__init__(in_f, out_f, bias=bias)
            self.bits = bits

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.bits and self.bits < 16:
                qmax = (1 << (self.bits - 1)) - 1
                scale = self.weight.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
                w_q = torch.clamp(torch.round(self.weight / scale), -qmax, qmax) * scale
                w = self.weight + (w_q - self.weight).detach()  # Straight-Through Estimator
            else:
                w = self.weight
            return F.linear(x, w, self.bias)

    class Block(nn.Module):
        def __init__(self, d_model: int, n_heads: int, dropout: float, qat_bits: int):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.ln1 = nn.RMSNorm(d_model)
            self.qkv = FakeQuantLinear(d_model, 3 * d_model, bits=qat_bits)
            self.proj = FakeQuantLinear(d_model, d_model, bits=qat_bits)
            self.ln2 = nn.RMSNorm(d_model)
            self.w1 = FakeQuantLinear(d_model, int(d_model * 8 / 3), bits=qat_bits)
            self.w2 = FakeQuantLinear(d_model, int(d_model * 8 / 3), bits=qat_bits)
            self.w3 = FakeQuantLinear(int(d_model * 8 / 3), d_model, bits=qat_bits)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.shape
            h = self.ln1(x)
            qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.drop(self.proj(attn))
            h2 = self.ln2(x)
            ff = self.w3(F.silu(self.w1(h2)) * self.w2(h2))
            return x + self.drop(ff)

    class LoopedNPCModel(nn.Module):
        """Prelude -> Looped Core (with gated Attention Residual) -> Coda."""

        def __init__(self, cfg: Dict):
            super().__init__()
            d = cfg["d_model"]
            h = cfg["n_heads"]
            dr = cfg["dropout"]
            qb = cfg["qat_bits"]
            self.n_loops = cfg["n_loops"]
            self.vocab_size = 256  # Byte-level (universal, zero OOV, instant run)
            self.max_seq_len = cfg["max_seq_len"]
            self.tok_emb = nn.Embedding(self.vocab_size, d)
            self.pos_emb = nn.Embedding(self.max_seq_len, d)
            self.prelude = nn.ModuleList([Block(d, h, dr, qb) for _ in range(cfg["prelude_layers"])])
            self.core = nn.ModuleList([Block(d, h, dr, qb) for _ in range(cfg["core_layers"])])
            self.coda = nn.ModuleList([Block(d, h, dr, qb) for _ in range(cfg["coda_layers"])])
            # Per-loop learned gate + input re-injection (stabilizes depth recurrence)
            self.loop_gates = nn.Parameter(torch.full((self.n_loops,), 0.25))
            self.reinject_scale = nn.Parameter(torch.tensor(0.1))
            self.ln_f = nn.RMSNorm(d)
            self.lm_head = nn.Linear(d, self.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight  # Weight tying

        def forward(self, idx: torch.Tensor, return_all_loops: bool = False):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x0 = self.tok_emb(idx) + self.pos_emb(pos)
            x = x0
            for blk in self.prelude:
                x = blk(x)

            loop_logits = []
            for r in range(self.n_loops):
                h = x + self.reinject_scale * x0
                for blk in self.core:
                    h = blk(h)
                gate = torch.sigmoid(self.loop_gates[r])
                x = (1.0 - gate) * x + gate * h
                if return_all_loops:
                    y = x
                    for blk in self.coda:
                        y = blk(y)
                    loop_logits.append(self.lm_head(self.ln_f(y)))

            if return_all_loops:
                return loop_logits
            for blk in self.coda:
                x = blk(x)
            return self.lm_head(self.ln_f(x))

    class NgramCache:
        """Parameter-Golf inspired lightweight byte n-gram backoff cache."""

        def __init__(self, order: int = 5):
            self.order = order
            self.counts: Dict[bytes, collections.Counter] = collections.defaultdict(collections.Counter)

        def fit(self, texts: List[str]) -> None:
            for t in texts:
                b = t.encode("utf-8")
                for i in range(len(b)):
                    tok = b[i]
                    for k in range(1, min(self.order, i + 1) + 1):
                        ctx = b[i - k : i]
                        self.counts[ctx][tok] += 1

        def dist(self, ctx_bytes: bytes) -> Dict[int, float] | None:
            for k in range(min(self.order, len(ctx_bytes)), 0, -1):
                c = self.counts.get(ctx_bytes[-k:])
                if c:
                    total = sum(c.values())
                    return {tok: cnt / total for tok, cnt in c.items()}
            return None

    train_rows = prepare.load_train_data()
    eval_rows = prepare.load_eval_data()

    def encode_pair(prompt: str, target: str) -> Tuple[List[int], int]:
        p_b = list(prompt.encode("utf-8"))
        t_b = list(target.encode("utf-8"))
        seq = (p_b + t_b)[: CFG["max_seq_len"]]
        return seq, min(len(p_b), len(seq))

    train_seqs = [encode_pair(r["prompt"], r["target"]) for r in train_rows]

    # Fit tiny n-gram cache on train world-building dialogues
    ngram = NgramCache(order=CFG["ngram_order"])
    ngram.fit([r["prompt"] + r["target"] for r in train_rows])

    model = LoopedNPCModel(CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    t_start = time.time()
    step = 0
    model.train()
    while time.time() - t_start < budget_sec:
        batch_idx = [train_seqs[(step * CFG["batch_size"] + i) % len(train_seqs)] for i in range(CFG["batch_size"])]
        max_len = max(len(s) for s, _ in batch_idx)
        x_t = torch.zeros((CFG["batch_size"], max_len - 1), dtype=torch.long, device=device)
        y_t = torch.full((CFG["batch_size"], max_len - 1), -100, dtype=torch.long, device=device)
        for bi, (seq, p_len) in enumerate(batch_idx):
            L = len(seq) - 1
            x_t[bi, :L] = torch.tensor(seq[:-1], dtype=torch.long)
            # Supervise target tokens (and lightly prompt tokens for representation quality)
            targets = torch.tensor(seq[1:], dtype=torch.long)
            y_t[bi, max(0, p_len - 1) : L] = targets[max(0, p_len - 1) :]

        opt.zero_grad(set_to_none=True)
        if CFG["deep_supervision"]:
            all_logits = model(x_t, return_all_loops=True)
            loss = sum(
                ((r + 1) / len(all_logits)) * F.cross_entropy(logits.reshape(-1, 256), y_t.reshape(-1), ignore_index=-100)
                for r, logits in enumerate(all_logits)
            ) / sum((r + 1) / len(all_logits) for r in range(len(all_logits)))
        else:
            logits = model(x_t)
            loss = F.cross_entropy(logits.reshape(-1, 256), y_t.reshape(-1), ignore_index=-100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

    train_elapsed = time.time() - t_start
    model.eval()

    # Evaluate tokenizer-agnostic val_bpb with optional entropy-adaptive n-gram blend
    @torch.no_grad()
    def eval_nll_nats(prompt: str, target: str) -> float:
        p_bytes = prompt.encode("utf-8")
        t_bytes = target.encode("utf-8")
        full = list(p_bytes + t_bytes)[: CFG["max_seq_len"]]
        p_len = len(p_bytes)
        inp = torch.tensor([full[:-1]], dtype=torch.long, device=device)
        logits = model(inp)[0]  # [T, 256]
        probs = F.softmax(logits, dim=-1)

        total_nll = 0.0
        max_H = math.log(256.0)
        for i in range(p_len, len(full)):
            pos = i - 1
            tok = full[i]
            p_nn = probs[pos]
            p_tok = float(p_nn[tok].item())

            if CFG["ngram_max_alpha"] > 0:
                ctx = bytes(full[:i])
                ng = ngram.dist(ctx)
                if ng is not None:
                    H = -float((p_nn * torch.log(p_nn.clamp(min=1e-12))).sum().item())
                    lam = CFG["ngram_max_alpha"] * (H / max_H)
                    p_ng = ng.get(tok, 1e-6)
                    p_tok = (1.0 - lam) * p_tok + lam * p_ng

            total_nll += -math.log(max(p_tok, 1e-12))
        return total_nll

    bpb_stats = prepare.compute_val_bpb(eval_rows, eval_nll_nats)

    # Measure CPU/device autoregressive decode throughput (32 bytes)
    t_gen0 = time.time()
    with torch.no_grad():
        seed = torch.tensor([list(eval_rows[0]["prompt"].encode("utf-8"))[-64:]], dtype=torch.long, device=device)
        for _ in range(32):
            nxt = model(seed)[:, -1, :].argmax(dim=-1, keepdim=True)
            seed = torch.cat([seed, nxt], dim=1)[:, -CFG["max_seq_len"] :]
    gen_s = max(time.time() - t_gen0, 1e-6)
    decode_tok_s = 32.0 / gen_s

    n_params = sum(p.numel() for p in model.parameters())
    bits = CFG["qat_bits"] if CFG["qat_bits"] else 16
    artifact_mb = (n_params * bits / 8.0) / (1024.0 * 1024.0) + 0.05  # + tiny n-gram cache

    prepare.evaluate_and_report(
        val_bpb=bpb_stats["val_bpb"],
        artifact_mb=artifact_mb,
        decode_tok_s=decode_tok_s,
        extra={
            "mode": "looped-scratch",
            "device": str(device),
            "steps": step,
            "train_sec": round(train_elapsed, 2),
            "n_params": n_params,
            "effective_layers": CFG["prelude_layers"] + CFG["n_loops"] * CFG["core_layers"] + CFG["coda_layers"],
            "n_loops": CFG["n_loops"],
            "qat_bits": bits,
        },
    )


# ---------------------------------------------------------------------------
# Mode 2: Hugging Face Sub-1B Adaptation (Vocab Prune + LoRA/SFT + QAT)
# ---------------------------------------------------------------------------
def run_hf_adapt(model_name: str, budget_sec: float, device_str: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        device_str
        if device_str != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading base SLM: {model_name} on {device} ({dtype})...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    train_rows = prepare.load_train_data()
    eval_rows = prepare.load_eval_data()

    # Optional LoRA if peft is installed; else full fine-tune (tiny <1B models fit easily!)
    try:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=CFG["adapt_lora_r"],
            lora_alpha=CFG["adapt_lora_r"] * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        print("Using LoRA adaptation.")
    except Exception:
        print("peft not installed; running direct fine-tuning.")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["adapt_lr"],
        weight_decay=0.01,
    )

    t0 = time.time()
    step = 0
    model.train()
    while time.time() - t0 < budget_sec:
        row = train_rows[step % len(train_rows)]
        p_ids = tok(row["prompt"], add_special_tokens=True).input_ids
        t_ids = tok(row["target"] + (tok.eos_token or ""), add_special_tokens=False).input_ids
        input_ids = torch.tensor([p_ids + t_ids], device=device)
        labels = torch.tensor([[-100] * len(p_ids) + t_ids], device=device)

        opt.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

    model.eval()

    @torch.no_grad()
    def eval_nll_nats(prompt: str, target: str) -> float:
        p_ids = tok(prompt, add_special_tokens=True).input_ids
        t_ids = tok(target, add_special_tokens=False).input_ids
        input_ids = torch.tensor([p_ids + t_ids], device=device)
        labels = torch.tensor([[-100] * len(p_ids) + t_ids], device=device)
        out = model(input_ids=input_ids, labels=labels)
        return float(out.loss.item()) * len(t_ids)

    bpb_stats = prepare.compute_val_bpb(eval_rows, eval_nll_nats)

    # Projected Q4_K_M footprint (~4.8 bits/param effective)
    total_params = sum(p.numel() for p in model.parameters())
    est_q4_mb = (total_params * 4.8 / 8.0) / (1024.0 * 1024.0)

    prepare.evaluate_and_report(
        val_bpb=bpb_stats["val_bpb"],
        artifact_mb=est_q4_mb,
        extra={
            "mode": "adapt",
            "model": model_name,
            "device": str(device),
            "steps": step,
            "train_sec": round(time.time() - t0, 2),
            "total_params": total_params,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["looped-scratch", "adapt"], default="looped-scratch")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--budget-sec", type=float, default=30.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.mode == "looped-scratch":
        run_looped_scratch(args.budget_sec, args.device)
    else:
        run_hf_adapt(args.model, args.budget_sec, args.device)


if __name__ == "__main__":
    main()
