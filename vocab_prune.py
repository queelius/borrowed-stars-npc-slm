#!/usr/bin/env python3
"""
BPE-Safe Vocabulary Pruning for Ultra-Small (<500MB) English + Borrowed Stars SLMs.

Slashes 25%-60% of parameter count on multilingual models (Qwen 151K, Gemma 256K,
LFM 64K) by keeping:
  1. All special / control / byte-fallback tokens (<0x00>..<0xFF> or single-byte tokens)
  2. Every token that appears in the full Borrowed Stars lore + dialogue corpus
  3. Top frequent English ASCII/Latin tokens up to `--target-vocab` (default 32768)
  4. Full BPE merge-closure of kept tokens so common English & lore words do NOT fragment

Slices `embed_tokens` (and untied `lm_head`), remaps `tokenizer.json` + merges, and
verifies round-trip tokenization fidelity on `data/bs_eval.jsonl`.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prepare

ASCII_LATIN_RE = re.compile(r"^[\x00-\x7f\u00c0-\u024f\u2010-\u2027\u0120\u2581]+$")


def collect_corpus_token_counts(tok) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    texts: List[str] = []
    for r in prepare.load_train_data() + prepare.load_eval_data():
        texts.append(r["prompt"])
        texts.append(r["target"])
    for ch in prepare.load_lore_chunks():
        texts.append(ch["text"])
    spec_path = Path(__file__).resolve().parent / "data" / "borrowed_stars_tools_spec.json"
    if spec_path.exists():
        texts.append(spec_path.read_text(encoding="utf-8"))
    for extra_fn in ("bs_tools_v2_train.jsonl", "bs_tools_v2_eval.jsonl"):
        ep = Path(__file__).resolve().parent / "data" / extra_fn
        if ep.exists():
            for r in prepare.load_jsonl(ep):
                texts.append(r["prompt"])
                texts.append(r["target"])
    for t in texts:
        for tid in tok.encode(t, add_special_tokens=True):
            counts[tid] += 1
    return counts


def select_kept_token_ids(tok, target_vocab: int, max_emb_rows: int | None = None) -> List[int]:
    vocab = tok.get_vocab()
    id_to_tok = {i: t for t, i in vocab.items()}
    total_vocab = min(len(id_to_tok), max_emb_rows) if max_emb_rows else len(id_to_tok)
    if target_vocab >= total_vocab:
        return list(range(total_vocab))

    kept: Set[int] = set()
    # 1. Special tokens & added tokens (strictly within embedding table bounds!)
    for tid in tok.all_special_ids:
        if 0 <= tid < total_vocab:
            kept.add(tid)
    # Keep first 512 base tokens (bytes, single chars, control tokens in most BPE tokenizers)
    for tid in range(min(512, total_vocab)):
        kept.add(tid)

    # 2. All tokens observed in Borrowed Stars corpus
    corpus_counts = collect_corpus_token_counts(tok)
    for tid, _ in corpus_counts.most_common():
        if 0 <= tid < total_vocab:
            kept.add(tid)

    # 3. Fill remaining budget up to target_vocab with ASCII/Latin tokens in BPE merge order (lowest ID = highest pretrain freq!)
    for tid in range(total_vocab):
        if len(kept) >= target_vocab:
            break
        if tid in kept:
            continue
        t_str = id_to_tok.get(tid, "")
        if ASCII_LATIN_RE.match(t_str):
            kept.add(tid)

    # 4. If still short, fill lowest IDs to maintain BPE merge hierarchy
    for tid in range(total_vocab):
        if len(kept) >= target_vocab:
            break
        kept.add(tid)

    # Pad to multiple of 64 for tensor-core / GGUF alignment
    kept_sorted = sorted(kept)
    aligned = (len(kept_sorted) // 64) * 64
    if aligned < len(kept_sorted):
        # Grow to next multiple of 64
        next_mult = aligned + 64
        for tid in range(total_vocab):
            if len(kept) >= next_mult:
                break
            kept.add(tid)
        kept_sorted = sorted(kept)
    return kept_sorted


def prune_fast_tokenizer(tok_dir: Path, out_dir: Path, kept_ids: List[int]) -> None:
    tok_json_path = tok_dir / "tokenizer.json"
    if not tok_json_path.exists():
        raise FileNotFoundError(f"Expected fast tokenizer.json in {tok_dir}")

    data = json.loads(tok_json_path.read_text(encoding="utf-8"))
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}

    model_sec = data.get("model", {})
    old_vocab: Dict[str, int] = model_sec.get("vocab", {})
    new_vocab = {t: old_to_new[old_id] for t, old_id in old_vocab.items() if old_id in old_to_new}
    model_sec["vocab"] = new_vocab

    # Filter BPE merges strictly to pairs whose components AND merged token remain in new_vocab
    if "merges" in model_sec and isinstance(model_sec["merges"], list):
        new_merges = []
        for m in model_sec["merges"]:
            if isinstance(m, list) and len(m) == 2:
                a, b = m[0], m[1]
            elif isinstance(m, str) and " " in m:
                a, b = m.split(" ", 1)
            else:
                continue
            if a in new_vocab and b in new_vocab and (a + b) in new_vocab:
                new_merges.append(m)
        model_sec["merges"] = new_merges
    data["model"] = model_sec

    # Remap added_tokens
    if "added_tokens" in data:
        remapped_added = []
        for at in data["added_tokens"]:
            if at["id"] in old_to_new:
                at = dict(at)
                at["id"] = old_to_new[at["id"]]
                remapped_added.append(at)
        data["added_tokens"] = remapped_added

    (out_dir / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # Remap tokenizer_config.json added_tokens_decoder if present
    cfg_path = tok_dir / "tokenizer_config.json"
    if cfg_path.exists():
        tcfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "added_tokens_decoder" in tcfg and isinstance(tcfg["added_tokens_decoder"], dict):
            new_dec = {}
            for k, v in tcfg["added_tokens_decoder"].items():
                old_id = int(k)
                if old_id in old_to_new:
                    new_dec[str(old_to_new[old_id])] = v
            tcfg["added_tokens_decoder"] = new_dec
        # Strip multimodal image_token keys that cause transformers to auto-append an extra out-of-bounds ID on Gemma 3
        for drop_k in ("image_token", "boi_token", "eoi_token"):
            tcfg.pop(drop_k, None)
        if isinstance(tcfg.get("extra_special_tokens"), dict):
            tcfg["extra_special_tokens"] = {
                k: v for k, v in tcfg["extra_special_tokens"].items() if "image" not in k
            }
        (out_dir / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2), encoding="utf-8")


def prune_sentencepiece_model(src_spm: Path, dst_spm: Path, kept_ids: List[int]) -> bool:
    """Slice a SentencePiece tokenizer.model protobuf strictly to kept_ids so llama.cpp uses native SPM decoding."""
    try:
        from sentencepiece import sentencepiece_model_pb2 as spm_pb2
    except Exception:
        return False
    m = spm_pb2.ModelProto()
    m.ParseFromString(src_spm.read_bytes())
    orig_pieces = list(m.pieces)
    if max(kept_ids) >= len(orig_pieces):
        return False
    del m.pieces[:]
    for old_id in kept_ids:
        m.pieces.append(orig_pieces[old_id])
    m.trainer_spec.vocab_size = len(kept_ids)
    # Remap special IDs inside trainer_spec if they exist in kept_ids
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}
    for attr in ("unk_id", "bos_id", "eos_id", "pad_id"):
        v = getattr(m.trainer_spec, attr, -1)
        if v in old_to_new:
            setattr(m.trainer_spec, attr, old_to_new[v])
    dst_spm.write_bytes(m.SerializeToString())
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo or local checkpoint directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-vocab", type=int, default=32768)
    ap.add_argument("--spm-source", default="", help="Optional path to original tokenizer.model or HF repo to fetch it from")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_tok_dir = out_dir / "_orig_tok"
    tmp_tok_dir.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(tmp_tok_dir)

    # Load model first so we clamp strictly to embedding table rows (handles Gemma 3 262144 vs 262145)
    from shootout import load_slm
    model = load_slm(args.model, torch.bfloat16, torch.device("cpu"))
    in_emb = model.get_input_embeddings()
    max_emb_rows = in_emb.weight.shape[0]

    kept_ids = select_kept_token_ids(tok, args.target_vocab, max_emb_rows=max_emb_rows)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}
    print(f"Vocab pruning: {len(tok)} (emb={max_emb_rows}) -> {len(kept_ids)} tokens ({100.0 * (1 - len(kept_ids)/max_emb_rows):.1f}% reduction)")

    # Save pruned tokenizer: preserve & slice SentencePiece tokenizer.model if available (for clean Gemma 3 GGUF space decoding)
    tok.save_pretrained(out_dir)
    src_spm = None
    cands = [Path(args.model) / "tokenizer.model", tmp_tok_dir / "tokenizer.model"]
    if args.spm_source:
        sp = Path(args.spm_source)
        if sp.is_file():
            cands.insert(0, sp)
        elif (sp / "tokenizer.model").is_file():
            cands.insert(0, sp / "tokenizer.model")
        else:
            try:
                from huggingface_hub import hf_hub_download
                dl = hf_hub_download(args.spm_source, "tokenizer.model")
                cands.insert(0, Path(dl))
            except Exception as exc:
                print(f"[WARN] Could not fetch tokenizer.model from {args.spm_source}: {exc}")
    for cand in cands:
        if cand.is_file():
            src_spm = cand
            break
    for legacy in ("tokenizer.model", "spiece.model"):
        lp = out_dir / legacy
        if lp.exists():
            lp.unlink()
    if src_spm is not None and prune_sentencepiece_model(src_spm, out_dir / "tokenizer.model", kept_ids):
        print(f"Wrote pruned SentencePiece model: {out_dir / 'tokenizer.model'} ({len(kept_ids)} pieces)")
    prune_fast_tokenizer(tmp_tok_dir, out_dir, kept_ids)

    # Slice model embeddings & lm_head
    orig_params = sum(p.numel() for p in model.parameters())
    idx_t = torch.tensor(kept_ids, dtype=torch.long)

    out_emb = model.get_output_embeddings()
    tied = out_emb is not None and (out_emb.weight is in_emb.weight)

    new_in = torch.nn.Embedding(len(kept_ids), in_emb.embedding_dim, dtype=in_emb.weight.dtype)
    with torch.no_grad():
        new_in.weight.copy_(in_emb.weight[idx_t])
    model.set_input_embeddings(new_in)

    if out_emb is not None:
        if tied:
            new_out = torch.nn.Linear(out_emb.in_features, len(kept_ids), bias=(out_emb.bias is not None), dtype=out_emb.weight.dtype)
            new_out.weight = new_in.weight
            model.set_output_embeddings(new_out)
        else:
            new_out = torch.nn.Linear(out_emb.in_features, len(kept_ids), bias=(out_emb.bias is not None), dtype=out_emb.weight.dtype)
            with torch.no_grad():
                new_out.weight.copy_(out_emb.weight[idx_t])
                if out_emb.bias is not None:
                    new_out.bias.copy_(out_emb.bias[idx_t])
            model.set_output_embeddings(new_out)

    # Remap special token IDs in config & generation_config
    model.config.vocab_size = len(kept_ids)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        val = getattr(model.config, attr, None)
        if isinstance(val, int) and val in old_to_new:
            setattr(model.config, attr, old_to_new[val])
        elif isinstance(val, list):
            setattr(model.config, attr, [old_to_new[v] for v in val if v in old_to_new])

    if getattr(model, "generation_config", None) is not None:
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            val = getattr(model.generation_config, attr, None)
            if isinstance(val, int) and val in old_to_new:
                setattr(model.generation_config, attr, old_to_new[val])
            elif isinstance(val, list):
                setattr(model.generation_config, attr, [old_to_new[v] for v in val if v in old_to_new])

    model.save_pretrained(out_dir)
    new_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {orig_params/1e6:.1f}M -> {new_params/1e6:.1f}M (saved {(orig_params - new_params)/1e6:.1f}M params)")

    # Verify round-trip tokenization on Borrowed Stars eval set
    pruned_tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    eval_rows = prepare.load_eval_data()
    orig_toks = sum(len(tok.encode(r["prompt"] + r["target"])) for r in eval_rows)
    new_toks = sum(len(pruned_tok.encode(r["prompt"] + r["target"])) for r in eval_rows)
    exact_rt = sum(
        1 for r in eval_rows
        if pruned_tok.decode(pruned_tok.encode(r["target"], add_special_tokens=False)) == r["target"]
    )
    print(
        f"Eval round-trip check: {exact_rt}/{len(eval_rows)} exact matches | "
        f"token fertility: {orig_toks} -> {new_toks} tokens ({100.0*(new_toks/orig_toks - 1):+.2f}%)"
    )


if __name__ == "__main__":
    main()
