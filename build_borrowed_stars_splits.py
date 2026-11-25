#!/usr/bin/env python3
"""
Build deterministic Borrowed Stars train/eval splits and domain-adaptation lore chunks.

Outputs (under ./data/):
  - bs_train.jsonl       : Stratified train pairs (NPC + Interactive Narrator)
  - bs_eval.jsonl        : Frozen held-out eval pairs (8 NPC + 8 Interactive Narrator)
  - bs_lore_chunks.jsonl : ~39.6k words of canonical lore + 30 narrator monologues chunked
                           for continued pretraining (CPT) / domain adaptation
  - bs_domain_terms.json : High-frequency Borrowed Stars proper nouns & domain terms
                           for vocabulary pruning / lore-token injection
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LORE_DIR = DATA_DIR / "borrowed_stars_lore"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chunk_markdown(text: str, source: str, max_chars: int = 1600, overlap_paras: int = 1):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = []
    buf_len = 0
    for p in paras:
        if buf and buf_len + len(p) + 2 > max_chars:
            chunks.append({"source": source, "text": "\n\n".join(buf)})
            buf = buf[-overlap_paras:] if overlap_paras > 0 else []
            buf_len = sum(len(x) + 2 for x in buf)
        buf.append(p)
        buf_len += len(p) + 2
    if buf:
        chunks.append({"source": source, "text": "\n\n".join(buf)})
    return chunks


def extract_domain_terms(texts: list[str], top_k: int = 256) -> list[dict]:
    """Extract capitalized / hyphenated Borrowed Stars proper nouns & domain terms."""
    counter: collections.Counter[str] = collections.Counter()
    pattern = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+| [A-Z][a-z]+)?\b")
    stop = {
        "The", "This", "That", "They", "There", "Their", "These", "Those",
        "When", "Where", "What", "Which", "While", "With", "Without", "From",
        "Into", "Only", "Even", "Every", "After", "Before", "Under", "Over",
        "Because", "Though", "Still", "Never", "Always", "Some", "Many", "Most",
        "Chapter", "Section", "Note", "None", "True", "False", "Episode",
    }
    for t in texts:
        for m in pattern.findall(t):
            if m not in stop and len(m) >= 4:
                counter[m] += 1
    return [{"term": term, "count": cnt} for term, cnt in counter.most_common(top_k)]


def main() -> None:
    npcs = load_jsonl(DATA_DIR / "borrowed_stars_npcs.jsonl")
    narr_int = load_jsonl(DATA_DIR / "narrator_interactive.jsonl")
    narr_sp = load_jsonl(DATA_DIR / "narrator_speech.jsonl")

    for r in npcs:
        r["subset"] = "npc"
    for r in narr_int:
        r["subset"] = "narrator_interactive"

    # Deterministic stride split: every 4th NPC (8 eval, 24 train) and every 5th Interactive Narrator (8 eval, 32 train)
    eval_npcs = [r for i, r in enumerate(npcs) if i % 4 == 3]
    train_npcs = [r for i, r in enumerate(npcs) if i % 4 != 3]

    eval_narr = [r for i, r in enumerate(narr_int) if i % 5 == 4]
    train_narr = [r for i, r in enumerate(narr_int) if i % 5 != 4]

    bs_train = train_npcs + train_narr
    bs_eval = eval_npcs + eval_narr

    write_jsonl(DATA_DIR / "bs_train.jsonl", bs_train)
    write_jsonl(DATA_DIR / "bs_eval.jsonl", bs_eval)

    # Lore + Narrator speech chunks for Continued Pretraining (CPT)
    lore_chunks = []
    all_texts = []
    for md in sorted(LORE_DIR.rglob("*.md")):
        txt = md.read_text(encoding="utf-8")
        all_texts.append(txt)
        rel = str(md.relative_to(DATA_DIR))
        lore_chunks.extend(chunk_markdown(txt, source=rel))

    for row in narr_sp:
        t = f"[STYLE: {row.get('style', 'narrator_monologue')}] [TOPIC: {row.get('topic', '')}]\n{row['text']}"
        all_texts.append(row["text"])
        lore_chunks.append({"source": "narrator_speech.jsonl", "text": t})

    write_jsonl(DATA_DIR / "bs_lore_chunks.jsonl", lore_chunks)

    terms = extract_domain_terms(all_texts)
    (DATA_DIR / "bs_domain_terms.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")

    print(
        f"Built Borrowed Stars splits: {len(bs_train)} train ({len(train_npcs)} npc + {len(train_narr)} narrator), "
        f"{len(bs_eval)} eval ({len(eval_npcs)} npc + {len(eval_narr)} narrator), "
        f"{len(lore_chunks)} lore/speech CPT chunks, {len(terms)} extracted domain terms."
    )
    print("Top 20 domain terms:", [x["term"] for x in terms[:20]])


if __name__ == "__main__":
    main()
