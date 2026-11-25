# borrowed-stars-npc-slm

Offline, sub-500 MB CPU language models for in-game RPG NPCs and the Interactive Narrator in ***Borrowed Stars*** (Anima engine).

Every model runs strictly on CPU via `llama.cpp` (zero GPU contention with rendering), uses a single-shot JSON contract with 23 validated in-game tools, and enforces per-turn dynamic GBNF allowlists so NPCs cannot invent items, break quests, or leak retired lore names.

* **Hugging Face Collection:** [queelius/borrowed-stars-offline-cpu-npc-models](https://huggingface.co/collections/queelius/borrowed-stars-offline-cpu-npc-models-6b06eb71ff68721ad0f37bda)
* **Full Engineering Devlog & Post-Mortem:** [docs/DEVLOG_POSTMORTEM.md](docs/DEVLOG_POSTMORTEM.md)

---

## Pretrained Models (`v2.1`, 23-Tool Contract, `Q4_K_M` GGUF)

Scored unconstrained (before GBNF) on a frozen 120-row stratified eval split (2 rows per tool across all 23 tools, 5 rows per each of 6 refusal categories), then verified live on a regular **AMD Ryzen 5 7430U laptop CPU (6 threads, 0 GPU VRAM)** with `npc_grammar_v2.gbnf`:

| Tier | Hugging Face Repo | Size | `val_bpb` | Raw JSON | 23-Tool F1 | Anti-Exploit Refusals | Kill-List Leaks | Laptop CPU Decode (6t) | Full Turn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Balanced (Default, Apache 2.0)** | [`queelius/borrowed-stars-npc-qwen3.5-0.8b-32k-GGUF`](https://huggingface.co/queelius/borrowed-stars-npc-qwen3.5-0.8b-32k-GGUF) | `327.9 MB` | `0.4908` | `91.7%` | **`0.747`** | **`96.7%`** | **`0.00%`** | `66–68 tok/s` | `1.5–1.7 s` |
| **Ultra-Fast (Low-Spec Laptop)** | [`queelius/borrowed-stars-npc-lfm2.5-350m-GGUF`](https://huggingface.co/queelius/borrowed-stars-npc-lfm2.5-350m-GGUF) | **`218.7 MB`** | `0.4944` | **`92.5%`** | `0.683` | `90.0%` | **`0.00%`** | **`123–125 tok/s`** | **`0.88–0.94 s`** |
| **Max Prose (32K Native SPM)** | [`queelius/borrowed-stars-npc-gemma3-1b-32k-GGUF`](https://huggingface.co/queelius/borrowed-stars-npc-gemma3-1b-32k-GGUF) | `495.5 MB` | **`0.4107`** | `90.8%` | `0.716` | `86.7%` | **`0.00%`** | `48–51 tok/s` | `2.6–3.0 s` |

Every shipping model passed `4/4` live `llama.cpp` + `npc_grammar_v2.gbnf` checks (`give_item`, `steer_dialogue_node`, Ontological Horizon refusal, pure chitchat) with `0/480` retired-name Kill-List leaks and `0/480` out-of-character leaks.

---

## Quick Start

### 1. Run straight from Hugging Face with `llama-server`
```bash
# Default 328 MB Apache 2.0 model (note --no-jinja so Qwen3.5 skips <think> tags):
llama-server -hf queelius/borrowed-stars-npc-qwen3.5-0.8b-32k-GGUF:Q4_K_M -t 6 -c 2048 --port 8091 --no-jinja

# Or the 219 MB 125 tok/s laptop tier:
llama-server -hf queelius/borrowed-stars-npc-lfm2.5-350m-GGUF:Q4_K_M -t 6 -c 2048 --port 8091
```

### 2. Use the bundled launcher (`serve_npc.py`)
`serve_npc.py` downloads/builds a standalone `bin/llama-server`, exposes both native `POST /completion` (with per-turn GBNF + KV prefix caching) and OpenAI-compatible `POST /v1/completions` / `POST /v1/chat/completions`, and includes a live CPU smoke test:

```bash
# Laptop CPU (exposes ANIMA_SLM_URL=http://127.0.0.1:8091 and OpenAI http://127.0.0.1:8091/v1):
python3 serve_npc.py --tier balanced --port 8091

# Run the built-in GBNF + prompt-injection smoke test:
python3 serve_npc.py --tier balanced --smoke-test

# Host on a LAN GPU box (e.g. muse):
python3 serve_npc.py --tier balanced --gpu --lan --port 8091
```

### 3. Pull into `muse`
```bash
muse pull hf://queelius/borrowed-stars-npc-qwen3.5-0.8b-32k-GGUF@q4_k_m
```

---

## How the Runtime Contract Works

Instead of slow multi-turn agent loops (`get_inventory` -> wait -> `give_item`), Anima splits read state from write actions:

1. **Inject compact read state into the prompt header (~280–440 tokens):** `[LOC]`, `[NPC]`, `[VOICE]`, optional `[BIO]`/`[RECENT]`/`[PEER_CHAT]`/`[TREE]` (omitted when empty), `[NPC_INV]`, `[PLAYER]`, `[QUESTS]`, `[FACTS]`, `[ALLOWED_TOOLS]`, `[HIST]`, `[SAY]`.
2. **Emit speech and 0–2 write actions in one JSON pass:**
   ```json
   {"mood":"wary","line":"Aye, a storm's coming. Folks are fretting, and the tides are shifting.","calls":[{"tool":"steer_dialogue_node","args":{"node_id":"ask_squall"}}]}
   ```
3. **Compile a per-turn GBNF grammar:** Narrows `give_item` strictly to items in `[NPC_INV]` and `steer_dialogue_node` strictly to open `[TREE]` branches.
4. **Train explicit refusals and a ~55% zero-call balance:** Prevents tool-happy small models and blocks unowned-item demands, 1-gold price exploits, wrong-role demands, Ontological Horizon spoiler probes, retired Naming Kill-List bait, and prompt injections.

See [data/BORROWED_STARS_TOOLS_SPEC.md](data/BORROWED_STARS_TOOLS_SPEC.md) and [npc_grammar_v2.gbnf](npc_grammar_v2.gbnf) for the full 23-tool schema.

---

## Key Engineering Lessons (Summary)

* **Vocabulary pruning is a massive win below 1B parameters:** 30–63% of multilingual sub-1B models live in the embedding table. BPE-safe / native-SentencePiece pruning to 32,768 English + game tokens cut `Gemma 3 1B` from `776.5 MB` to `495.5 MB` (`-281 MB`) and boosted 4-thread CPU decode by **+77%**.
* **56 hand-authored dialogues overfit 1B models in 60 seconds:** Grounded lore synthesis (`1,000` teacher pairs + `3x` golden replay + `0.05` label smoothing + best-epoch restoration) fixed calibration collapse immediately.
* **Watch silent rejection filters in synthetic pipelines:** Always log per-category acceptance counts before launching SFT. Showing a concrete per-tool JSON example in the teacher prompt jumped acceptance to 91% across all 23 tools.

Read the full story, failure analysis, and benchmark tables in [docs/DEVLOG_POSTMORTEM.md](docs/DEVLOG_POSTMORTEM.md).

---

## Repository Map

| File | Role |
| :--- | :--- |
| `serve_npc.py` | Local & LAN `llama-server` launcher (native GBNF `/completion` + OpenAI `/v1/*` + `--smoke-test`) |
| `npc_grammar_v2.gbnf` | Full 23-tool v2.1 GBNF grammar for `llama.cpp` |
| `validate_v2.py` | Strict JSON schema, dynamic `[ALLOWED_TOOLS]`, refusal, and Kill-List validator |
| `shootout.py` | Stage 1 calibrated multi-model shootout (`val_bpb` + lore CPT + SFT) |
| `synth_lore_data.py` | Stage 1 grounded lore-to-dialogue synthesizer (`gemma-3-12b-it`) |
| `synth_tools_v2.py` / `synth_tools_v21_targeted.py` | Stage 2 balanced 23-tool + refusal + v2.1 context-tag synthesizer |
| `train_tools_v2.py` | Stage 2 23-tool SFT + regression-guarded On-Policy Distillation (OPD) + GGUF export |
| `vocab_prune.py` | BPE-safe & native SentencePiece (`tokenizer.model`) 32K vocabulary pruner |
| `export_and_bench_gguf.py` | `llama.cpp` F16 -> `Q4_K_M` quantizer, 4t/8t `llama-bench`, and live GBNF verifier |
| `publish_hf.py` | Principled Hugging Face multi-repo + Collection publisher |
| `prepare.py` / `train.py` | Frozen `val_bpb` evaluation harness and `autoresearch` training loop |
| `docs/DEVLOG_POSTMORTEM.md` | Engineering blog post, X/Twitter thread, and YouTube post-mortem outline |
| `experiments.tsv` & `results/` | Complete Stage 1 and Stage 2 benchmark metrics |

*(Note: Full *Borrowed Stars* world-bible and campaign storyline files are intentionally excluded from this public repo to avoid game spoilers; starter schema and sample files are included in `data/`.)*
