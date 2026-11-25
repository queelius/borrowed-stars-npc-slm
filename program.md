# program.md — Autonomous Research Director for `npc-text-gen`

You are running an autonomous Karpathy-style `autoresearch` loop to produce the **best possible <=500MB, CPU-fast in-game NPC language model** specialized for world-building dialogue.

## Rules of the Loop
1. **NEVER modify `prepare.py` or `data/eval_npc.jsonl`.** Those define the frozen, apples-to-apples evaluation ground truth.
2. **ONLY modify `train.py`** (and optionally `npc_grammar.gbnf` if explicitly requested).
3. Every trial runs under a fixed wall-clock budget (`--budget-sec 300` on GPU, or `--budget-sec 60` for rapid CPU/looped-scratch ablations).
4. **Hard Gates (must pass all to keep a commit):**
   - `artifact_mb <= 500.0` (quantized weight size + tokenizer + any auxiliary n-gram table)
   - `valid_schema_rate >= 0.95` (when GBNF/structured decoding is evaluated)
   - No NaNs, divergence, or OOM.
5. **Primary Optimization Metric:** Minimize **`val_bpb`** (validation bits-per-byte on `data/eval_npc.jsonl`). Because `val_bpb` is normalized by UTF-8 bytes rather than tokens, experiments that prune vocabularies or inject custom world-lore tokens remain strictly comparable.
6. **Ratchet Discipline:**
   - Before each run, log a 1-sentence hypothesis in `experiments.tsv`.
   - Run `train.py`, record `val_bpb`, `artifact_mb`, `decode_tok_s`, and `status`.
   - If `val_bpb` improves AND hard gates pass: `git commit -am "exp: <summary> (val_bpb=X.XXXX)"`.
   - Otherwise: `git checkout -- train.py` and move to the next hypothesis.

## High-Leverage Research Axes to Explore
1. **Vocabulary Pruning + World-Lore Token Injection:**
   - Trim unused multilingual tokens (e.g., 151K–256K -> 24K–32K) using BPE merge-closure.
   - Reallocate 256–512 pruned slots to high-frequency game proper nouns, factions, and schema markers (`<mood>`, `<action>`, `<line>`).
2. **Looped Depth & Parameter Sharing (Inkling 2 / Ouro / Parameter Golf):**
   - Test Prelude + Recurrent Core ($R \in \{1, 2, 3, 4\}$) + Coda with **attention residuals** and **intermediate deep supervision** ($\sum w_t \mathcal{L}_t$).
   - Verify whether smaller physical weights looped $2\times$–$3\times$ beat a flat model at the same byte budget AND still clear the CPU tok/s floor.
3. **Quantization-Aware Training (QAT):**
   - Train/adapt through fake INT4 / INT5 / 1.58-bit ternary STE weights so quantization causes near-zero `val_bpb` regression.
4. **On-Policy Distillation (OPD):**
   - When a teacher model is available (e.g., `Qwen3-4B`, `gemma-3-12b-it` on `siue`, or Colab teacher), sweep reverse-KL $\beta$, student rollout temperature, and SFT-to-OPD mixing ratio.
5. **Entropy-Adaptive N-Gram Cache Blending (Parameter Golf Trick):**
   - Blend a lightweight per-zone / per-dialogue n-gram cache ($\le 2\text{ MB}$) with neural logits using an entropy-adaptive interpolation weight $\lambda(H)$ to boost lore & entity consistency at near-zero compute cost.
6. **Optimizer & Training Stability:**
   - Muon vs. AdamW, cosine vs. WSD learning-rate schedules, EMA weight averaging, SmearGate / BOS masking.
