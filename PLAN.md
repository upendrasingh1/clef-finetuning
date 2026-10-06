# Clef Fine-Tuning for Real-Time Entity Resolution — Project Plan

Status: **ideation**. Nothing here is implemented yet. Items marked *(estimate)* or *(risk)* must be verified in Phase 0.

## 1. Goal

Fine-tune Cloudflare's open-weight **Clef** decision model locally (single RTX 3060, 12 GB) for **entity resolution (ER)**,
show measurable improvements from **Hugging Face TRL**-based training alongside Clef-native training, and ship a
**real-time local ER service**.

### Deliverables

1. Open-source repo (code, configs, eval harness, reproducible results).
2. Medium article series explaining the nuances (one article per phase, see §9).
3. Optional YouTube series / demo videos.

### Constraints (decided)

| Item | Decision |
| --- | --- |
| Hardware | Local only: RTX 3060 12 GB, 29 GB RAM, 20 CPU threads |
| Data | Public ER benchmarks first, then our own data |
| Serving | Local Python service |
| TRL role | All three tracks: A (comparison), B (teacher), C (TRL adapted to Clef) |
| TRL generative models | Both Qwen3.5-4B (bf16 LoRA) and Qwen3.5-9B (QLoRA) |

## 2. What Clef is (facts from sources, §11)

- **Clef** (27B, from Qwen3.8-27B) and **Clef-flash** (9B, from Qwen3.5-9B), Apache 2.0, vision-capable.
- Input: `state` (text/JSON/images) + `questions` typed as `noul` (bool), `choice` (named options), `score` (ordinal).
- Inference: backbone **prefill only**, then a **joint schema head** scores every option of every question in one pass.
  No text generation; output is one logit per option, softmax per question.
- Cloudflare training recipe: frozen backbone + rank-256 LoRA + head, **label-smoothed CE + Brier loss**, then
  **RLCD** (partial credit for adjacent ordinal options, exact-record reward, reference/KL penalty).
- Released: inference code (`joint_schema_model.py`) and weights. **Not released: training code, training data.**
- Clef-flash config: 32 layers (3:1 Gated-DeltaNet linear attention : full attention), hidden 4096, vocab 248,320.
  Head: width 1024, 2 evidence-routing layers + 4 decoder layers, ~120M params *(estimate)*.
- `ClefModel.forward` already unwraps PEFT models (`get_base_model`), so PEFT LoRA plugs in directly.
- Community: for the 27B model the Clef delta lives only in layers 40–63 (trevest GGUF/MLX ports); quantized
  backbones (4.4 bpw) reproduce Clef answers closely at inference.

## 3. Hardware reality and the key nuances

| Model | bf16 inference | Training on 12 GB |
| --- | --- | --- |
| Clef 27B | No | No |
| Clef-flash 9B | No (~18 GB) — needs 8-bit/4-bit/GGUF for serving *(risk: verify fidelity)* | Not as full bf16 LoRA (~22 GB per Unsloth) |
| Qwen3.5-4B | Yes | bf16 LoRA — yes |
| Qwen3.5-9B | No | QLoRA only — **Unsloth advises against 4-bit QLoRA on Qwen3.5** (high quantization error) |

### Nuance 1 — TRL cannot train Clef out of the box

All TRL trainers (SFT, DPO, GRPO, RLOO, KTO, ORPO, Reward, Distillation, …) target generative causal LMs.
`GRPOTrainer` docs: *"Only causal language models are supported."* Clef never generates tokens.
This is why the project has three TRL tracks (§5).

### Nuance 2 — Split-backbone training (makes Clef-flash trainable in bf16 on 12 GB)

Each decoder layer's input is only the residual stream from the layer below (per-layer recurrent/attention state
is recomputed inside the layer from that sequence). Therefore:

1. Run layers `0..K-1` once (bf16, with CPU offload — slow but one-time) and cache the residual stream at layer `K`
   for every training example to disk.
2. Cache the per-option lexical vectors (`lm_head` rows averaged over option tokens) — `lm_head` stays frozen.
3. Train LoRA on layers `K..31` + final norm + the joint head, in bf16, reading cached activations.

Starting point: `K = 24` (top 8 layers ≈ 1.7B params ≈ 3.5 GB bf16 *(estimate)*).
Disk: ~8 KB/token → ~3 MB per 400-token pair → ~30 GB per 10k pairs *(estimate)*; store as fp16/bf16 shards.
*(risk)* Must pass correct `position_embeddings` (M-RoPE) and masks when calling upper layers directly;
validate with a parity test (split forward vs. full forward, cosine ≥ 0.999 on hidden states, identical argmax).

### Nuance 3 — RL on a decision model needs no sampling

With a finite option set, expected reward is exact: `J = Σ_i p_i · r_i − β · KL(p ‖ p_ref)`.
Gradients are exact and cheap, with no rollouts. Cost-sensitive rewards (false merge ≫ missed match) fit naturally.
Track C (sampling via GRPO) vs. this exact objective is a direct, publishable comparison.

## 4. Entity resolution formulation

### 4.1 Pairwise (benchmarks)

```json
{
  "state": {"left": {"title": "...", "brand": "...", "price": 0}, "right": {"title": "...", "brand": "...", "price": 0}},
  "questions": {
    "same_entity": {"type": "noul", "instructions": "Do both records describe the same real-world product?"},
    "relation": {"type": "choice", "criteria": {
      "same": "Identical entity", "variant": "Same product line, different variant (size/colour/model)",
      "accessory": "One is an accessory/part of the other", "different": "Unrelated"}}
  }
}
```

The generative TRL models get the same records as a chat prompt and must answer `MATCH` / `NO_MATCH`
(+ relation). Same splits, same metrics.

### 4.2 Multi-candidate (real-time service)

One `choice` question whose options are the top-k candidates (`c_<id>`, description = candidate record JSON)
plus `new` (create new entity). One forward pass scores all candidates jointly.
Optional `score` question for routing: `Reject / Human review / Auto-merge`.

## 5. TRL tracks

| Track | What | Why |
| --- | --- | --- |
| **A — Two-paradigm comparison** | TRL `SFTTrainer` → `GRPOTrainer` (reward = correctness, asymmetric cost, format) on generative Qwen3.5-4B (bf16 LoRA) and Qwen3.5-9B (QLoRA). Compare against Clef-flash trained with the custom trainer. | Core "improvements post TRL" story with honest apples-to-apples evaluation. |
| **B — TRL as teacher** | Best TRL model from Track A labels unlabeled pairs and mines hard negatives; Clef-flash is trained on gold-subset + teacher labels. Also the bridge to our own (mostly unlabeled) data. | Shows TRL improving Clef indirectly; realistic low-label regime. |
| **C — TRL adapted to Clef** | Subclass `GRPOTrainer`: `rollout_func` (experimental hook) samples option IDs from Clef's distribution; override per-token log-prob computation to use the joint head instead of `lm_head`. Pinned TRL version. | Experimental chapter; compare sampled GRPO vs. exact expected-reward RL (Nuance 3). *(risk: relies on TRL internals.)* |

## 6. Experiment matrix

| ID | Model | Method | Track |
| --- | --- | --- | --- |
| E0a/b/c | Qwen3.5-4B / Qwen3.5-9B / Clef-flash | Zero-shot baselines | — |
| E1a/b | Qwen3.5-4B bf16 / Qwen3.5-9B QLoRA | TRL SFT | A |
| E2a/b | E1a/E1b | TRL GRPO from SFT checkpoint | A |
| E3 | Clef-flash | Head-only training (fully cached backbone) | Clef-native |
| E4 | Clef-flash | Split-backbone LoRA + head, CE + Brier | Clef-native |
| E5 | E4 | + exact expected-reward RL with KL to E4 | Clef-native |
| E6 | Clef-flash | E4 recipe on 10% gold + TRL-teacher labels | B |
| E7 | E4 | GRPO via adapted TRL trainer | C |
| E8 | (bonus) | 4-bit vs bf16 nuance: E1b vs a 4B/9B bf16 reference where feasible | A |

Ablations worth an article: label smoothing on/off, Brier on/off, K (split depth), LoRA rank, schema/field-order
permutation augmentation (Cloudflare used it), instruction wording.

## 7. Data

### Public benchmarks (Phase 1)

- **DeepMatcher / ER-Magellan** (fixed splits): Abt-Buy, Amazon-Google, Walmart-Amazon (structured + dirty),
  DBLP-ACM, DBLP-Scholar, iTunes-Amazon, Beer, Fodors-Zagats, Company.
  Source: github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
- **WDC Products** (seen/unseen entities, corner cases, multiple train sizes) — tests generalization to unseen entities.
- Comparison points: Ditto, fine-tuned LLM results in arXiv 2409.08185.

### Own data (Phase 6)

- Schema mapping into `state` JSON; label bootstrapping via Track B teacher + human review queue.
- Privacy: local only, no data leaves the machine.

## 8. Metrics

- Quality: precision, recall, **F1** (match class), macro-F1 on `relation`.
- Calibration: **ECE**, Brier score, reliability diagrams (Clef's selling point).
- Operations: p50/p95 latency on the 3060, throughput, VRAM; cost-weighted error (false merges weighted higher).
- Real-time: end-to-end latency (candidate retrieval + decision), human-review rate at chosen thresholds.

## 9. Phases and content

### 9.1 Series narrative (set in Article 1)

**Series thesis:** *Real-time entity resolution needs decisions, not text. Can one consumer GPU (12 GB) turn an open
decision model into a calibrated, real-time entity resolver, and what does TRL actually contribute along the way?*

Article 1 does not only introduce Clef. It frames the whole series: it states the problem, makes the TRL tension
explicit, lists the questions the series will answer, and explains why the articles come in this order. Every later
article answers one of those questions and ends by raising the next one.

**Questions posed in Article 1 (each answered by one later article):**

| # | Question | Answered in |
| --- | --- | --- |
| Q1 | How good is an untrained decision model at entity matching, and where exactly does it fail? | Article 2 |
| Q2 | How far does the standard TRL path (SFT → GRPO) get a generative model, and at what cost in latency and calibration? | Article 3 |
| Q3 | Can we fine-tune the decision model itself on 12 GB, even though TRL can't train it? | Article 4 |
| Q4 | Can TRL still improve Clef, indirectly (teacher) or directly (adapted GRPO)? | Article 5 |
| Q5 | Does the best model survive a real-time latency budget end to end? | Article 6 |
| Q6 | Does any of it hold on messy real-world data? | Article 7 |

**Why this order (explained in Article 1):**

1. **Measure before training** (Art. 2). No improvement claim is credible without a fixed baseline, and the
   baseline's failure modes (variants, accessories, unseen entities) tell us what the training has to fix.
2. **Familiar path before the novel one** (Art. 3). TRL SFT → GRPO is what most readers already know. It sets the
   bar that Clef-native training has to beat, and it shows the costs of generating answers as text (latency,
   format errors, poor calibration). Those costs motivate the next article.
3. **Train the decision model itself** (Art. 4). Only after the TRL bar exists does the harder split-backbone work
   pay off narratively. This article delivers the main technical result.
4. **Bring TRL back** (Art. 5). This closes the TRL tension from Article 1. It comes after Art. 3 and Art. 4
   because Track B needs Art. 3's best model as teacher and Track C needs Art. 4's model to start from.
5. **Ship** (Art. 6). Thresholds and latency budgets are tuned on the final model. Building the service earlier
   would mean tuning them for a model we then replace.
6. **Real data last** (Art. 7). Public benchmarks give reproducible, comparable numbers first. Own data then tests
   whether the findings transfer, and uses the review queue and feedback loop built in Art. 6.

Article 1 also states that negative results (e.g. 9B QLoRA degrading, Track C underperforming) will be reported.

**Recurring devices (every article):**

- **Scoreboard:** one table updated every article (F1, ECE, Brier, p50/p95 latency, VRAM) on the same splits.
- **Running example:** the same small set of hard pairs (an exact match, a product variant, an accessory, an
  unseen entity), shown through each model so readers watch the decisions change.
- **Nuance box:** the one non-obvious lesson of the article (see §3 nuances).
- **Hand-off:** the closing section states the open question that the next article answers.

### 9.2 Phase and article map

| Phase | Repo work | Article (question) | Hook into next article |
| --- | --- | --- | --- |
| 0 | Env (uv, torch, transformers 5.x, PEFT, TRL pinned), download Clef-flash, inference smoke test in 8-bit/4-bit, split-forward parity test | **1: "Decide, don't generate: building a real-time entity resolver on a 12 GB GPU"** (series framing, Clef vs LLMs, why TRL doesn't fit, Q1–Q6, running example scored zero-shot) | "Before training anything, how good is it already?" |
| 1–2 | Dataset loaders → unified pair format, eval harness (§8), E0 baselines | **2: "Zero-shot entity matching with Clef: the baseline scoreboard"** (Q1) | Clef is fast and calibrated but misses variants and unseen entities. Can the standard TRL recipe fix this? |
| 3 | Track A: E1, E2 (+E8) | **3: "SFT → GRPO with TRL for entity matching on a 3060"** (Q2) | TRL raised F1, but answers are text: slower, format errors, uncalibrated. Can we train the decision model directly? |
| 4 | Clef-native: E3, E4, E5 | **4: "Fine-tuning a 9B decision model on 12 GB (split-backbone)"** (Q3) | Clef-native wins on the scoreboard, but needs labels and skips TRL. Can TRL still help? |
| 5 | Tracks B and C: E6, E7 | **5: "TRL as teacher, and forcing GRPO onto a non-generative model"** (Q4) | We have a final model. Is it fast enough in a real-time pipeline? |
| 6 | Real-time service: FastAPI + blocking (exact keys + embeddings, FAISS/Qdrant) + Clef multi-candidate decision + identity graph (union-find) + human review queue + feedback loop | **6: "A real-time entity resolver: latency budgets, thresholds, feedback"** (Q5) + YouTube demo | Benchmarks are clean. What happens on our data? |
| 7 | Own data | **7: "From benchmarks to our data, and the final scoreboard"** (Q6, answers Q1–Q6 in a retrospective) | Series close; next steps (27B on rented GPU, Workers AI deployment) |

YouTube (optional) follows the same order: one video per article, with Article 1's video as a trailer showing the
running example scored zero-shot and the final demo target from Article 6.

### Real-time pipeline (Phase 6)

```
record in ─► normalize ─► blocking (exact keys + ANN top-k) ─► Clef-flash multi-candidate decision
          ─► policy (auto-merge > τ_hi, review band, new entity) ─► identity graph update
          ─► review decisions logged ─► training data for E5-style RL
```

## 10. Risks / open questions

1. Clef-flash bf16 inference does not fit 12 GB → serving needs 8-bit/4-bit or llama.cpp; measure answer drift vs bf16.
2. Split-backbone implementation details for Qwen3.5 hybrid layers (M-RoPE, linear-attention masks) — parity test first.
3. Qwen3.5 Triton/Gated-DeltaNet kernels on Ampere (sm_86) and compile time.
4. TRL `rollout_func` is experimental; Track C must pin a TRL version.
5. 9B QLoRA quality degradation (Unsloth warning) — treat as a measured finding, not a blocker.
6. GRPO generation cost on 12 GB (short completions help; vLLM colocate likely not feasible).
7. Benchmark licenses for redistribution — repo should download, not vendor, datasets.
8. Choice of embedding model and vector store for blocking.

## 11. References

- Cloudflare blog: https://blog.cloudflare.com/clef-decision-models/
- Clef: https://huggingface.co/Cloudflare/clef — Clef-flash: https://huggingface.co/Cloudflare/clef-flash
- Workers AI docs: https://developers.cloudflare.com/workers-ai/models/clef/
- Community LoRA/GGUF port: https://huggingface.co/trevest/Clef-LoRA-for-Qwen3.8-27B-GGUF
- TRL GRPO: https://huggingface.co/docs/trl/grpo_trainer
- Unsloth Qwen3.5 fine-tuning: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Fine-tuning LLMs for Entity Matching: https://arxiv.org/html/2409.08185v2
- WDC Products benchmark: https://openproceedings.org/2024/conf/edbt/paper-14.pdf
- DeepMatcher datasets: https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
- Ditto: https://arxiv.org/pdf/2004.00584

## 12. Repository structure and release rules

Repo `clef-finetuning`, package `clef_finetuning`, Apache-2.0. Skeleton is in place; modules are stubs whose
docstrings name the article they land with.

```
configs/        experiments/ (one YAML per id e0a…e8), data/, hardware/, models.yaml (pinned HF revisions)
src/clef_finetuning/
  config.py     loads and cross-validates configs
  data/ schema/ models/ (clef_upstream/ = Cloudflare code, unmodified) cache/ train/ eval/ serve/
scripts/        run_experiment.py, validate_configs.py
tests/          CPU tests; GPU tests marked `gpu` (make test-gpu)
results/        committed metrics.json per experiment + scoreboard.md only
articles/NN-*/  README (question, experiments, tag, reproduce commands, hand-off) + figures/
```

Rules:

1. **Experiments, not articles, own the code.** `make <id>` runs an experiment; articles reference experiment ids.
2. **Published configs are immutable.** New ideas get new ids; `validate()` rejects dependencies on later articles.
3. **One tag per article** (`article-NN`) with a GitHub release linking Medium and YouTube.
4. **Upstream stays untouched.** `clef_upstream/joint_schema_model.py` is pinned by SHA-256 in a test.
5. **No data, weights, caches or checkpoints in git.** Datasets are downloaded from source; trained adapters go to
   the Hugging Face Hub.
6. **Supply chain.** Dependencies are added with `uv add`; `[tool.uv] exclude-newer = "7 days"` blocks releases
   younger than a week. Heavy dependencies (torch, transformers 5.x, PEFT, TRL pinned, bitsandbytes) are added in
   Phase 0 as `uv` optional groups.
