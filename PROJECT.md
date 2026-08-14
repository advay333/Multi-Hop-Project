# Learning Multi-Hop Table Reasoning through RL: PPO-Based Entity Selection

**Project context document.** Read this first. It is the single source of truth for
scope, architecture, module interfaces, and success criteria. Individual subtasks have
their own spec files (`TASK_01_DATASET.md`, etc.); those specs assume this document has
been read.

Team size: 2. Duration: 12 weeks. Target: workshop paper + open-source release.

---

## 1. The problem in one page

### 1.1 Task

Given a natural-language question `Q` and a relational database of tables
`T = {T_1 ... T_m}` connected by foreign keys, produce the answer.

Example (from MMQA):

```
Tables:
  Head       (Head_ID, Name, Born_State)
  Management (Department_ID, Head_ID)
  Department (Department_ID, Name, Creation, Ranking, Num_Employees)

Q: "What are the distinct creation years of the departments managed by a
    secretary born in state Alabama?"

Correct reasoning:
  hop 1: Head       WHERE Born_State='Alabama'        -> Head_ID = {1}
  hop 2: Management JOIN on Head_ID, carry {1}        -> Department_ID = {7}
  hop 3: Department JOIN on Department_ID, carry {7}   -> Creation = {1903}

Answer: 1903
```

### 1.2 Why it is hard

Two facts, both from the MMQA paper (ICLR 2025):

1. **Tables do not fit in context.** Reported average rows per table: 1833 (2-table
   subset), 1369 (3-table subset). Three tables of ~1800 rows flattened to text is
   >100k tokens.
   *(This figure MUST be verified empirically — see gate G-2 in §8. Spider databases
   are often tiny, so this mean may be driven by a few outlier DBs.)*
2. **Accuracy collapses before the context limit.** MMQA found an elbow at ~800 rows,
   past which LLM QA performance drops sharply.

So the bottleneck is **not reading comprehension** — a reader handles 20 correct rows
fine. The bottleneck is **finding those 20 rows**.

### 1.3 The motivating number

MMQA foreign-key selection accuracy:

| Model | 2-table FKS | 3-table FKS |
|---|---|---|
| o1-preview (1-shot) | 34.17 | 40.78 |
| GPT-4 (1-shot) | 26.88 | 28.38 |
| Human | 95.3 | 97.5 |

Frontier models are ~55–60 points below human on a purely *structural* task. That gap
is what this project attacks. Put this table on slide 2.

For reference, end-to-end Table QA Exact Match on MMQA: o1-preview 1-shot reaches
50.78 (2-table) / 48.28 (3-table); human is 89.8 / 92.3.

### 1.4 Why RL and not supervised learning

The quantity we care about (did the final answer match gold?) is a function of a
sequence of **discrete** choices passed through a **frozen black-box LLM**. There is no
differentiable path from that outcome back to the selection parameters. Between the
parameters and the loss sit: a sample/argmax over candidates, a SQL execution, and
another network we are not training. Gradients die at each. RL replaces the missing
gradient with a score-function estimator built from sampled outcomes.

### 1.5 The asymmetry that justifies the project

Every training instance has **executable gold SQL**. Parse it, truncate the join chain
into prefixes, execute each prefix, and you obtain the exact correct entity set at every
hop — for free, with zero annotation.

StepSearch (EMNLP 2025) needed a synthetic data pipeline to *manufacture* this
step-level supervision for text retrieval. Relational schemas hand it to us. That is the
core argument of the paper.

---

## 2. Scope decision: decision-level RL, not token-level RL

**State this explicitly in the first slide and the first paragraph of the paper.**
Reviewers will otherwise assume Level 1 and conclude we cut corners.

| | Level 1: token-level RL | Level 2: decision-level RL (**ours**) |
|---|---|---|
| Policy is | the LLM itself | a small MLP head |
| An action is | emit next token | pick 1 of ~5–30 enumerated candidates |
| Horizon | thousands of steps | 2–4 steps |
| LLM weights | trained (or LoRA) | **frozen, never differentiated through** |
| Compute | 4–8 A100s, verl/TRL | 1 GPU, ~300 lines of PPO |

Level 2 is not a compromise. It is what makes the critic ablation (§5.3) *answerable at
all*: at H=4 with |A|≈20 you can reason about whether a value function helps. At
H=2000 token-steps you cannot.

---

## 3. MDP formulation

| Element | Instantiation |
|---|---|
| Episode | one question. Length H = 2..4. |
| State `s_t` | fixed-length feature vector (§4.2). Not a discrete index. |
| Action `a_t` | one entry from an environment-supplied list: `(predicate, fk_edge, carry_strategy)`, plus a `STOP` action |
| Transition | **deterministic** — a SQL join on SQLite. All randomness is in the policy. |
| Reward | terminal answer match + per-hop shaping + FK correctness − token cost (§5.2) |
| Discount γ | **1.0**. Horizon is 4; discounting buys nothing and costs a knob. |

### Objective

```
J(θ) = E_{τ~π_θ} [ α·R_ans + Σ_t (β1·r_shape_t + β2·r_fk_t) − λ·C(τ) ]
```

where `C(τ)` is reader tokens consumed.

### Why not Q-learning / SARSA

Four reasons, increasing in importance:

1. State space is not enumerable. (Fixable — that's DQN.)
2. **The action space is content-dependent, not an index set.** "Action 2" means
   different things in different states. You cannot have a Q-head with fixed output
   units. You must score each candidate by embedding it (pointer-network shape). This is
   natural for a policy, awkward for a bootstrapped Q-function.
3. Reward is terminal, sparse, and noisy (comes from a stochastic LLM). Off-policy
   bootstrapping compounds error.
4. ε-greedy explores badly. The ~20 candidates are semantically *similar* join options;
   uniform-random choice among them is near-useless. You want a learned stochastic
   policy that concentrates on the plausible 3.

---

## 4. Architecture

### 4.1 Data flow at inference time

```
question + tables
      |
      v
  [ state featuriser ]  <-- frozen encoder, PRECOMPUTED CACHE (dict lookup)
      |
      v
  [ policy head + value head ]  <-- THE ONLY TRAINABLE PART (~3M params)
      |            ^
      |            |
      |     [ candidate action enumerator ]  <-- schema graph walk, no NN
      v
  [ environment step ]  <-- SQL join on SQLite, deterministic, milliseconds
      |
      +--> loop for t = 0 .. H-1
      |
      v
  [ frozen reader LLM ]  <-- vLLM, reads final sub-table, ~500 tokens
      |
      v
   answer string
```

During **training only**, a parallel branch runs the reward oracle (gold SQL prefixes)
and feeds `rewards.py`. It must NEVER feed `featurise.py`. See §6.

### 4.2 State representation

| Component | Source | Dim |
|---|---|---|
| Question | frozen encoder embedding (constant per episode) | 1024 |
| Current table | embedding of `"tablename: col1, col2, ..."` | 1024 |
| Frontier set `E_t` | attention-pool of value embeddings, query = question | 1024 |
| Frontier scalars | `|E_t|`, is_numeric, min/max, distinct ratio | ~8 |
| Consumed mask | which tables/columns already used | ~30 |
| Progress | one-hot hop index, tokens_used/budget | ~8 |

**Two rules that decide whether Week 10 transfer works:**

1. **Never index by database-specific integers.** No "table #7", no learned per-column
   embedding. Everything must be an embedding of the *name and content*, so a policy
   trained on a university DB can operate on an airline DB it has never seen. A
   table-ID one-hot will silently make transfer performance chance-level.
2. **Be permutation-invariant over the frontier set.** Carrying `{1,7}` must produce the
   same state as `{7,1}`. Pooling gives this free; ordered concatenation does not.

### 4.3 Action enumeration

```python
def candidate_actions(state) -> list[Action]:
    A = []
    for edge in schema_graph.edges_from(state.current_table):
        if edge.target in state.consumed:
            continue
        for pred in enumerate_predicates(state):      # from question literals
            for strat in ["carry_all", "top_1", "top_5", "top_20"]:
                A.append(Action(edge, pred, strat))
    A.append(Action.STOP)
    return A
```

`enumerate_predicates` scans the question for literals (string spans, numbers, dates)
and fuzzy-matches them against cached cell values per column. This is where the
`Born_State='Alabama'` decision actually gets made.

**Include a cardinality feature.** Run `SELECT COUNT(*)` on the candidate join before
scoring (milliseconds, or free from SQLite statistics). "This join returns 400 rows,
that one returns 3" is the first thing a human DBA looks at, and it is the signal that
distinguishes a cheap route from an expensive one.

### 4.4 Policy scoring (pointer-network shape)

```
logit_i = <f_θ(s_t), g_θ(a_i)> / sqrt(d) * exp(τ)
π(·|s_t) = softmax(logits)
```

Two towers, both projecting to the same `d = 256`. Handles |A| varying from 4 to 40
with no architectural change and generalises to unseen schemas because nothing is
indexed.

**Why d must be shared:** the dot product requires it. That is the only architectural
constraint; `d = 256` itself is a hyperparameter (sweep {64,128,256,512} in Week 9;
expect a flat curve).

**What d controls:** the rank of the implicit bilinear form `s^T W a`. It caps how many
independent "reasons a candidate could be good" the score can express — name/question
match, cardinality, is-PK, does-literal-appear-in-column, already-used, budget-left,
frontier-blowup. Maybe 10–20 latent factors, so 256 is comfortable headroom.

**Why not use the encoder's native 1024 directly:** dotting raw embeddings *is* cosine
similarity in the encoder's semantic space, which is exactly baseline (d) in §5.4. The
learned projection IS the entire difference between the policy and the heuristic. Frame
it in the paper as learning a task-specific metric over a frozen semantic space.

### 4.5 Parameter count

| Module | Shape | Params |
|---|---|---|
| State tower | 3100 → 512 → 256 | 1.72 M |
| Action tower | 2100 → 512 → 256 | 1.21 M |
| Value head | 256 → 128 → 1 | 0.03 M |
| **Total trainable** | | **~3 M** |

The critic shares the state tower with its own small head.

### 4.6 Three implementation details that matter more than d

```python
class Tower(nn.Module):
    def __init__(self, d_emb, d_scalar, d_hidden=512, d_out=256):
        self.emb_norm    = nn.LayerNorm(d_emb)
        self.scalar_norm = nn.BatchNorm1d(d_scalar)   # SEPARATE — different scales
        self.net = nn.Sequential(
            nn.Linear(d_emb + d_scalar, d_hidden), nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out))               # init std=0.01
```

1. **Normalise scalars separately.** Embedding entries are ~0.03; `|E_t|` can be 400;
   join cardinality can be 50000. Concatenated raw, the cardinality feature contributes
   ~1e6× the gradient of any embedding dim and the network learns to *ignore* the
   scalars. Log-transform count-like features: `log1p(cardinality)`.
   **This is the bug that will silently cost a week.**
2. **Scale logits and use a learnable temperature.** Raw dot products in 256-d have
   magnitude ~sqrt(d), which saturates softmax at init.
3. **Initialise the last layer small** (`std=0.01` or zero) so the policy starts near
   uniform. Otherwise BC spends its first epoch undoing random opinions.

**Ablation to run once (Week 8):** replace the dot product with
`mlp(cat[f_s, g_a, f_s*g_a])`. Strictly more expressive, ~1M more params, but loses the
"learned metric" story and prevents precomputing action embeddings independently of
state. Start with the dot product.

---

## 5. Learning algorithms

### 5.1 The ladder (needed for the seminar talk)

1. **Policy gradient.** `∇J = E[Σ_t ∇log π(a_t|s_t) · G_t]`. Read mechanically: push up
   log-prob of actions followed by high return. `loss = -(logprob * G).mean()`.
2. **Baseline.** Subtract any `b(s_t)` not depending on `a_t` — unbiased. Best choice is
   `V^π(s)`, giving advantage `A_t = G_t - V(s_t)`: *how much better than expected*.
3. **Actor-critic.** Learn `V_φ` by regression onto returns. Its loss is the squared TD
   error already familiar from SARSA. GAE(λ) interpolates 1-step TD (biased, low
   variance) and Monte-Carlo (unbiased, high variance) — same tradeoff as TD(λ)/
   eligibility traces. At H=4 this knob barely matters: λ=0.95 and forget it.
4. **PPO.** Rollouts are expensive (each ends in an LLM call), so reuse each batch for
   ~4 epochs. But after one step the batch is off-policy, so correct with
   `r_t = π_θ/π_old`. If `r_t` blows up, one update wrecks the policy. Hence:

```
L_CLIP = E[ min( r_t·A_t , clip(r_t, 1-ε, 1+ε)·A_t ) ]        ε = 0.2
L_total = L_CLIP - c1·L_value + c2·H[π]
```

The clip is a leash: you may move an action's probability, but the incentive to keep
moving it switches off once the ratio leaves [0.8, 1.2]. The entropy bonus is the
principled replacement for ε-greedy.

### 5.2 Reward design

**Sparse:** `R = 1[answer == gold]` at t=H only. With 4 hops × ~25 candidates that is
~400k trajectories and **one bit** per rollout. Hopeless alone.

**Dense, from gold SQL prefixes:**

```
r_shape_t = γ·Φ(s_{t+1}) − Φ(s_t)        where Φ(s_t) = recall(E_t, E_t^gold)
r_fk_t    = 1[chosen join edge == gold edge]
C(τ)      = reader tokens consumed
```

**Why potential-based shaping specifically (Ng, Harada & Russell 1999).** Hand-crafted
dense rewards change what the optimal policy IS — the agent could learn to slavishly
reproduce the gold SQL's join order even when a different order reaches the same answer
more cheaply. Potential-based shaping provably does not, because the total shaping
contribution telescopes:

```
Σ_{t=0}^{H-1} γ^t (γΦ(s_{t+1}) − Φ(s_t)) = γ^H·Φ(s_H) − Φ(s_0)
```

It depends only on the endpoints, not the path. So it cannot reorder preference over
trajectories — it only **redistributes credit in time**, which is exactly the help
wanted. This is the theoretically defensible core of the paper.

The token-cost penalty is **not** potential-based. It deliberately changes the
objective, because cheap answers are part of what we want.

### 5.3 The PPO vs GRPO ablation (the interesting finding)

GRPO (Shao et al. 2024, behind DeepSeek-R1) deletes the critic: sample G trajectories
for the *same* question and use the group's normalised return as the baseline.

```
Â_i = (R_i − mean(R_1..G)) / std(R_1..G)      applied uniformly to every step of τ_i
```

| | PPO | GRPO |
|---|---|---|
| Baseline | learned `V_φ(s)` | empirical group mean |
| Critic | yes | none |
| Rollouts/question | 1 | G (4–16) |
| Credit assignment | **per-step**, via GAE | **uniform** over trajectory |
| Cost per signal | 1 reader call | G reader calls |

**Why GRPO won for LLM training:** at 2000-token horizon the "state" is a half-written
essay; fitting V(s) is near-hopeless, and generation is batched so 8× costs little.

**Why PPO plausibly wins here:**
1. `V(s)` is genuinely learnable — the state is a 256-d vector and "will I find the
   answer from this frontier?" is largely predictable from `|E_t|` and question-frontier
   similarity.
2. The G× cost is expensive here, not cheap: every rollout ends in a reader call.
   ~1.5 h/run becomes ~12 h/run.
3. **Decisive:** GRPO's uniform advantage cannot express *which hop* was good. Our whole
   premise is that hop 1 is fatal and hop 3 is recoverable — an asymmetry GRPO
   structurally cannot represent.

**The interaction that makes this a real result.** Reason 3 also means our per-hop
shaping gets summed into the return and its temporal structure is discarded under GRPO.
**Potential-based shaping only pays off with a per-step advantage.** So run the 2×2:

| | Sparse | Shaped |
|---|---|---|
| PPO | baseline | expect best |
| GRPO | ≈ PPO-sparse | little gain over GRPO-sparse |

Prediction: shaping helps PPO substantially more than GRPO. If it holds, we have a
clean mechanistic finding — *dense process rewards and critic-free RL are partial
substitutes, and combining them wastes the dense signal.* Worth more than 3 points of
EM.

**Measurement rule:** plot return against **number of reader calls**, not gradient
steps. Measuring against gradient steps hands GRPO a free 8× and makes the comparison
meaningless.

### 5.4 Baselines (all with the identical frozen reader)

| id | Name | Description |
|---|---|---|
| a | truncate | flatten tables, take first B tokens. No selection. |
| b | cot | prompted Chain-of-Table-style planner, LLM decides operations |
| c | mtr | reimplementation of MMQA's Multi-Table Retrieval + reader |
| d | heuristic | greedy cosine-similarity join + top-k values, no learning |
| e | bc | behaviour cloning on oracle trajectories |
| f | bc+beam | **BC + beam search over 5 routes, reranked by reader** |
| g | ppo-sparse | PPO, terminal reward only |
| h | ppo-shaped | PPO + potential shaping + FK + cost |
| i | grpo-sparse / grpo-shaped | critic-free comparison |
| j | oracle | gold entity sets fed directly — the CEILING |

**Baseline (f) is the real competition, not (b) or (c).** If we benchmark against
Chain-of-Table and MTR and declare victory, someone will ask about (f) in Q&A. It must
exist from Week 4.

**The rule that makes all of this meaningful: swap the selector, hold the reader
fixed.** Same model, same prompt template, same token budget. Only then does a delta in
EM say something about selection quality rather than reading quality.

---

## 6. Repository layout and module contracts

```
mhtable-rl/
├── data/
│   ├── raw/          mmqa.json, spider/, train_synth.json
│   ├── db/           *.sqlite
│   ├── cache/        embeddings.npz, schema_graphs.pkl
│   ├── trajectories/ oracle_traj.jsonl
│   └── splits/       train_dbs.txt, dev_dbs.txt, test_dbs.txt
├── src/
│   ├── data_prep.py      TASK_01 — builds db/, cache/, splits/, train_synth
│   ├── schema_graph.py   Spider tables.json -> graph
│   ├── env.py            reset / candidate_actions / step
│   ├── featurise.py      state -> Tensor, via cache
│   ├── oracle.py         sqlglot prefixes -> E_t^gold
│   ├── policy.py         policy head + value head
│   ├── reader.py         vLLM client
│   ├── rewards.py        terminal + shaping + fk + cost
│   ├── rollout.py        SHARED trajectory collector
│   ├── bc_train.py
│   ├── ppo.py
│   ├── baselines.py
│   ├── evaluate.py       single entry point, all eval modes
│   └── metrics.py
├── configs/          env.yaml, bc.yaml, ppo.yaml, ablations/
└── results/
```

### 6.1 The one architectural decision that matters

**`rollout.py` is shared between training and every evaluation mode.** It takes flags —
`sample_or_argmax`, `oracle_mode`, `selector` — and nothing else changes. Train/eval
divergence caused by two separate rollout implementations is the single most common
source of silently wrong RL numbers.

### 6.2 Key interfaces

```python
# schema_graph.py
class SchemaGraph:
    def edges_from(self, table: str) -> list[FKEdge]
    def is_pk(self, table: str, col: str) -> bool

# env.py
class TableEnv:
    def reset(self, instance) -> State
    def candidate_actions(self, s: State) -> list[Action]
    def step(self, a: Action) -> tuple[State, bool]   # (next_state, done)
    def serialise_final(self, s: State, budget: int) -> str

# featurise.py
def featurise_state(s: State, cache: EmbCache) -> Tensor      # [d_state_in]
def featurise_actions(A: list[Action], cache) -> Tensor       # [|A|, d_act_in]

# oracle.py
def gold_entity_sets(instance) -> list[set]                   # E_1^gold .. E_H^gold
def gold_join_path(instance) -> list[FKEdge]
def match_action(A: list[Action], E_gold: set) -> int | None   # index or None

# reader.py
def read_batch(subtables: list[str], questions: list[str]) -> list[str]

# rewards.py
def assemble(traj, cfg) -> list[float]    # per-step reward vector

# rollout.py
def collect(env, policy, reader, instances, *,
            sample_or_argmax: str,        # "sample" | "argmax"
            oracle_mode: str,             # "reward" | "metric_only" | "off"
            selector: str) -> list[Trajectory]
```

### 6.3 The firewall — enforce in code, not by discipline

Gold SQL is a **training-time-only** signal. If it leaks into the reader's prompt or the
state featuriser, all numbers are garbage.

```python
# in the test path
assert cfg.oracle_mode != "reward", "oracle rewards must not be active at eval"
# in featurise.py
assert not hasattr(state, "gold_sql") and not hasattr(state, "E_gold")
```

`oracle_mode` semantics:
- `"reward"` — oracle feeds `rewards.py`. Training only.
- `"metric_only"` — oracle computes join-path accuracy / per-hop recall for logging.
  Never touches the policy. Eval.
- `"off"` — no oracle at all.

---

## 7. The six processes

| # | Process | Entry point | Runtime | Purpose |
|---|---|---|---|---|
| 0 | Data preparation | `data_prep.py` | 2–4 h, once | materialise DBs, schema graphs, embedding cache, splits, synthetic train corpus |
| 1 | Oracle extraction | `oracle.py` | 30 min, once | gold SQL -> per-hop gold entity sets + `oracle_traj.jsonl` |
| 2 | Behaviour cloning | `bc_train.py` | ~1 h | supervised init + strongest baseline. No env, no reader, no reward. |
| 3 | PPO training | `ppo.py` | hours | the main run |
| 4 | Baseline evaluation | `evaluate.py --selector {truncate,cot,mtr,heuristic,bc,bc_beam}` | hours | comparison points |
| 5 | Policy evaluation | `evaluate.py --selector policy --ckpt ...` | ~1 h | argmax, oracle_mode=metric_only, no updates |
| 6 | Oracle ceiling | `evaluate.py --selector oracle` | 1 afternoon | the number that makes the results table defensible |

### 7.1 Process 3 inner loop (batch of N=64 questions)

```
COLLECT  (rollout.py, sample, oracle_mode="reward")
  for hop t = 0..H-1, all live episodes in lockstep:
      featurise(s_t)                  [cache lookup, batched]
      env.candidate_actions(s_t)      [schema graph walk]
      policy forward -> logits over k [batched MLP]
      sample a_t; record logπ_old(a_t), V_φ(s_t)
      env.step(a_t)                   [SQL join]
      oracle -> Φ(s_t), r_shape_t, r_fk_t   [stored, NOT visible to policy]
  at terminal, ALL 64 at once:
      serialise sub-tables -> ONE batched vLLM call   <-- the bottleneck
      EM vs gold -> terminal reward; token count -> cost

ADVANTAGE
  rewards.assemble(); GAE(λ=0.95, γ=1) -> A_t, returns

UPDATE  (4 epochs × minibatches over the SAME 64 trajectories)
  ratio = exp(logπ_θ − logπ_old)
  L = −min(ratio·A, clip(ratio,0.8,1.2)·A) + c1·(V_φ−ret)² − c2·H[π]
  backward; clip_grad_norm; Adam

LOG  mean return, EM, KL(π‖π_BC), entropy, value explained-var, tokens/episode
every 10 iters: evaluate.py on a 500-question dev slice
```

**Batch the reader call.** One vLLM call for 64 terminal states, not 64 calls. This is
the difference between a 2-hour and a 20-hour iteration budget.

**Cache reader outputs.** Hash `(subtable, question)` and cache at temperature 0. Early
in training the policy repeats itself constantly; the hit rate is large and free.

### 7.2 What each process protects against

| Missing process | Consequence |
|---|---|
| 0 | every step re-encodes from scratch; nothing fits on one GPU |
| 1 | one bit of feedback per attempt; never learns a 3-hop route |
| 2 | PPO starts from uniform over 20 actions with terminal reward — never gets off the ground |
| 3 | only imitation; cannot beat the gold routes |
| 4 | a number with nothing to compare it to |
| 5 | reported score inflated by training-time exploration noise |
| 6 | a weak reader caps the result and you cannot demonstrate that's what happened |

---

## 8. Datasets, metrics, and verification gates

### 8.1 Datasets

| Role | Dataset | Notes |
|---|---|---|
| **Eval (primary)** | MMQA (ICLR 2025) | 2591 2-table + 721 3-table **instances**; ~710 unique tables. Gold SQL, PK/FK, gold answers. |
| **Train** | Spider-derived synthetic corpus | generated by us — see `TASK_01_DATASET.md`. Target 20k–50k instances. |
| Transfer | HybridQA, BIRD dev | Week 10, zero-shot |
| Fallback | BIRD (12,751 pairs, 95 large DBs) | Plan B only — no multi-hop framing, no MMQA baselines |

**Correction to commonly-cited numbers:** MMQA's paper labels 2591/721 as "Total
Tables" but its own text calls them *samples*. A later paper (CORE-T) that pooled MMQA
reports **710 unique tables, 3313 questions**. Use "3,312 instances over ~710 tables".

**MMQA has no official split.** Split **by database**, not by question — otherwise the
same schema appears on both sides and generalisation numbers are meaningless.

### 8.2 Metrics

- **Answer:** EM (primary), F1, Partial Match. MMQA's PM uses GPT-4-turbo as judge;
  their audit found Pearson correlation 0.885 with human check for o1-preview
  (0.827 GPT-4, 0.778 GPT-3.5). Publish the judge model + prompt and hand-audit 100
  samples.
- **Process:** join-path exact accuracy, per-hop entity recall/precision, FK-selection
  accuracy (directly comparable to MMQA Tables 5/6).
- **Efficiency:** reader tokens/question, #LLM calls, latency.
- **RL diagnostics:** return curves, KL to init, policy entropy, value explained
  variance, sample efficiency.
- **Robustness:** EM vs #hops (2 vs 3), EM vs table length (500/700/900/1100 buckets),
  EM vs #distractor tables, EM with column headers ablated to `col_0..col_n`.

### 8.3 Models

| Role | Choice | Notes |
|---|---|---|
| Reader (frozen, in-loop) | `Qwen2.5-7B-Instruct` | default; vLLM; 40 GB bf16 or 24 GB quantised |
| Reader (debug, Weeks 5–6) | `Qwen2.5-1.5B-Instruct` | 5× faster iteration while fixing PPO bugs |
| Reader (comparability) | `TableLlama-7b`, `Mistral-7b` | one run each — MMQA reports 7.58 / 5.36 EM zero-shot on 2-table, so this shows our harness lands in the published ballpark |
| Encoder (frozen, cached) | `bge-m3` or `gte-large` | used once in Process 0, then never |
| Frontier API | GPT-5 / Claude / Gemini | ONE eval row only. Cannot afford in-loop. |

**Pick one reader and freeze it for the whole project.** Every number in the results
table must share the same reader.

**The policy is not an LLM.** It is a 3M-param MLP pair.

### 8.4 Verification gates — run before building on top

| id | Gate | Check | Fail response |
|---|---|---|---|
| G-1 | Availability | MMQA data link (`anonymous.4open.science/r/MMQA-34B1`) live? **Mirror it day 1.** | Email Jian Wu (Institute of Science Tokyo). Else regenerate a proxy eval set from Spider *dev* DBs. |
| G-2 | Row counts | median AND mean rows per table | If median is small, reframe motivation around join-space size, not context length |
| G-3 | Path yield | join paths of length 2–3 per DB | If low, add BIRD databases |
| G-4 | Template source | does MultiTabQA's released corpus cover our join shapes? | write templates for gaps only |
| G-5 | Round-trip pass rate | % of generated instances surviving execution verification | <40% → improve prompt or swap generator |
| G-6 | Oracle coverage | % of hops where some candidate action reaches `E_t^gold` | <85% → widen action space (see §8.5) |
| G-7 | Leakage | EM gap between original and paraphrase-hardened splits | large gap → the header-ablation diagnostic becomes a headline experiment |
| G-8 | PPO sanity | can PPO overfit 20 questions to return 1.0? | if not, the bug is in the code, not the hyperparameters |

### 8.5 Coverage: why it breaks even when gold SQL is correct

Coverage is about **representability**, not correctness.

- **Aggregation.** MMQA has 889+289 Numerical and 200+42 Count questions.
  `SELECT AVG(salary)` cannot be expressed as "carry these IDs".
  **THE FIX:** define `E_t^gold` from the **FROM/JOIN/WHERE clauses only, stripping
  SELECT**. The selector fetches the right *rows*; the reader does the arithmetic. This
  brings aggregation fully in scope and is the difference between training on ~1900 and
  ~3300 instances. **Get this right in Week 1.**
- **Cardinality.** If prefix 1 returns 400 IDs and we carry top-20, no action reaches
  the gold set. Add a `carry_all` action or accept the gap.
- **Non-equality predicates.** "latitude above 37.4" needs range filters in the action
  space.
- **Order-specificity.** Prefixes come from ONE join order. A legitimate alternative
  route has no gold set indexed at that hop. Not a bug — this is exactly why shaping is
  potential-based rather than a per-step target.
- **Structural:** self-joins (same table as T1 and T2 — one graph node), composite FKs
  (MMQA averages 2.81 FKs/table), INTERSECT/EXCEPT, correlated subqueries.
- **Genuine annotation error.** MMQA inter-human agreement is 86% (2-table) / 82%
  (3-table). A CIDR 2026 analysis found correcting annotation errors on 100 BIRD dev
  problems shifted absolute performance by up to 19 points and moved CHESS from 4th to
  1st (62% → 81%). Assume a few percent of "gold" is wrong.

**Coverage is a number to measure and report, not a problem to eliminate.**

### 8.6 The validity threat: schema-vocabulary leakage

**Read this before designing the featuriser.**

MMQA's questions were generated *from* SQL, so they leak schema vocabulary. The authors
tested it: manually paraphrasing 100 questions to remove column hints dropped o1-preview
from 43.5 → 40.7 EM (2-table) and 39.8 → 34.4 (3-table). Their example: the original
says "heads whose temporary acting value is 'Yes'" (hands you `temporary_acting`); the
paraphrase says "department heads who are acting now" (does not).

**Why this hits us harder than it hits an LLM.** Our action scorer is a dot product
between question embedding and column-name embedding. If the question literally contains
the column name, that dot product is doing string matching, not reasoning. The policy
could score well while learning nothing about schema structure — and Week 10 transfer
would be chance-level with no diagnosable cause.

**Two defences, both cheap:**

1. **Paraphrase-hardened eval split.** Generate a second question variant instructed to
   avoid schema vocabulary; verify it round-trips to the same result set. Report both.
   A few hundred instances suffices.
2. **Header-ablation diagnostic.** Replace every column header with `col_0, col_1, ...`
   and re-evaluate. Collapse ⇒ string matching. Graceful degradation ⇒ the policy is
   using cardinality, PK/FK structure, and value overlap. **One flag in the featuriser,
   and the most convincing robustness experiment available.**

Apply the same to the training corpus: sample the filter literal *after* choosing the
column, then paraphrase so the column name is absent but the value is present.

---

## 9. Related work — and what each paper is FOR

The three anchors triangulate a gap; they are not "the three best methods".

| Paper | Venue | Role |
|---|---|---|
| **MMQA** (Wu, Yang, Li, Ji, Okumura, Zhang) | ICLR 2025 **Oral** | **the problem.** Task, DBs, gold SQL, PK/FK, metric protocol, baseline numbers. The Oral status means the community already agreed the problem matters. Cite for the problem, never the solution. |
| **Chain-of-Table** (Wang et al., Google) | ICLR 2024 | **the formulation, and our foil.** Established table reasoning as sequential decision-making with a *frozen, unlearned* planner. Our contribution compresses to: same formulation, learned policy. |
| **StepSearch** (Zheng, An, Wang, Wang, Wu) | EMNLP 2025 Main (`2025.emnlp-main.1106`) | **the method we port.** PPO + step-wise rewards for multi-hop retrieval; open repo, published hyperparameters. Reported +11.2% / +4.2% absolute for 3B/7B over global-reward RL with only 19k training examples. Our delta: they *manufactured* step supervision; relational schemas give it free. |

Also required in the related-work section:

- **Arctic-Text2SQL-R1** (Snowflake) — closest methodological cousin. RL with
  execution-verified reward; 32B reached 71.83% BIRD execution accuracy, beating all
  open and proprietary models; the 7B beat DeepSeek-v3 and GPT-4o with 95× fewer
  parameters. Strongest evidence our family of method works. Cite prominently.
- **Table-R1 / TARPO** — closest *competitor*. Region selection within a **single**
  table, GRPO-based, decaying region rewards. Delimit explicitly: single vs multi-table,
  critic-free vs critic, decaying vs potential-based.
- **MTR** (inside the MMQA paper) — baseline (c). MTR(TableLlama-7b) reaches 68.3 F1
  top-2 on the 2-table subset.
- **MultiTabQA** (Pal et al., ACL 2023) — source of the 45 SQL templates MMQA used, and
  of a released 132,645-query multi-table corpus. Repo: `kolk/MultiTabQA`.
- **Beam Retrieval / S3HQA** — HybridQA lineage, for the transfer section.
- **CIDR 2026 annotation-error analysis** — for the evaluation-caveat paragraph and as
  defence when a reviewer questions unexplained failures.

### 9.1 Honest positioning

Multi-hop table reasoning as a field is crowded. **Our specific corner is not** — MMQA
is ~18 months old with little published work on it. But the corner is empty partly
because the current "solution" is "use a better model," and frontier models improve
monthly.

**Do not claim SOTA EM.** That race is lost to whatever ships next quarter. Claim:

> Comparable or better answer accuracy at 5–10% of the reader context, using a 3M-param
> learned head instead of a frontier reasoning model — plus a mechanistic account of
> where the gain comes from.

That survives the next model release.

**Framing warning.** If this is presented as "we apply PPO to table QA," a reviewer asks
why not GRPO and the project reads as incremental. Lead with the structural argument:
*relational schemas make dense, verifiable process rewards free, which is exactly the
resource StepSearch had to manufacture for text* — and the small constrained action
space is where a critic has a fighting chance.

---

## 10. Twelve-week plan

A = RL lead (policy, PPO, ablations). B = environment/eval lead.

| Wk | A | B | Gate |
|---|---|---|---|
| 1 | MDP spec doc (symbols fixed) | MMQA+Spider ingest, mirror data, G-1/G-2 | written MDP spec signed off |
| 2 | policy/value arch, featuriser | schema graph + action enumerator + reader harness | random policy runs 1000 episodes without crashing; log `|A(s)|` and episode-length distributions |
| 3 | oracle prefix extractor | baselines (a)–(d), full metric suite | **baselines reproduced** |
| 4 | behaviour cloning | reward oracle: Φ, r_fk, cost; **baseline (f) bc+beam** | BC beats heuristic (d); G-6 coverage reported |
| 5 | PPO, sparse reward only | distributed rollout + reward caching | **G-8: overfit 20 questions to return 1.0** |
| 6 | shaping + FK + cost, all behind flags | reward-hacking audit harness | shaped PPO ≥ BC on 2-table dev slice |
| 7 | **freeze v1, write methods section** | error taxonomy on 100 failures | interim report. **If shaped PPO < BC, STOP and debug — do not run ablations on a broken system.** |
| 8 | PPO vs GRPO vs REINFORCE, matched reader-call budget; 2×2 shaping grid | scoring-function ablation (dot vs MLP) | ablation table complete |
| 9 | 3-table subset; d sweep; hyperparameter sensitivity | table-length + distractor + **header-ablation** sweeps | robustness plots |
| 10 | HybridQA / BIRD transfer | **oracle ceiling (Process 6)** | generalisation section + headroom number |
| 11 | 3 seeds × all configs, paired bootstrap CIs | repo cleanup, README, one-command repro script | frozen results |
| 12 | paper writing | poster + 15-min talk | **final deliverables** |

Load-bearing gates: **Week 3** (baselines exist), **Week 5** (PPO can overfit 20
questions), **Week 7** (shaping beats BC). Weeks 7 and 12 are deliberately buffered —
something in 5–8 will overrun.

If Week 7 fails: drop transfer and any LoRA stretch goal; convert Weeks 8–11 into a
diagnostic study. With an oracle ceiling and a clean ablation grid that is still a
legitimate, publishable outcome.

---

## 11. Deliverables and success criteria

**Deliverables**

1. 8-page workshop-format paper.
2. Open-source repo: environment, PPO implementation, oracle extractor, repro scripts.
3. **The Spider-derived training corpus** with gold per-hop entity sets and
   paraphrase-hardened variants. *This is a reusable artefact independent of whether our
   method wins.* Treat it as a deliverable, not a workaround.
4. Results tables + cost-accuracy Pareto plots.
5. Poster and seminar talk.

**Success criteria — stated relative to OUR OWN matched-reader baselines.**

A frozen 7B reader will sit well below 50 EM in absolute terms. Comparing against
o1-preview's 50.78 is comparing two experiments that share nothing, and is the mistake
that will get the work shredded in Q&A.

- **Minimum:** environment + oracle + reproduced baselines released; shaped PPO ≥ BC;
  a clean 3-seed answer to the PPO-vs-GRPO question.
- **Target:** +5 EM over the strongest non-learned baseline at ≤50% reader tokens;
  +8 points join-path accuracy; gains hold on the 3-table subset.
- **Stretch:** gains transfer to HybridQA with no retraining; workshop acceptance
  (TRL @ NeurIPS, or an ACL/EMNLP workshop).

**Report headroom closed, not raw EM.** If ceiling = 62, PPO = 41, BC = 35, truncate =
19, the honest claim is "captured 6 of the 27 points available above BC." If the ceiling
had been 43, the reader was the bottleneck all along and no policy training would have
helped.

---

## 12. Risks and fallbacks

| Risk | Sev | Fallback |
|---|---|---|
| MMQA data link dead (anonymised host) | High | Mirror day 1. Email authors. Else our own generator produces a functional equivalent — which is why we build it rather than depend on the download. |
| Data scarcity — 3312 instances too few for on-policy RL | High | Already mitigated: generate 20k–50k from Spider train DBs (`TASK_01_DATASET.md`). If template regeneration is poor, train BC on Spider text-to-SQL trajectories and use RL only for fine-tuning. |
| Schema-vocabulary leakage inflates results | High | §8.6 — paraphrase-hardened split + header-ablation diagnostic. Convert the threat into a headline experiment. |
| PPO instability (small batches, sparse terminal reward) | High | The comparison IS a contribution: fall back to REINFORCE-with-baseline or GRPO and report that the critic does not pay off at H=4. Also γ=1, GAE λ=0.95, KL penalty to BC init, entropy floor. |
| Reward hacking — policy dumps the whole table | Med | Hard token budget enforced **in the environment**, not merely penalised. Audit selected-cell counts per episode; report the cost distribution, not the mean. |
| Reader ceiling masks policy gains | Med | Process 6 in Week 10; report fraction of headroom closed. Single most useful defensive number available. |
| PM metric is LLM-judged, non-reproducible | Med | EM as headline; publish judge model + prompt; hand-audit 100 samples mirroring MMQA's own protocol. |
| Action space too easy (MMQA gives only 2–3 candidate tables) | Med | Either run the open-corpus setting (pool all ~710 tables, as MMQA's own retrieval task and CORE-T do) or, at minimum, report the distractor-injection sweep. **State which up front** — otherwise a reviewer notes we are choosing between two options. |
| Compute constraints | Med | Design needs ONE 24–40 GB GPU: embeddings precomputed and cached, reader served once via vLLM, only ~3M params train. Any LoRA-reader work is explicitly a stretch goal. |
| Scooped by a concurrent paper | Med | Differentiator is the execution-derived shaping guarantee and the critic ablation, not the headline number. Both survive being scooped on raw performance. |
| Fully negative result | Low-Med | Reframe as a diagnostic study: "when does learned entity selection beat prompted planning over tables?" A well-instrumented negative with an oracle ceiling and a clean ablation grid is a legitimate seminar outcome. |
