# Verifiable Training Contributions for Psyche

*Working name: `freivalds-pol` — probabilistic + optimistic verification for DisTrO steps.*

## 0. Current state we are improving on

Psyche verifies a node's work via **redundant recompute-and-compare** across participants,
**Bloom filters** to confirm DisTrO results were gossiped, and **health checks** for
liveness; the Solana coordinator orders rounds, apportions data shards, and re-trains over
nodes that drop. Redundant recompute costs ~2× compute to verify 1× of work, scales poorly,
and offers no data privacy. Goal: detect cheating at **verifier cost ≪ recompute cost**.

## 1. Threat model

- **Coordinator** (Solana): public, semi-trusted; orders rounds, apportions shards/seeds.
- **Training nodes**: untrusted, mostly rational, some Byzantine.
- **Cheats to catch**
  - (a) **lazy** — submits zero/random update; never did the work.
  - (b) **wrong-compute** — plausible-but-wrong: wrong data, fewer steps, wrong LR.
  - (c) **free-ride** — copies a peer's update.
  - (d) **poison** — trained on the wrong/tampered shard.
- **Security goal:** a cheater is caught with probability ≥ 1−ε at verifier cost ≪ recompute
  cost, while honest nodes pay only small prover overhead.
- **Out of scope (v1):** >threshold witness/committee collusion; model-extraction attacks.

## 2. Building blocks

1. **Trajectory commitments** — node commits (Merkle/KZG) to `{θ_t root, shard root D_i,
   VRF seed s_i, activation-checkpoint root, update u_i}`. Cheap; published before challenge.
2. **Freivalds' check** — verify a claimed `C = A·B` via `A·(B·r) = C·r` for random `r`, in
   `O(n²)` not `O(n³)`. Spot-check the dominant GEMMs of a randomly chosen layer/microbatch.
   Kills (a) and (b) cheaply.
3. **Proof-of-Learning-style segment re-execution** — re-run `k` random micro-steps from
   committed checkpoints; mismatch ⇒ slash. Must resist known PoL *spoofing* attacks (bind
   via commitments + unpredictable VRF challenges).
4. **Data-shard binding** — node opens a Merkle path into its assigned shard and folds the
   shard root into the transcript hash ⇒ blocks (d), binds work to assigned data.
5. **Anti-free-ride** — bind `u_i` to a per-node VRF seed + checkpoint chain; duplicate
   updates across nodes flagged ⇒ blocks (c). (Cross-node check, not single-transcript.)
6. **ZK layer (phase 2)** — wrap the spot-check as a SNARK: prove "the Freivalds check on
   layer ℓ over committed θ_t, D_i passed" **without** publishing full activations (saves
   bandwidth) and optionally **without revealing the shard** (privacy). Prefer a
   sumcheck/GKR system (zkLLM / Lasso / Jolt-style) — natural for layered matmul circuits,
   far better than generic R1CS.

## 3. Protocol (optimistic + fraud-proof, rollup-shaped)

1. Coordinator apportions shard `D_i` + VRF seed `s_i` to node `i`.
2. Node computes DisTrO update `u_i`, posts commitment `Cmt_i`.
3. Updates accepted **optimistically** and aggregated.
4. **Challenge window:** a random committee (or anyone) picks `(layer ℓ, microbatch m,
   positions)`. The Freivalds **probe `r` is derived from `Cmt_i` plus a fresh public beacon
   drawn *after* the commit** (Fiat-Shamir + VRF) — never reused or predictable, or an
   adaptive node hides its cheat in `r`'s nullspace (see §6). Node answers with the Freivalds
   witness / opened checkpoints / (phase 2) a SNARK.
5. Verify in `O(n²) ≪ O(n³)`. On failure: slash stake, drop update, re-apportion shard
   (Psyche already re-trains over removed nodes).

Soundness: corrupting a fraction `f` of operations escapes a `k`-challenge audit with
probability ≈ `(1−f)^k`; tune `k` (and Freivalds rounds) to the target ε.

## 4. MVP (this repo)

- **Implemented:** Merkle commitments, Freivalds verifier, challenge sampler, the FP error
  model, the adaptive-adversary analysis, adversary suite, experiments, tests.
- **Real training steps.** `training.py` is a two-layer MLP (5 GEMMs); `transformer.py` is a
  Llama-style pre-norm block — RMSNorm, single-head causal attention, GELU MLP — emitting the
  **8 GEMMs** of a real step (`attn.Q/K/V`, the data-dependent `attn.scores = QKᵀ` and
  `attn.ctx = PV`, `attn.out`, `mlp.h`, `mlp.y`). Both hand-derive backprop validated against
  finite differences (`grad_check`: MLP ~4e-8, transformer ~9e-7). `experiments/real_step.py`
  runs the full verifier on a real block step: honest accepted; lazy / wrong-compute (caught
  inside attention) / poison-shard rejected.
- **The verifier now embodies the findings.** `verify_step` derives its probes from the
  commitment + beacon (Fiat-Shamir, §6), uses the calibrated per-record threshold from each
  matmul's claimed `dtype` (§5), runs a **two-sided** check (right + left probes, §6), and
  enforces a minimum precision (`min_dtype="fp32"`) on the challenged layer.
- **Compressed update** (`compressor.py`, §7): the DeMo wire format (momentum + per-tile DCT
  + top-k + error feedback) is implemented and verified per tile on a real transformer
  gradient.
- **Multi-round dynamics** (`trainer.py`, §8): a full DeMo training loop over many rounds with
  a budget-constrained adversary, plus a curvature probe (`curvature.py`: Hessian-vector products
  + power iteration) for the worst-case curvature-targeted adversary.
- **Next:** scale the single block to a multi-layer / multi-head transformer at nanoGPT scale
  (≈100M–1B); fuse the gradient (Freivalds) and compression (per-tile) checks into one
  `verify_step` over the committed accumulator chain across rounds.
- **Deliverable = the paper's core table:** detection probability vs #challenges; verifier
  cost vs full recompute; prover overhead (`experiments/run_detection.py`).

## 5. The FP crux (the real research)

Psyche's premise is **heterogeneous hardware**, so two honest GPUs do **not** produce
bit-identical results — exact recompute/Freivalds equality fails. The threshold separating
"honest numerical drift" from "cheating" must come from a floating-point error model.

**Error model** (`numerics.py`). Computing `C = A·B` in floating point gives
`C = A·B + E` with `|E| ≤ γ·(|A|·|B|)` elementwise, where `γ` is the effective unit
roundoff: `γ ≈ k·u` for naive accumulation, or `γ ≈ c·u_in` (c≈3–4) for a tensor core
(low-precision in, fp32 accumulate). For a Rademacher probe `r`, the rigorous worst-case
honest residual is `‖E r‖∞ ≤ γ·‖|A|·|B|‖∞`. Critically, that bound is computable in
**O(n²)** as `γ·max_i (|A|·(|B|·1))_i` — two matvecs — so it preserves Freivalds' advantage
over recompute. The tighter 2-norm (typical-case) bound is O(n³) and offline-only.

**Findings** (`experiments/fp_crux.py`, n=256, tensor-core model, verifier recomputes fp64):

| node dtype | honest noise floor | O(n²) rigorous bound | usable? (bound < signal) | min detectable cheat ρ\* (statistical) |
|---|---|---|---|---|
| fp32 | 3.7e-4 | 0.022 | yes (≪ signal 1.1e3) | **5e-7** |
| fp16 | 0.48 | 184 | yes | 7e-4 |
| bf16 | 4.1 | 1.5e3 | **no — exceeds signal** | 7e-3 |

1. The rigorous O(n²) bound never yields a false positive, but at **bf16 it is looser than
   the signal `‖C r‖`**, so it cannot catch *any* cheat. At **fp32 it sits ~5×10⁴ below the
   signal** and is tight enough to use directly.
2. A **statistical threshold** (mean + 8σ of honest residuals) is far tighter and its
   detection floor **ρ\* scales ~linearly with the unit roundoff** (≈1–8× of `u` across
   fp64→bf16) — but it carries a small false-positive / adversary-gaming risk.

**Protocol rule that follows:** require the *challenged* layer to be recomputable at
**≥ fp32** (submit it in fp32, or compute it with a deterministic kernel). Then the cheap,
rigorous O(n²) bound is usable and the undetectable band shrinks to ρ\* ~ 10⁻⁵–10⁻⁷ — well
below the scale of honest SGD/DisTrO-compression gradient noise, so a hideable cheat can't
do damage. Precision on the checked layer is the security knob.

Open sub-questions: verify DisTrO's **compressed** (DCT/top-k) update directly; tie ρ\* to a
measured gradient-noise scale to *prove* harmlessness; game theory of challenge-rate +
slashing to make cheating −EV; collusion resistance of the committee.

## 6. Adaptive adversary (probe timing is everything)

A node that *shapes* its cheat to the probe is the worst case. The decisive question is when
the probe is fixed relative to the cheat (`adaptive.py`, `experiments/adaptive.py`, n=256,
fp32 node, rigorous O(n²) threshold).

- **Predictable probe = total break.** If the probe `r` is reused, leaked, or otherwise
  known before the node commits, the adversary returns `C + Δ` with `Δ r = 0` — a rank-1
  cheat in the probe's nullspace. Measured: at impact `‖Δ‖/‖C‖ = 0.5`, **detection 0.000**,
  at *unbounded* impact. Freivalds offers zero security against a known probe.
- **Fix — commit-then-sample.** Derive the probe by Fiat-Shamir from the commitment (and,
  to stop grinding, a fresh public beacon / VRF drawn *after* the commit). Same 50% cheat is
  now caught **1.000** of the time. The probe must be unpredictable *and* posterior to the
  commitment; the repo's commit→challenge ordering already provides the hook.
- **Under fresh probes, adaptivity buys nothing beyond the FP band.** A fixed `Δ` meets a
  random `r`, so `Δr` is a `±` sum with std ~ `‖row of Δ‖₂` (anti-concentration). Detection
  rises geometrically with rounds `k` (e.g. at ρ=3e-5: 0.55 → 0.83 → 0.95 for k=1,2,4) and
  the only evadable cheats are those with impact below `ρ* ~ 2.7e-5` — exactly the FP band
  from §5. Grinding is futile: per-probe evasion `q ≈ 0` for any ρ ≥ 1e-4, so the expected
  `1/qᵏ` commitments to find an evading probe is astronomical; a beacon removes it entirely.
- **One real adversary edge, now closed.** A **rank-1** cheat evades slightly more often
  near the band (one anti-concentration draw per probe instead of many); placed in a *known*
  right probe's nullspace it evades the right check entirely. The fix — **two-sided** probing
  (`rᵀAB` vs `rᵀC` as well as `ABr` vs `Cr`) — is implemented and measured: on a rank-1
  nullspace cheat, one-sided detection 0.000 → two-sided 1.000 (`experiments/adaptive.py`).

**Net protocol requirements:** (i) probe drawn *after* the commitment via VRF/beacon;
(ii) `k` rounds large enough that grinding cost `1/qᵏ` dwarfs the slashing deterrent;
(iii) two-sided probing to close the low-rank edge.

## 7. Compressed-update verification (the DisTrO wire format)

A real node transmits not the dense gradient but a **DeMo-compressed** update: momentum
accumulation with error feedback, a per-tile **DCT**, and the **top-k** coefficients per tile
(`compressor.py`). It is verifiable cheaply because it decomposes:

1. **momentum** `m = decay·m_prev + g` — elementwise, O(n) to recompute from the
   Freivalds-verified gradient `g` and the committed prior accumulator;
2. **DCT** `coeff = C·m` — a *linear map*, so transmitted-value correctness is a (sparse)
   matmul check, Freivalds-amenable (the repo shows `freivalds_check(C, m, coeff)` passing);
3. **top-k** — the only nonlinear part; the verifier recomputes the per-tile DCT (O(tile²),
   or O(tile·log tile) by FFT) and confirms the indices are the k largest.

Verification is **per tile**, so a challenge over random tiles catches a node that corrupts a
fraction `f` with probability `1-(1-f)^c`. Measured on a real transformer gradient
(dim 131328 → 12.5% kept; 2052 tiles, k=8): the honest payload verifies, and each cheat —
lazy / fake-values / wrong-top-k / bad-residual — is caught with detection 0.25 → 0.89 as
`c` goes 1 → 8 (matching `1-(1-0.25)^c`), while the verifier inspects ~0.4% of tiles
(`experiments/compressed.py`).

This closes the loop: **the expensive gradient is verified probabilistically by Freivalds;
the cheap compression is verified by direct per-tile recompute; both are bound by commitments.**

## 8. Multi-round security: do sub-threshold cheats accumulate?

The deepest question. A node that cheats *below* the per-step detection threshold every round
is never caught — over a whole run, does its effect accumulate? DeMo's error feedback carries
the residual forward to preserve gradient information, so it also preserves an injected bias.
Naive worst case: `drift ≤ lr·Σ_t(budget·‖g_t‖)` — **linear in R**.

Tested in `trainer.py` / `experiments/multiround.py`: train the real transformer block for
R=300 rounds under DeMo compression three ways — honest, an *aligned* sub-threshold adversary
(fixed bias direction), a *random* one — with identical data and init, measuring drift
`‖θ − θ_honest‖`. Findings:

- **Drift grows sublinearly** (aligned p≈0.27, random p≈0.25) vs the naive bound's p=1 — a
  sub-threshold cheat does **not** accumulate as feared.
- **A directed bias has a real edge but no runaway:** it accumulates faster than random noise
  (the along-direction component grows with p≈0.65 > random's 0.25), so error feedback is *not*
  neutral — yet it stays **sublinear** (p<1), not free accumulation.
- **The loss is barely moved** (honest 0.4961 vs cheats 0.4958 / 0.4962). The drift that occurs
  lands mostly in flat, loss-irrelevant directions; the loss-relevant component is bounded.
- **Mechanism:** near a stable minimum the optimizer supplies a restoring force (off-minimum →
  larger true gradient → pushed back), settling a sub-threshold bias at a bounded equilibrium
  `~ budget/curvature` rather than letting it run away.

**Worst case tested — and not realized.** The natural fear is a *curvature-targeted* adversary
that aims the bias at the Hessian's flattest eigenvector (weakest restoring force). We built it
(`curvature.py`, `experiments/curvature_attack.py`): estimate the flat and steep eigenvectors
via Hessian-vector products (central difference of the gradient) + power iteration, then attack
along each — including a variant that **re-tracks** the flat direction as it moves. Result: with
curvature spanning ~4 orders of magnitude (−0.30 → +2.26), end drift varies <10% and held-out
test-loss harm is negligible (~6e-5) for *every* direction, tracked flat included. Targeting
low-curvature directions does **not** beat random: the drift is set by generic trajectory
sensitivity, and flat directions are flat precisely because the loss ignores them — so the
accumulation lands where it does no functional harm.

**Protocol rule.** Worst-case-safe design still keeps `budget < D/(lr·R·‖g‖)`: the detection
threshold tightens as ~1/R with run length — but this is now a *conservative margin*, not a
demonstrated necessity. An *above*-threshold cheat is caught with probability `1−(1−p)^R → 1`
within a few rounds. Caveats on the negative result: single small block, R=200. The
*targeted backdoor* objective — the attack that could evade the population-loss metric too — is
taken up directly in §9 (and also finds no stealthy attack).

## 9. Targeted backdoor: the one attack that could evade both defenses

The honest open question from §8: a *targeted backdoor* — a model normal everywhere but wrong on
a chosen trigger — could in principle slip under both per-step detection (sub-threshold) *and*
loss monitoring (population loss flat). We built it (`trainer.backdoor_attack`,
`experiments/backdoor.py`): each round inject a budget-limited dose of the gradient
`∇_θ‖f(x_trigger) − y_target‖²` (here, flip an off-distribution trigger's output sign),
recomputed at the current parameters. Sweeping the per-round budget — which *is* the
detectability knob:

| budget (of ‖g‖) | per-step? | backdoor implanted | test loss (× honest) |
|---|---|---|---|
| 1e-6 | stealthy | 0.0% | 1.0× |
| 1e-3 … 1e-1 | caught | ≤ 2.4% | 1.0× |
| 0.3 | caught | 7.7% | 1.1× |
| 1.0 | caught | 85% | 6.2× |
| 3.0 | caught | 97% | 4.8× |

**There is no stealthy-and-effective regime.** Meaningful implant (>10%) needs budget ≳ 0.3·‖g‖
— ~10⁴× the fp32 per-step detection floor `ρ*~1e-5` (caught by Freivalds) *and* it blows the
population test loss up several-fold (caught by loss monitoring). Stealthy budgets implant ~0%
with a flat test loss. Even on an off-distribution trigger, hijacking `f(x_trigger)` requires
weight changes large enough to wreck in-distribution behaviour — the two are entangled through
shared weights — so the attacker is loud on at least one detector. (Ablation: keeping *all* DCT
coefficients instead of top-k gives the same result, so DeMo compression is not what blocks it.)

At matched capacity, then, the layered scheme — per-step Freivalds / compression checks plus the
population-loss sanity any training run already watches — leaves no quiet corner for a backdoor.

**But over-parameterization erodes this** (`experiments/backdoor_capacity.py`). A capacity-rich
student fitting a low-rank task (teacher d_ff=8) has spare directions that are flat for the
training loss but functional for a trigger. Holding the budget fixed and growing the student
width, the implant climbs while the loss cost grows far slower:

| student d_ff | budget 0.1 | budget 0.3 |
|---|---|---|
| 8 (matched) | 2.9% / 1.00× | 10.5% / 1.03× |
| 64 | 3.9% / 1.00× | 13.2% / 1.04× |
| 256 | 6.1% / 1.01× | 20.9% / 1.11× |
| 1024 | 11.6% / 1.04× | 41.4% / 1.43× |

(cell = backdoor implanted % / test-loss ratio). At width 1024, budget 0.1, ~12% of the
backdoor lands for only a 1.04× loss bump — a genuinely loss-quiet partial backdoor. So the
**stealth window widens with capacity**: wider models take more backdoor per unit of loss
damage, precisely because they have more loss-flat-but-functional directions.

**This is the sharpest design consequence of the whole project:** per-step verification is *not*
a luxury for the large, capacity-rich models Psyche actually targets — it is exactly where loss
monitoring alone is weakest, so the per-step check carries the security that the loss metric
cannot. Remaining caveats: single block, one trigger/target, MSE objective, modest widths.

## 10. Prior art / positioning

- zkFL (gradient aggregation) · ZKML survey · VeriLLM (inference side).
- Proof-of-Learning (Jia et al.) and its spoofing attacks (Fang et al.).
- Niche: training-step verification for *real* decentralized runs is wide open.

## 11. Landing it with Nous

Build against the open Psyche repo; open a discussion framing the recompute-cost problem;
share MVP benchmarks; offer as an optional verification module. Parallel track: a paper.
