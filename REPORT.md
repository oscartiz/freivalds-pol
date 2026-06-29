# Cheap Verification of Decentralized Training Steps

**A probabilistic + optimistic scheme for Nous Research's Psyche / DisTrO, with a
floating-point, adaptive-adversary, and multi-round security analysis.**

*Oscar Tiznado — CIMAT. Research prototype (`freivalds-pol`).*

---

## Abstract

Decentralized training networks such as [Nous Research](https://nousresearch.com)'s
[Psyche](https://psyche.network) let untrusted nodes contribute gradient work and reward them
for it, which creates an incentive to cheat. Psyche today verifies work by **redundant
recompute-and-compare**, costing ~2× the compute to verify 1× of work. We design and measure a
verifier that catches cheating at **far less than recompute cost**: Merkle commitments + random
challenges + **Freivalds' probabilistic matmul check** (O(n²) vs O(n³)), a **floating-point
error model** that makes the acceptance threshold sound on heterogeneous hardware, a
**commit-then-sample** (Fiat–Shamir) probe with **two-sided** checks that closes adaptive
attacks, and a **per-tile** verifier for DisTrO's compressed (DeMo) wire format. We then ask
the questions that decide whether the scheme is sound for a *training run*, not just one step:
do never-detected **sub-threshold** cheats accumulate (no — sublinearly, loss unharmed); does a
**curvature-targeted** adversary beat random (no edge); and can a **targeted backdoor** evade
both per-step detection and loss monitoring? The answer is the sharpest result here: at matched
capacity, no — but **over-parameterization widens the stealth window**, so per-step verification
is *most necessary* exactly for the large models Psyche targets. Everything runs end-to-end on a
real, gradient-checked transformer block; 55 tests; all figures reproducible.

---

## 1. Motivation

Psyche coordinates training on Solana and rewards nodes for gradient work computed with the
DisTrO/DeMo optimizers. Its current integrity mechanism is redundancy: results are recomputed
and compared, with Bloom filters confirming gossip and health checks for liveness. Redundancy
is expensive and offers no data privacy. The question this project answers: **can we verify a
node's training step probabilistically, for a small fraction of recompute cost, and does that
verification actually hold up over a full run against an adversary who knows the scheme?**

The angle is a natural fit for zero-knowledge / verifiable-computation tooling: the heavy object
(the gradient) is a chain of matrix products, and matrix products admit a classical cheap
probabilistic check.

## 2. Threat model

- **Coordinator** (Solana): public, semi-trusted; orders rounds, apportions data shards + seeds.
- **Nodes**: untrusted, mostly rational, some Byzantine.
- **Cheats:** (a) lazy — never did the work; (b) wrong-compute — plausible but wrong; (c)
  free-ride — copy a peer; (d) poison — train on the wrong shard; and the adaptive/long-run
  variants studied in §6–§9.
- **Goal:** catch a cheater with probability ≥ 1−ε at verifier cost ≪ recompute, honest nodes
  paying only small overhead.

## 3. The scheme

A step is recorded as a transcript of its matmuls (the GEMMs of the forward + backward pass),
plus bindings (model-state root, shard root, per-node seed, submitted update), all committed by
a Merkle root. The verifier challenges a random subset and, per challenged matmul `C = A·B`:

1. **Freivalds' check** — verify `A(Br) = Cr` for a random probe `r`: O(n²), not O(n³).
2. **Calibrated threshold** — accept iff the residual is within a bound derived from a
   floating-point error model (§4.2), so honest heterogeneous-hardware drift is not flagged.
3. **Commit-then-sample, two-sided** — the probe is derived from the commitment + a public
   beacon (Fiat–Shamir), and checked on both sides, defeating adaptive probe-targeting (§4.3).

The DisTrO **compressed** update (momentum + DCT + top-k + error feedback) is verified per tile
(§4.4). Soundness everywhere follows the spot-check law: corrupting a fraction `f` escapes a
`k`-challenge audit with probability `(1−f)^k`.

## 4. Results

All results use a real two-layer transformer block (RMSNorm, single-head causal attention, GELU
MLP) whose backprop is validated against finite differences (max rel. err ~9e-7), so the
matmuls being verified are a genuinely correct step.

### 4.1 Gradient verification: cheap and sound

Detection of a wrong-compute cheat follows `1−(1−f)^k`; at `k=8` challenges the verifier does
**~0.5% of recompute work**. Lazy / poison / free-ride cheats are caught at `k=1`. This is the
core efficiency claim: probabilistic verification replaces 2× redundancy with sub-1% spot-checks.

### 4.2 The floating-point crux

On heterogeneous hardware an honest `C ≠ A·B` exactly, so the threshold needs a numerical model:
`C = A·B + E`, `|E| ≤ γ·(|A||B|)`. The worst-case residual bound `γ·‖|A||B|‖∞` is computable in
**O(n²)** (two matvecs), preserving Freivalds' advantage. Measured: the bound is usable (sits
below the signal) only at **≥ fp32** precision on the checked layer; at bf16 it exceeds the
signal and catches nothing. A statistical threshold is tighter, and its smallest detectable
cheat `ρ*` **scales linearly with the unit roundoff** across five orders of magnitude — so
precision on the challenged layer is the security knob.

![FP crux](figures/fp_crux.png)

### 4.3 Adaptive adversary

If the probe is predictable, a rank-1 cheat in its nullspace evades with certainty at unbounded
impact — Freivalds is then worthless. Deriving the probe from the commitment + a fresh beacon
(**commit-then-sample**) fixes it: the same 50%-impact cheat goes from **0% to 100% detection**.
A residual rank-1 edge is closed by **two-sided** probing (one-sided 0.000 → two-sided 1.000),
and grinding the commitment for a favorable probe is infeasible for any meaningful cheat.

![Adaptive](figures/adaptive.png)

### 4.4 The compressed update (DisTrO wire format)

A node transmits not the dense gradient but a DeMo-compressed update (~12.5% of the data here).
It decomposes into an elementwise momentum step (O(n) recompute), a **linear DCT** (matmul check,
Freivalds-amenable), and a **top-k** selection (cheap per-tile DCT recompute). Verifying per tile,
each cheat — lazy / fake-values / wrong-top-k / bad-residual — is caught with the spot-check law,
while the verifier inspects a tiny fraction of tiles.

![Compressed update](figures/compressed.png)

### 4.5 Multi-round: do sub-threshold cheats accumulate?

A node that cheats just under the threshold every round is never caught. The naive fear is
linear accumulation via error feedback. Measured over a real run: parameter drift grows
**sublinearly** (exponent ≈0.27 vs the linear bound's 1.0) and the loss is barely moved — the
optimizer's restoring force settles a sub-threshold bias at a bounded equilibrium. A *directed*
bias does have a real (but still sublinear) edge over random noise.

![Multi-round drift](figures/multiround.png)

### 4.6 Curvature-targeted worst case

The natural worst case aims the bias at the loss Hessian's **flattest** direction (weakest
restoring force), found via Hessian-vector products + power iteration. It gives **no edge**:
curvature spans ~4 orders of magnitude while drift varies <10% and test-loss harm stays
negligible, even when the flat direction is re-tracked as it moves. Flat directions are flat
*because the loss ignores them* — so the accumulation lands where it does no functional harm.

![Curvature attack](figures/curvature.png)

### 4.7 Targeted backdoor — and the capacity crack

The one attack that could evade per-step detection *and* loss monitoring: implant a trigger
(normal everywhere, wrong on a chosen off-distribution input). At **matched capacity there is no
stealthy-and-effective regime** — a real backdoor needs a budget ~10⁴× the per-step floor that
also blows the population loss up several-fold (left). But **over-parameterization erodes this**
(right): a wide student fitting a low-rank task has spare loss-flat-but-functional directions, so
the implant climbs with width while the loss cost grows far slower — at width 1024, budget 0.1,
~12% of the backdoor lands for only a 1.04× loss bump.

![Backdoor tradeoff](figures/backdoor.png)
![Backdoor vs capacity](figures/backdoor_capacity.png)

**This is the sharpest design consequence of the project:** per-step verification is *not* a
luxury for the large, capacity-rich models Psyche targets — it is exactly where loss monitoring
alone is weakest, so the per-step check carries the security the loss metric cannot.

## 5. Limitations

- Single transformer block, single head, numpy/CPU, modest widths and run lengths.
- DeMo is a faithful *simplification* (whole-tensor tiling, per-tile top-k); not byte-identical
  to Nous's implementation. The phase-2 zero-knowledge spot-check is designed (§ design doc) but
  not implemented.
- The backdoor study uses one trigger/target and an MSE objective; larger models and richer
  objectives are the natural next probes.

## 6. Related work

Freivalds' algorithm (1977); Proof-of-Learning (Jia et al.) and its spoofing attacks (Fang et
al.); zkFL gradient aggregation; the ZK-verifiable-ML survey; VeriLLM (verifiable decentralized
*inference*). The niche — cheap, FP-sound, adaptive-and-multi-round-tested verification of
*training* steps for real decentralized runs — is largely open.

## 7. Reproducibility

```bash
pip install -e ".[dev,viz]"
make test            # 59 tests
make coverage        # 99% line coverage on the library
make figures         # regenerate every figure in figures/
make experiments     # rerun every experiment script
```

Every number above comes from a script under `experiments/`; every figure from
`experiments/figures.py`. Design details and derivations are in [`docs/DESIGN.md`](docs/DESIGN.md).

**Determinism.** Every experiment seeds its NumPy `Generator` explicitly (e.g.
`default_rng(0)`), and every Monte-Carlo inner loop is seeded by its iteration index, so a run
reproduces exactly on a fixed environment. *Caveat:* bit-for-bit identical figures across
machines also require the same NumPy/BLAS build and thread count, because BLAS may reorder
floating-point matmul accumulation; set `OMP_NUM_THREADS=1` for the strictest reproducibility.
The qualitative results (detection rates, drift exponents, the capacity trend) are stable across
environments. Reference environment for the numbers and figures here: Python 3.14, NumPy 2.x,
single-threaded BLAS, CPU.

**Test coverage.** 59 tests, **99%** line coverage of `src/freivalds_pol` (`make coverage`).
The few uncovered lines are defensive zero-norm guards, the single-leaf Merkle edge case, and
the `two_sided=False` verifier branch; the offline `l2` bound, the naive-accumulation γ regime,
and `freivalds_check_threshold` are now tested. CI (`.github/workflows/ci.yml`) runs ruff +
pytest on Python 3.10 and 3.12 on every push.

## 8. Conclusion

Probabilistic verification can replace redundant recompute for decentralized training at sub-1%
cost, *if* the threshold is grounded in a floating-point model and the probe is committed before
it is revealed. Across an FP analysis, an adaptive adversary, the compressed wire format, and
multi-round dynamics, the scheme holds — and the one place it weakens, over-parameterized
backdoors, is precisely the place it is most needed. Four of the project's own hypotheses were
overturned by its own measurements; each correction sharpened the result.
