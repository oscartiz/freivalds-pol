# CHANGES — from prototype to validated system

This log summarizes the hardening pass (Milestones M0–M5) that took `freivalds-pol` from a
single-block numpy prototype to a CI-green, scaled, and honestly-bounded system. Each milestone
is one commit; "claim movement" records what the work **confirmed**, **refined**, or
**overturned**.

## Net movement at a glance

| | before | after |
|---|---|---|
| tests | 55 | 80 (≈99% line coverage) |
| CI | hardcoded badge | live GitHub Actions (ruff + pytest, py3.10/3.12) |
| model | single block, single head, toy SGD | up to 4-layer / 8-head + AdamW |
| DeMo | 1D-tiled simplification | faithful 2D-chunk DeMo + conformance tests |
| soundness | "(1−f)^k + couldn't find an attack" | stated security definition + bound with marked gaps |
| multi-node | single-node only | free-riding detection + quantified grinding |
| ZK | design prose | working non-interactive sumcheck proof + PCS interface |
| headline backdoor claim | "no stealthy backdoor" | **overturned at scale** (loss-stealthy with AdamW) |

## M0 — Trust the harness
Live CI badge (workflow runs ruff + pytest on push); `pytest-cov` + `make coverage` (99%);
tests added for previously-uncovered claim-bearing code; determinism + reference environment
documented (REPORT §7).
*Claim movement:* the "tests pass / CI green" claims became machine-checked rather than asserted.

## M1 — Scale the experimental regime
`model.py`: multi-layer, multi-head transformer (grad-checked to ~2e-8) with an **AdamW** path;
generic flat Hessian-vector/power-iteration helpers; `experiments/scale.py`.
*Claim movement:*
- **CONFIRMED at scale:** sub-threshold drift stays sublinear (p≈0.21/0.28 vs naive 1);
  curvature-targeted attack has no edge (1.09× drift spread).
- **OVERTURNED at scale:** the §9 "no stealthy backdoor" claim was an artifact of toy SGD. With
  AdamW + depth the backdoor is **loss-stealthy** (~98% implant at <1.1× test loss); loss
  monitoring misses it, so per-step verification is necessary, not optional. (Every effective
  budget is ≫ the per-step floor, so per-step Freivalds still catches it.)

## M2 — Faithful DeMo wire format
`demo.py`: 2D per-chunk DCT + top-k + decayed (0.999) error feedback, matching reference DeMo
(bloc97/DeMo, arXiv:2411.19870) for 2D tensors; `tests/test_demo.py` pins conformance against
hand-computed known vectors (constant block → DC = v·c; one-hot coefficient → exact top-1).
*Claim movement:* **REFINED** — the "DisTrO wire format" claim now matches the real DeMo payload;
exact deltas (decay/k/chunk, 1D-vs-2D, sign-quantization-at-aggregation, ULP differences) are
tabulated (DESIGN §7b), and the per-block verifier provably transfers (any orthonormal transform).

## M3 — Formal soundness, collusion, grinding
DESIGN §10: security definition (completeness = 1; per-matmul soundness) and a Berry–Esseen
anti-concentration **soundness bound** for the two-sided commit-then-sample check, with gaps
marked. `collusion.py` + tests (free-riding/copying detection; identity binding).
`experiments/grinding.py` quantifies grinding work (`1/q_k`: ρ=3e-5 → 2.5/6.2/38 at k=1/2/4;
ρ≥1e-4 infeasible).
*Claim movement:* **REFINED** — soundness and multi-node resistance are now stated/quantified,
not asserted.

## M4 — ZK spot-check
`zk.py`: a sound non-interactive **sumcheck** argument for one GEMM over `F_p` (Fiat–Shamir),
with a `PolyCommitment` interface and an INSECURE `RevealCommitment` reference.
`experiments/zk_matmul.py` reports cost honestly.
*Claim movement:* **REFINED** — the ZK claim moved from prose to a working, sound prototype, with
the honest caveat that a real polynomial commitment (KZG/FRI) is still needed for hiding +
succinctness, and that ZK does not beat Freivalds on speed (its value is privacy + public
verifiability).

## M5 — Honest reporting
Rewrote the REPORT abstract to separate **validated** from **suggestive/scoped**; added a
**Threats to validity** subsection per major result; updated limitations (closed vs open); this
CHANGES.md.
*Claim movement:* confidence aligned to evidence throughout.

## Still open (honestly)
- A complete soundness proof (tight constant; rigorous at bf16).
- A real polynomial-commitment ZK opening (hiding + succinct).
- nanoGPT scale and a language objective; reproduce the AdamW backdoor on a richer objective.
- One fused verifier over a committed accumulator chain across rounds.
