"""Step verifier: given a transcript and a set of challenges, accept or reject.

This is the protocol verifier, and it bakes in the three research findings:

  1. the opened transcript must match the published commitment (no after-the-fact edits);
  2. the data-shard root must match the coordinator's assignment (blocks shard poisoning);
  3. each challenged matmul must pass a **two-sided** Freivalds check (right + left probes,
     closing the rank-1 adaptive edge) against a **calibrated threshold** derived from the
     node's claimed precision (the FP-crux bound), using probes **derived from the commitment
     plus a public beacon** (Fiat-Shamir, so an adaptive node cannot target the probe).

A single failed check rejects the whole step, which on Psyche maps to: slash stake, discard
the update, and re-apportion the shard (the network already re-trains over removed nodes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .challenge import Challenge, fiat_shamir_rng
from .freivalds import freivalds_residual_left_with, freivalds_residual_with
from .numerics import UNIT_ROUNDOFF, calibrated_threshold
from .transcript import StepTranscript

# Precisions ranked coarse -> fine; the protocol can require a minimum on the checked layer.
_PRECISION_ORDER = ["bf16", "fp16", "fp32", "fp64"]


def _probes(commitment: bytes, beacon: bytes, name: str, m: int, n: int, rounds: int):
    """Right probe (n, rounds) and left probe (rounds, m), both bound to the commitment."""
    rng = fiat_shamir_rng(b"probe", commitment, beacon, name.encode())
    R = rng.choice(np.array([-1.0, 1.0]), size=(n, rounds))
    L = rng.choice(np.array([-1.0, 1.0]), size=(rounds, m))
    return R, L


@dataclass
class VerifyResult:
    accepted: bool
    reason: str = ""
    checks_run: int = 0


def verify_step(
    transcript: StepTranscript,
    challenges: list[Challenge],
    expected_commitment: bytes,
    *,
    expected_shard_root: bytes | None = None,
    beacon: bytes = b"",
    safety: float = 8.0,
    two_sided: bool = True,
    min_dtype: str | None = "fp32",
) -> VerifyResult:
    if transcript.commitment() != expected_commitment:
        return VerifyResult(False, "commitment mismatch", 0)

    if expected_shard_root is not None and transcript.shard_root != expected_shard_root:
        return VerifyResult(False, "shard root mismatch", 0)

    for i, ch in enumerate(challenges):
        rec = transcript.matmuls[ch.matmul_index]

        if min_dtype is not None and (
            rec.dtype not in UNIT_ROUNDOFF
            or _PRECISION_ORDER.index(rec.dtype) < _PRECISION_ORDER.index(min_dtype)
        ):
            return VerifyResult(False, f"'{rec.name}' below min precision {min_dtype}", i + 1)

        m, n = rec.C.shape
        R, L = _probes(expected_commitment, beacon, rec.name, m, n, ch.freivalds_rounds)

        tau_r = calibrated_threshold(rec.A, rec.B, rec.dtype, mode="inf", side="right",
                                     safety=safety)
        if freivalds_residual_with(rec.A, rec.B, rec.C, R).max() > tau_r:
            return VerifyResult(False, f"freivalds(right) failed on '{rec.name}'", i + 1)

        if two_sided:
            tau_l = calibrated_threshold(rec.A, rec.B, rec.dtype, mode="inf", side="left",
                                         safety=safety)
            if freivalds_residual_left_with(rec.A, rec.B, rec.C, L).max() > tau_l:
                return VerifyResult(False, f"freivalds(left) failed on '{rec.name}'", i + 1)

    return VerifyResult(True, "ok", len(challenges))
