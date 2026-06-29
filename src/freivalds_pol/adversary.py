"""Adversarial transcript transforms, one per cheat in the threat model.

Each takes an honest ``StepTranscript`` and returns a cheating one that the corresponding
verifier check should catch. The detection experiment uses these to measure detection rate
versus number of challenges.

  (a) lazy          -> caught by Freivalds (every output is wrong)
  (b) wrong_compute -> caught by Freivalds with prob ~ 1 - (1 - frac)^k
  (d) poison_shard  -> caught by the shard-root check, independent of k

  (c) free_ride (copying a peer's update) is a *cross-node* check -- detect duplicate
      updates / mismatched VRF-seed binding across submissions -- so it is exercised at the
      experiment level, not as a single-transcript transform.
"""

from __future__ import annotations

import copy

import numpy as np

from .transcript import StepTranscript


def lazy(t: StepTranscript, *, rng=None) -> StepTranscript:
    """(a) Never did the work: zero out the submitted update and all outputs."""
    t = copy.deepcopy(t)
    t.update = np.zeros_like(t.update)
    for rec in t.matmuls:
        rec.C = np.zeros_like(rec.C)
    return t


def wrong_compute(t: StepTranscript, *, rng=None, frac: float = 0.5) -> StepTranscript:
    """(b) Plausible-but-wrong: corrupt a random subset of matmul outputs."""
    rng = np.random.default_rng() if rng is None else rng
    t = copy.deepcopy(t)
    n_victims = max(1, int(frac * len(t.matmuls)))
    victims = rng.choice(len(t.matmuls), size=n_victims, replace=False)
    for i in victims:
        rec = t.matmuls[int(i)]
        rec.C = rec.C + rng.normal(scale=np.std(rec.C) + 1e-6, size=rec.C.shape)
    return t


def poison_shard(t: StepTranscript, *, rng=None) -> StepTranscript:
    """(d) Trained on the wrong shard: shard_root no longer matches the assignment."""
    t = copy.deepcopy(t)
    t.shard_root = bytes(len(t.shard_root))
    return t
