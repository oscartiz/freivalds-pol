"""Data model for a single training-step transcript.

A transcript records the matmul operations a node *claims* it performed for one DisTrO
step, plus the bindings (model-state root, data-shard root, per-node RNG seed, submitted
update) that tie the work to the assigned data and node identity. The ``commitment`` is the
Merkle root the node publishes before the challenge window opens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .commitments import hash_array, merkle_root


@dataclass
class MatMulRecord:
    """One claimed matrix product ``C = A @ B`` inside the step (e.g. a layer's GEMM).

    ``dtype`` is the precision the node claims it computed the product in; the verifier uses
    it to pick the calibrated Freivalds threshold (see ``numerics.calibrated_threshold``).
    """

    name: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    dtype: str = "fp32"


@dataclass
class StepTranscript:
    node_id: str
    seed: int                 # per-node VRF seed for this round (binds work to identity)
    theta_root: bytes         # commitment to model state theta_t
    shard_root: bytes         # commitment to the assigned data shard
    update: np.ndarray        # the DisTrO update u_i the node submits
    matmuls: list[MatMulRecord] = field(default_factory=list)

    def leaves(self) -> list[bytes]:
        """Canonical leaf list committed by the Merkle root."""
        out = [
            b"node:" + self.node_id.encode(),
            b"seed:" + int(self.seed).to_bytes(8, "big", signed=False),
            b"theta:" + self.theta_root,
            b"shard:" + self.shard_root,
            b"update:" + hash_array(self.update),
        ]
        for r in self.matmuls:
            out.append(b"mm:" + r.name.encode() + b":" + r.dtype.encode()
                       + b":" + hash_array(r.C))
        return out

    def commitment(self) -> bytes:
        return merkle_root(self.leaves())
