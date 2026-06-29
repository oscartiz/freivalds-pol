"""freivalds-pol: probabilistic + optimistic verification for decentralized training steps.

A research prototype of a verification layer for Psyche/DisTrO-style decentralized training:
catch nodes that submit incorrect gradient work at far less than full-recompute cost, using
Merkle commitments + random challenges + Freivalds' matmul check, with a path to a ZK
spot-check. See ``docs/DESIGN.md``.
"""

from .adaptive import fixed_cheat, nullspace_cheat
from .challenge import (
    Challenge,
    fiat_shamir_probes,
    fiat_shamir_rng,
    sample_challenges,
)
from .commitments import merkle_proof, merkle_root, verify_merkle_proof
from .freivalds import (
    freivalds_check,
    freivalds_check_threshold,
    freivalds_residual,
    freivalds_residual_left_with,
    freivalds_residual_with,
)
from .numerics import (
    calibrated_threshold,
    effective_gamma,
    honest_bound_inf,
    honest_bound_l2,
    node_matmul,
    to_bf16,
)
from .training import MLP, grad_check, make_task, step_transcript, training_step
from .transcript import MatMulRecord, StepTranscript
from .transformer import TransformerBlock, block_step
from .verifier import VerifyResult, verify_step

__all__ = [
    "Challenge",
    "sample_challenges",
    "fiat_shamir_rng",
    "fiat_shamir_probes",
    "nullspace_cheat",
    "fixed_cheat",
    "merkle_root",
    "merkle_proof",
    "verify_merkle_proof",
    "freivalds_check",
    "freivalds_check_threshold",
    "freivalds_residual",
    "freivalds_residual_with",
    "freivalds_residual_left_with",
    "calibrated_threshold",
    "effective_gamma",
    "honest_bound_inf",
    "honest_bound_l2",
    "node_matmul",
    "to_bf16",
    "MatMulRecord",
    "StepTranscript",
    "VerifyResult",
    "verify_step",
    "MLP",
    "make_task",
    "training_step",
    "step_transcript",
    "grad_check",
    "TransformerBlock",
    "block_step",
]
