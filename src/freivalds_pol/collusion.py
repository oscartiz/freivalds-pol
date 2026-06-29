"""Multi-node threats: free-riding and collusion.

The Freivalds + commit-then-sample scheme verifies each node against *math*, not against peer
witnesses, so colluding provers cannot vouch for each other's wrong work the way they could in a
recompute-and-witness scheme — the probe is fixed by the commitment + public beacon, not by
peers. Two multi-node vectors remain:

  * **free-riding / update-copying** — a lazy node resubmits a peer's update (or a colluding
    group submits one shared update) to collect rewards without doing the work. Detected here by
    grouping identical submitted updates across nodes (`detect_free_riders`). Honest nodes train
    on different shards/seeds, so their updates differ; an exact match across node ids is the
    signature of copying.
  * **beacon grinding** — quantified separately in ``experiments/grinding.py``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .commitments import hash_array


def detect_free_riders(transcripts, *, decimals: int = 6) -> list[list[str]]:
    """Return groups of node ids that submitted the *same* update (copying / collusion).

    Updates are compared after rounding to ``decimals`` places so that bit-level noise does not
    hide an otherwise-identical copy. A returned group has length >= 2.
    """
    by_update: dict[bytes, list[str]] = defaultdict(list)
    for t in transcripts:
        key = hash_array(np.round(np.asarray(t.update, dtype=np.float64), decimals))
        by_update[key].append(t.node_id)
    return [ids for ids in by_update.values() if len(ids) > 1]


def identity_bound(transcript) -> bool:
    """True iff the transcript's commitment actually binds its claimed node id and seed.

    A copier that resubmits a peer's transcript verbatim under its own identity would have to
    change ``node_id``/``seed``, which changes the Merkle commitment — so a replayed commitment
    cannot be re-attributed. This checks that the commitment depends on (node_id, seed).
    """
    import copy
    base = transcript.commitment()
    t2 = copy.deepcopy(transcript)
    t2.node_id = transcript.node_id + "_other"
    t3 = copy.deepcopy(transcript)
    t3.seed = transcript.seed + 1
    return t2.commitment() != base and t3.commitment() != base
