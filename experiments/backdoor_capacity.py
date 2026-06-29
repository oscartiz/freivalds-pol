"""Does over-parameterization open a stealthy backdoor? (the honest open question from §9)

Run:  python -m experiments.backdoor_capacity   (after `pip install -e .`)

In §9 a matched-capacity model had no stealthy-and-effective backdoor: implanting one wrecked
the population loss. The open worry was a capacity-rich model -- it has spare directions that
are flat for the (low-rank) training loss but functional for a trigger, so a backdoor might hide
there. We test it: a fixed low-rank teacher (d_ff=8), students of growing width fitting it, and
the same sub-threshold-style backdoor attack. The grid is implant% / test-loss-ratio per
(student width, budget).

Finding: over-parameterization WIDENS the stealth window -- wider students take more backdoor
per unit of loss damage -- so per-step verification is not a luxury for capacity-rich models;
it is where loss-monitoring alone is weakest.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.trainer import (
    backdoor_attack,
    block_from_flat,
    eval_loss,
    run_training,
    trigger_loss,
)
from freivalds_pol.transformer import forward

D, T, TEACHER_DFF = 64, 32, 8        # fixed low-rank task
R0, R = 60, 180
WIDTHS = [8, 64, 256, 1024]
BUDGETS = [0.1, 0.3]


def attack(student_dff, budget):
    cfg = dict(d=D, d_ff=student_dff, T=T, teacher_d_ff=TEACHER_DFF)
    ev = dict(d=D, d_ff=student_dff, T=T, teacher_d_ff=TEACHER_DFF)
    blk0 = block_from_flat(run_training(R0, lr=0.1, **cfg)["final"], D, student_dff)
    trng = np.random.default_rng(2024)
    x = 3.0 * trng.choice([-1.0, 1.0], size=(8, D))
    y_honest, _, _ = forward(blk0, x)
    y_trig = -y_honest

    honest = run_training(R0 + R, lr=0.1, **cfg)
    hb = block_from_flat(honest["final"], D, student_dff)
    rms_h = np.sqrt(trigger_loss(hb, x, y_trig))
    base = eval_loss(hb, **ev)

    adv, hook = backdoor_attack(R0, budget, x, y_trig)
    run = run_training(R0 + R, lr=0.1, adversary=adv, hook=hook, **cfg)
    fb = block_from_flat(run["final"], D, student_dff)
    implanted = 1.0 - np.sqrt(trigger_loss(fb, x, y_trig)) / rms_h
    test_ratio = eval_loss(fb, **ev) / base
    return implanted, test_ratio


def main():
    print(f"low-rank teacher d_ff={TEACHER_DFF}; students fit it then are attacked.")
    print("cells = backdoor implanted%% / test-loss ratio. Stealthy = high%% at ~1.0x.\n")
    head = "student d_ff".ljust(14) + "".join(f"budget {b}".rjust(18) for b in BUDGETS)
    print(head + "\n" + "-" * len(head))
    for w in WIDTHS:
        row = f"{w}".ljust(14)
        for b in BUDGETS:
            impl, tr = attack(w, b)
            row += f"{impl:6.1%} / {tr:4.2f}x".rjust(18)
        print(row)

    print("\nReading: down each column (rising width = more over-parameterization) the implant")
    print("grows while the loss ratio grows far slower -- a wider model gives MORE backdoor per")
    print("unit of loss damage, i.e. a wider stealth window. The earlier 'no stealthy backdoor'")
    print("(§9) held only at matched capacity; capacity erodes it. Consequence: per-step")
    print("verification is most necessary exactly for the large, capacity-rich models Psyche")
    print("targets -- loss monitoring alone is weakest there.")


if __name__ == "__main__":
    main()
