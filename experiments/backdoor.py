"""The targeted-backdoor adversary: the one attack that could evade BOTH defenses at once.

Run:  python -m experiments.backdoor   (after `pip install -e .`)

A sub-threshold cheat is never caught per step (§6/§7) and barely moves the population loss
(§8) -- but does it stay harmless under a *targeted* objective? Here the adversary injects, each
round, a budget-limited dose of the gradient that drives a chosen TRIGGER input toward a target
output (here: flip the trigger output's sign). It wants the model to behave normally everywhere
(population/test loss unchanged) yet wrongly on the trigger.

We sweep the per-round budget and report, for each: how much of the backdoor is implanted vs.
the honest baseline, and the collateral population test-loss harm. The budget is the same knob
as detectability -- a smaller budget is harder to detect -- so the table is the stealth-vs-harm
tradeoff, and the decisive question is whether a *stealthy* (small) budget can still plant the
backdoor while leaving the test loss flat.
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

R0, R = 40, 200
D, D_FF, T = 64, 128, 32
BUDGETS = [1e-6, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0]
RHO_STAR = 1e-5        # representative per-step detection floor at fp32 (see fp_crux.py)


def main():
    warm = run_training(R0, lr=0.1)
    blk0 = block_from_flat(warm["final"], D, D_FF)

    # Trigger input + target: an OFF-DISTRIBUTION trigger (a +/-3 pattern; training inputs are
    # N(0,1), so the population loss never sees it -> the backdoor direction is genuinely flat
    # for the population loss and the attack gets its best, most dangerous shot). Target: flip
    # the sign of the model's trigger output.
    trng = np.random.default_rng(2024)
    x_trig = 3.0 * trng.choice([-1.0, 1.0], size=(T, D))
    y_honest, _, _ = forward(blk0, x_trig)
    y_trig = -y_honest

    honest = run_training(R0 + R, lr=0.1)
    hb = block_from_flat(honest["final"], D, D_FF)
    base_test = eval_loss(hb)
    rms_honest = np.sqrt(trigger_loss(hb, x_trig, y_trig))   # how close honest gets to target

    print(f"warmup R0={R0}, attack R={R}; OFF-distribution trigger, target = -f(x_trigger)\n")
    print(f"honest baseline: test loss = {base_test:.4f}, "
          f"trigger RMS-to-target = {rms_honest:.4f} (far = no backdoor)")
    print(f"per-step detection floor rho* ~ {RHO_STAR:.0e} of ||g|| (fp32); a budget above it "
          "is caught per step.\n")
    print("budget".ljust(9) + "per-step?".rjust(11) + "implanted".rjust(12)
          + "test loss".rjust(12) + "(x honest)".rjust(12))
    print("-" * 56)
    for b in BUDGETS:
        adv, hook = backdoor_attack(R0, b, x_trig, y_trig)
        run = run_training(R0 + R, lr=0.1, adversary=adv, hook=hook)
        fb = block_from_flat(run["final"], D, D_FF)
        implanted = 1.0 - np.sqrt(trigger_loss(fb, x_trig, y_trig)) / rms_honest
        test = eval_loss(fb)
        detect = "CAUGHT" if b > RHO_STAR else "stealthy"
        print(f"{b:<9.0e}{detect:>11}{implanted:12.1%}{test:12.4f}{test / base_test:11.1f}x")

    print("\nReading: there is NO stealthy-and-effective regime. Meaningful implant (>10%) needs")
    print("budget >= ~0.3 of ||g|| -- which is ~10^4x the per-step detection floor (caught by")
    print("Freivalds) AND blows up the population test loss several-fold (caught by loss")
    print("monitoring). Stealthy budgets implant ~0% with a flat test loss. Implanting a backdoor")
    print("on an off-distribution trigger still requires weight changes large enough to wreck")
    print("in-distribution behaviour -- so the attacker is loud on at least one detector. (Cause")
    print("is not DeMo top-k: keeping all coefficients gives the same result.)")


if __name__ == "__main__":
    main()
