import numpy as np

from freivalds_pol.trainer import (
    backdoor_attack,
    block_from_flat,
    eval_loss,
    run_training,
    trigger_loss,
)
from freivalds_pol.transformer import forward

D, D_FF, T = 16, 32, 8
CFG = dict(d=D, d_ff=D_FF, T=T)


def _trigger(blk):
    rng = np.random.default_rng(2024)
    x = 3.0 * rng.choice([-1.0, 1.0], size=(4, D))   # off-distribution trigger
    y_honest, _, _ = forward(blk, x)
    return x, -y_honest                               # target: flip the sign


def test_trigger_loss_zero_at_target():
    blk = block_from_flat(run_training(5, lr=0.1, **CFG)["final"], D, D_FF)
    yhat, _, _ = forward(blk, 3.0 * np.ones((4, D)))
    assert trigger_loss(blk, 3.0 * np.ones((4, D)), yhat) < 1e-12


def test_large_budget_implants_but_harms_population():
    R0, R = 10, 70
    blk0 = block_from_flat(run_training(R0, lr=0.1, **CFG)["final"], D, D_FF)
    x, y = _trigger(blk0)
    honest = run_training(R0 + R, lr=0.1, **CFG)
    hb = block_from_flat(honest["final"], D, D_FF)
    adv, hook = backdoor_attack(R0, 1.0, x, y)
    run = run_training(R0 + R, lr=0.1, adversary=adv, hook=hook, **CFG)
    ab = block_from_flat(run["final"], D, D_FF)
    # the backdoor is implanted ...
    assert trigger_loss(ab, x, y) < 0.5 * trigger_loss(hb, x, y)
    # ... but it wrecks the population loss (no stealthy free lunch)
    assert eval_loss(ab, **CFG) > 1.5 * eval_loss(hb, **CFG)


def test_stealthy_budget_neither_implants_nor_harms():
    R0, R = 10, 70
    blk0 = block_from_flat(run_training(R0, lr=0.1, **CFG)["final"], D, D_FF)
    x, y = _trigger(blk0)
    honest = run_training(R0 + R, lr=0.1, **CFG)
    hb = block_from_flat(honest["final"], D, D_FF)
    adv, hook = backdoor_attack(R0, 1e-2, x, y)
    run = run_training(R0 + R, lr=0.1, adversary=adv, hook=hook, **CFG)
    ab = block_from_flat(run["final"], D, D_FF)
    # a stealthy budget barely moves the trigger and barely touches the population loss
    assert trigger_loss(ab, x, y) > 0.9 * trigger_loss(hb, x, y)
    assert abs(eval_loss(ab, **CFG) - eval_loss(hb, **CFG)) < 0.1 * eval_loss(hb, **CFG)


def test_overparameterization_widens_backdoor_window():
    # A wider student fitting the same low-rank task absorbs more backdoor at the same budget.
    def implant(student_dff):
        cfg = dict(d=16, d_ff=student_dff, T=8, teacher_d_ff=4)
        blk0 = block_from_flat(run_training(20, lr=0.1, **cfg)["final"], 16, student_dff)
        rng = np.random.default_rng(2024)
        x = 3.0 * rng.choice([-1.0, 1.0], size=(4, 16))
        y_honest, _, _ = forward(blk0, x)
        y = -y_honest
        hb = block_from_flat(run_training(100, lr=0.1, **cfg)["final"], 16, student_dff)
        rms_h = np.sqrt(trigger_loss(hb, x, y))
        adv, hook = backdoor_attack(20, 0.3, x, y)
        fb = block_from_flat(
            run_training(100, lr=0.1, adversary=adv, hook=hook, **cfg)["final"], 16, student_dff)
        return 1.0 - np.sqrt(trigger_loss(fb, x, y)) / rms_h

    assert implant(128) > implant(8)
