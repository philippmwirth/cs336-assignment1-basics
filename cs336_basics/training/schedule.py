import math


def lr_cosine_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    annealing_steps: int,
) -> float:
    if step < warmup_steps:
        return step * max_lr / warmup_steps
    if step < annealing_steps:
        cos = math.cos(math.pi * (step - warmup_steps) / (annealing_steps - warmup_steps))
        return min_lr + 0.5 * (1 + cos) * (max_lr - min_lr)
    return min_lr
