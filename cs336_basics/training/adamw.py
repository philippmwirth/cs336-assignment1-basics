from typing import Collection, Mapping
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import torch.nn as nn
import math


def clip_gradients(params: Collection[nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clips gradients in place"""
    norm = 0.0
    for param in params:
        if param.grad is not None:
            norm += (param.grad**2).sum()
    norm = norm.sqrt()
    if norm >= max_l2_norm:
        for param in params:
            if param.grad is not None:
                param.grad *= max_l2_norm / (norm + eps)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Collection[Mapping[str, nn.Parameter]],  # TODO: accurate?
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                m = state.get("m", torch.zeros_like(param.grad))
                m = beta1 * m + (1.0 - beta1) * param.grad
                v = state.get("v", torch.zeros_like(param.grad))
                v = beta2 * v + (1 - beta2) * (param.grad**2)
                t = state.get("t", 0)
                t = t + 1
                adjusted_lr = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                param.data -= adjusted_lr * m / (v.sqrt() + eps)
                param.data -= lr * weight_decay * param.data
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss
