import torch
import torch.nn as nn

from jaxtyping import Float, Int


def cross_entropy_loss(
    logits: Float[torch.Tensor, "... c"],
    targets: Int[torch.Tensor, "..."],
    dim: int = -1,
) -> Float[torch.Tensor, ""]:
    maxes = logits.max(dim=dim, keepdim=True).values
    logsumexp = (logits - maxes).exp().sum(dim=dim, keepdim=True).log().add(maxes)
    log_probs = -logits + logsumexp
    return log_probs.gather(dim=dim, index=targets.unsqueeze(dim)).mean()
