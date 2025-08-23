import torch
import einops
from jaxtyping import Float, Bool
import math


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    y = (x - x.max(dim=dim, keepdim=True).values).exp()
    return y / y.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[torch.Tensor, "b ... s d"],
    k: Float[torch.Tensor, "b ... s d"],
    v: Float[torch.Tensor, "b ... s d"],
    mask: Bool[torch.Tensor, "s s"],
) -> Float[torch.Tensor, "b ... s d"]:
    d_k = k.shape[-1]
    qk_scaled = einops.einsum(q, k, "... seq1 d, ... seq2 d -> ... seq1 seq2") / math.sqrt(d_k)
    qk_scaled[..., ~mask] = -math.inf
    weights = softmax(x=qk_scaled, dim=-1)
    return einops.einsum(weights, v, "... seq1 seq2, ... seq2 d -> ... seq1 d")
