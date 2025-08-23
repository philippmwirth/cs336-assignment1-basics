from collections.abc import Sequence

from jaxtyping import Float, Int
import torch
import torch.nn as nn

from cs336_basics.transformer import modules


class Transformer(nn.Module):
    def __init__(
        self,
        token_embeddings: modules.Embedding,
        layers: Sequence[modules.TransformerBlock],
        ln_final: modules.RMSNorm,
        lm_head: modules.Linear,
    ):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.layers = nn.ModuleList(layers)
        self.ln_final = ln_final
        self.lm_head = lm_head

    def forward(self, x: Int[torch.Tensor, "b s"]) -> Float[torch.Tensor, "b s v"]:
        token_positions = torch.arange(x.shape[-1], dtype=torch.long, device=x.device)
        y = self.token_embeddings(x=x)
        for layer in self.layers:
            y = layer(x=y, token_positions=token_positions)
        return self.lm_head(self.ln_final(y))


def get_transformer(
    d_model: int,
    heads: int,
    d_ff: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    rope_theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    pre_norm: bool = False,
    nope: bool = False,
    gated: bool = False, # TODO switch back to True
) -> Transformer:
    token_embeddings = modules.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=device,
        dtype=dtype,
    )
    if nope:
        rope = modules.NoPE()
    else:
        rope = modules.RoPE(
            theta=rope_theta,
            d_k=d_model // heads,
            max_seq_len=context_length,
            dtype=dtype,
        )
    layers = [
        modules.TransformerBlock(
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            rope=rope,
            device=device,
            dtype=dtype,
            pre_norm=pre_norm,
            gated=gated,
        )
        for _ in range(num_layers)
    ]
    ln_final = modules.RMSNorm(
        d_model=d_model,
        device=device,
    )
    lm_head = modules.Linear(
        in_features=d_model,
        out_features=vocab_size,
        device=device,
        dtype=dtype,
    )
    return Transformer(
        token_embeddings=token_embeddings,
        layers=layers,
        ln_final=ln_final,
        lm_head=lm_head,
    ).to(device)

