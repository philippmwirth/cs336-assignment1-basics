import math

import einops
from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn

from cs336_basics.transformer import utils


FLOAT_32_DTYPE = torch.float32


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(data=torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        self._reset_parameters()

    def forward(
        self,
        x: Int[torch.Tensor, "b s"],
    ) -> Float[torch.Tensor, "b s d"]:
        return self.weight[x]

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self._reset_parameters()

    def forward(
        self,
        x: Float[torch.Tensor, "s"],
    ) -> Float[torch.Tensor, "s n"]:
        return einops.einsum(self.weight, x, "out_dim in_dim, ... in_dim -> ... out_dim")

    def _reset_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.d_model, dtype=FLOAT_32_DTYPE, device=device))
        self._reset_parameters()

    def forward(
        self,
        x: Float[torch.Tensor, "... d"],
    ) -> Float[torch.Tensor, "... d"]:
        in_dtype = x.dtype
        x = x.to(FLOAT_32_DTYPE)
        rms = (self.eps + x.pow(2).sum(axis=-1) / self.d_model).sqrt()
        result = (x / rms.unsqueeze(-1)) * self.weight
        return result.to(in_dtype)

    def _reset_parameters(self) -> None:
        self.weight.data = torch.ones(self.d_model)


# NoPE implementation.
class NoPE(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.identity = nn.Identity(args, kwargs)

    def forward(
        self,
        x: Float[torch.Tensor, "... s d"],
        token_positions: Int[torch.Tensor, "... s"],
    ) -> Float[torch.Tensor, "... s d"]:
        return self.identity(x)


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.register_buffer(
            name="rotation_matrices",
            tensor=self._get_rotation_matrices(
                theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device, dtype=dtype
            ),
            persistent=False,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... s d"],
        token_positions: Int[torch.Tensor, "... s"],
    ) -> Float[torch.Tensor, "... s d"]:
        rotation_matrices = self.rotation_matrices[token_positions]
        y = einops.rearrange(x, "... s (blocks two) -> ... s blocks two", two=2)
        y = einops.einsum(rotation_matrices, y, "... s blocks i j, ... s blocks j -> ... s blocks i")
        return einops.rearrange(y, "... s blocks two -> ... s (blocks two)")

    def _get_rotation_matrices(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Float[torch.Tensor, "i k two two"]:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=dtype, device=device) / d_k))
        t = torch.arange(max_seq_len, dtype=dtype, device=device)
        freqs = einops.einsum(t, inv_freq, "i, j -> i j")
        cos_vals = freqs.cos()
        sin_vals = freqs.sin()
        rot_matrix_components = torch.stack(
            [
                cos_vals,
                -sin_vals,
                sin_vals,
                cos_vals,
            ],
            dim=0,
        )
        return einops.rearrange(rot_matrix_components, "(a b) i k -> i k a b", a=2, b=2).contiguous()


class SiLU(nn.Module):
    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        return torch.sigmoid(x) * x


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        gated: bool = True,
    ):
        super().__init__()
        self.gated = gated
        if self.gated:
            self.w3 = Linear(d_model, d_ff, dtype=dtype, device=device)
        else:
            d_ff = (d_ff * 3) // 2
        self.w1 = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff, d_model, dtype=dtype, device=device)
        self.silu = SiLU()

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        assert not self.gated, "Remove me after the experiments!"
        if self.gated:
            return self.w2(self.silu(self.w1(x)) * self.w3(x))
        else:
            return self.w2(self.silu(self.w1(x)))


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        heads: int,
        d_model: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_k = self.d_v = d_model // heads
        self.q_proj = Linear(d_model, heads * self.d_k, dtype=dtype, device=device)
        self.k_proj = Linear(d_model, heads * self.d_k, dtype=dtype, device=device)
        self.v_proj = Linear(d_model, heads * self.d_v, dtype=dtype, device=device)
        self.output_proj = Linear(heads * self.d_v, d_model, dtype=dtype, device=device)
        self.rope = rope

    def forward(
        self,
        x: Float[torch.Tensor, "... s d"],
        token_positions: Int[torch.Tensor, "... s"] | None = None,
    ) -> Float[torch.Tensor, "... s d"]:
        q = einops.rearrange(self.q_proj(x), "... s (h d) -> ... h s d", d=self.d_k)
        k = einops.rearrange(self.k_proj(x), "... s (h d) -> ... h s d", d=self.d_k)
        v = einops.rearrange(self.v_proj(x), "... s (h d) -> ... h s d", d=self.d_v)
        if token_positions is not None and self.rope is not None:
            q = self.rope(x=q, token_positions=token_positions)
            k = self.rope(x=k, token_positions=token_positions)
        seq_len = q.shape[-2]
        o = einops.rearrange(
            utils.scaled_dot_product_attention(q=q, k=k, v=v, mask=self._get_causal_attention_mask(seq_len=seq_len)),
            "... h s d -> ... s (h d)",
        )
        return self.output_proj(o)

    def _get_causal_attention_mask(self, seq_len: int) -> Bool[torch.Tensor, "s s"]:
        return ~torch.triu(torch.ones((seq_len, seq_len), dtype=torch.long), diagonal=1).bool()


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        d_ff: int,
        rope: RoPE,
        pre_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        gated: bool = True,
    ):
        super().__init__()
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
        )
        self.attn = CausalMultiHeadSelfAttention(
            heads=heads,
            d_model=d_model,
            rope=rope,
            dtype=dtype,
            device=device,
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            dtype=dtype,
            device=device,
            gated=gated,
        )
        self.pre_norm = pre_norm

    def forward(
        self,
        x: Float[torch.Tensor, "... s d"],
        token_positions: Int[torch.Tensor, "... s"] | None = None,
    ) -> Float[torch.Tensor, "... s d"]:
        if self.pre_norm:
            y = self.ln1(x=x)
            y = x + self.attn(x=y, token_positions=token_positions)
            z = self.ln2(x=y)
            return y + self.ffn(x=z)
        else:
            z = self.ln1(x + self.attn(x=x, token_positions=token_positions))
            return self.ln2(z + self.ffn(x=z))
