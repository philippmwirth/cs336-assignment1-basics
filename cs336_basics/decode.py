from collections.abc import Iterator, Sequence
from cs336_basics.transformer import transformer
from cs336_basics.transformer import utils
from cs336_basics.tokenizer import bpe
from cs336_basics.training import adamw
from cs336_basics.training import checkpoints

import argparse
from pathlib import Path
import torch
from jaxtyping import Int


def decode(
    model: transformer.Transformer,
    tokenizer: bpe.BPE,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    end_of_text_token="<|endoftext|>",
) -> Iterator[str]:
    # TODO: Raise if sequence length is too long?
    inputs = torch.LongTensor(tokenizer.encode(text=prompt)).to(device)
    n_tokens_generated = 0
    while n_tokens_generated < max_tokens:
        next_token = _decode_one(
            model=model,
            inputs=inputs,
            temperature=temperature,
            top_p=top_p,
        )
        inputs = torch.cat([inputs, next_token])
        generated_string = tokenizer.decode([int(i.detach().cpu().numpy()) for i in inputs])
        yield generated_string
        n_tokens_generated += 1
        if generated_string.endswith(end_of_text_token):
            break


@torch.no_grad()
def _decode_one(
    model: transformer.Transformer,
    inputs: Int[torch.Tensor, "s"],
    temperature: float,
    top_p: float,
) -> Int[torch.Tensor, "one"]:
    logits = model(x=inputs) / temperature
    softmax = utils.softmax(x=logits, dim=-1)[-1]
    argsort = torch.argsort(softmax, dim=-1, descending=True)
    sum_p = 0.0
    i = 0
    candidate_probabilities = []
    while sum_p <= top_p:
        candidate_probabilities.append(softmax[argsort[i]])
        sum_p += candidate_probabilities[-1]
        i += 1
    candidate_probs_tensor = torch.stack(candidate_probabilities)
    winner = torch.multinomial(candidate_probs_tensor, 1)
    return argsort[winner.item()].unsqueeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode using Transformer model")

    # Model Arguments
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of the model's embeddings and hidden states."
    )
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the feed-forward network.")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Size of the vocabulary.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="Theta parameter for RoPE.")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"], help="Model dtype."
    )
    parser.add_argument("--sequence_length", type=int, default=256, help="Length of input sequences (context length).")

    # Optimizer Arguments
    parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate for AdamW.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW beta coefficients.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (L2 penalty).")
    parser.add_argument("--device", type=str, default="mps", help="Device to run model on (e.g., 'cuda', 'cpu').")

    # Inference/Decoding Specific Arguments
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to a trained model checkpoint.")
    parser.add_argument("--vocabulary_path", type=Path, required=True, help="Path to BPE vocabulary file.")
    parser.add_argument("--merges_path", type=Path, required=True, help="Path to BPE merges file.")
    parser.add_argument(
        "--special_tokens",
        nargs="*",  # 0 or more arguments
        default=["<|endoftext|>"],
        help="List of strings to add to the vocabulary. E.g., '<|endoftext|> <|pad|>'.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to generate text from.")
    parser.add_argument("--max_tokens", type=int, default=250, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold.")

    args = parser.parse_args()

    model = transformer.get_transformer(
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.sequence_length,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=torch.float32,
    )
    optimizer = adamw.AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )
    checkpoints.load_checkpoint(
        src=args.checkpoint_path,
        model=model,
        optimizer=optimizer,
    )
    tokenizer = bpe.BPE.from_files(
        vocab_filepath=args.vocabulary_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_tokens,
    )
    print(args.prompt)
    for output in decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    ):
        print(output)  # , end="\r")
