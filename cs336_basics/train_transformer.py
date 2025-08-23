import dataclasses
import datetime
from torch.utils import tensorboard
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import numpy.typing as npt
import argparse
import logging
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from cs336_basics.training import adamw
from cs336_basics.training import checkpoints
from cs336_basics.training import cross_entropy_loss
from cs336_basics.training import data
from cs336_basics.training import schedule
from cs336_basics.transformer import transformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


@dataclasses.dataclass(slots=True, frozen=True)
class TrainConfig:
    max_lr: float
    min_lr: float
    warmup_steps: int
    max_grad_norm: float
    train_steps: int
    batch_size: int
    sequence_length: int
    device: str


@dataclasses.dataclass(slots=True, frozen=True)
class EvalConfig:
    eval_steps: int
    eval_every_n: int
    batch_size: int
    sequence_length: int
    device: str


@torch.no_grad()
def evaluation_step(
    model: nn.Module,
    eval_config: EvalConfig,
    eval_dataset: npt.NDArray,
) -> float:
    model.eval()
    inputs, targets = data.get_batch(
        dataset=eval_dataset,
        batch_size=eval_config.batch_size,
        context_length=eval_config.sequence_length,
        device=eval_config.device,
    )
    logits = model(inputs)
    loss = cross_entropy_loss.cross_entropy_loss(
        logits=logits,
        targets=targets,
    )
    model.train()
    return loss.detach().cpu().float().numpy()


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tensorboard_logger: tensorboard.SummaryWriter,
    train_config: TrainConfig,
    train_dataset: npt.NDArray,
    step: int,
) -> float:
    lr = schedule.lr_cosine_schedule(
        step=step,
        max_lr=train_config.max_lr,
        min_lr=train_config.min_lr,
        warmup_steps=train_config.warmup_steps,
        annealing_steps=train_config.train_steps,
    )
    for param_group in optimizer.param_groups:
        # Scaling the lr would be more flexible.
        param_group["lr"] = lr
    tensorboard_logger.add_scalar("lr/train", lr, step)
    inputs, targets = data.get_batch(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        context_length=train_config.sequence_length,
        device=train_config.device,
    )
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy_loss.cross_entropy_loss(logits=logits, targets=targets)
    # Cast loss to float32 for backward
    loss = loss.float()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = param.grad.data.to(torch.float32)
    adamw.clip_gradients(
        params=model.parameters(),
        max_l2_norm=train_config.max_grad_norm,
    )
    norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            norm += (param.grad**2).sum().cpu().float().numpy()
    tensorboard_logger.add_scalar("grad_norm/train", norm, step)
    optimizer.step()
    return loss.detach().cpu().float().numpy()


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tensorboard_logger: tensorboard.SummaryWriter,
    checkpointer: checkpoints.Checkpointer,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    train_dataset: npt.NDArray,
    eval_dataset: npt.NDArray,
    train_step: int,
) -> None:
    for train_step in range(train_step, train_config.train_steps):
        if train_step % eval_config.eval_every_n == 0:
            mean_loss = 0.0
            for eval_step in range(eval_config.eval_steps):
                mean_loss += (
                    evaluation_step(
                        model=model,
                        eval_config=eval_config,
                        eval_dataset=eval_dataset,
                    )
                    / eval_config.eval_steps
                )
            logger.info(f"Eval step {train_step:06d}/{train_config.train_steps:06d} | Loss/eval: {mean_loss:2f}")
            tensorboard_logger.add_scalar("Loss/eval", mean_loss, train_step)
            checkpointer.save(
                model=model,
                optimizer=optimizer,
                step=train_step,
            )
        train_loss = training_step(
            model=model,
            optimizer=optimizer,
            tensorboard_logger=tensorboard_logger,
            train_config=train_config,
            train_dataset=train_dataset,
            step=train_step,
        )
        logger.info(f"Train step {train_step:06d}/{train_config.train_steps:06d} | Loss/train: {train_loss:2f}")
        tensorboard_logger.add_scalar("Loss/train", train_loss, train_step)
    mean_loss = 0.0
    for eval_step in range(eval_config.eval_steps):
        mean_loss += (
            evaluation_step(
                model=model,
                eval_config=eval_config,
                eval_dataset=eval_dataset,
            )
            / eval_config.eval_steps
        )
    tensorboard_logger.add_scalar("Loss/eval", mean_loss, train_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model.")

    # Model Arguments
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of the model's embeddings and hidden states."
    )
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the feed-forward network.")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Size of the vocabulary.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="Theta parameter for RoPE (Rotary Positional Embeddings)."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for model parameters (e.g., float32, bfloat16).",
    )

    # Optimizer Arguments
    parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate for the AdamW optimizer.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for the AdamW optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps with cosine annealing.")
    parser.add_argument(
        "--betas", type=float, nargs=2, default=(0.9, 0.95), help="Beta coefficients for AdamW (beta1, beta2)."
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (L2 penalty) for AdamW.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")

    # Training Configuration
    parser.add_argument("--train_steps", type=int, default=5000, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--sequence_length", type=int, default=256, help="Length of input sequences (context length).")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run the model on (e.g., 'cuda', 'cpu').",
    )

    # Evaluation Configuration
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of steps to run during each evaluation.")
    parser.add_argument("--eval_every_n", type=int, default=100, help="Perform evaluation every N training steps.")

    # Logging and Checkpointing Paths
    parser.add_argument("--log_dir", type=Path, default="outputs/logs", help="Directory for TensorBoard logs.")
    parser.add_argument(
        "--checkpoint_dir", type=Path, default="outputs/checkpoints", help="Directory for saving model checkpoints."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=Path, default=None, help="If set, resumes training from checkpoint."
    )

    # Dataset Paths
    parser.add_argument(
        "--training_dataset", type=Path, required=True, help="Path to the memory-mapped training dataset (.bin file)."
    )
    parser.add_argument(
        "--eval_dataset", type=Path, required=True, help="Path to the memory-mapped evaluation dataset (.bin file)."
    )

    args = parser.parse_args()

    print(len(np.load(args.training_dataset, mmap_mode="r")))
    import sys
    sys.exit(0)
    match args.dtype:
        case "float32":
            model_dtype = torch.float32
        case "bfloat16":
            model_dtype = torch.bfloat16
        case "float16":
            model_dtype = torch.float16
        case _:
            raise ValueError(f"Unsupported dtype: {args.dtype}")

    model = transformer.get_transformer(
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.sequence_length,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=model_dtype,
    )
    model = torch.compile(model, backend="aot_eager")
    optimizer = adamw.AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )
    train_step = 0
    if args.resume_from_checkpoint is not None:
        train_step = checkpoints.load_checkpoint(
            src=args.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
        )
    train(
        model=model,
        optimizer=optimizer,
        tensorboard_logger=tensorboard.SummaryWriter(log_dir=args.log_dir / str(datetime.datetime.now())),
        checkpointer=checkpoints.Checkpointer(save_dir=args.checkpoint_dir),
        train_config=TrainConfig(
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=args.device,
        ),
        eval_config=EvalConfig(
            eval_steps=args.eval_steps,
            eval_every_n=args.eval_every_n,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=args.device,
        ),
        train_dataset=np.load(args.training_dataset, mmap_mode="r"),
        eval_dataset=np.load(args.eval_dataset, mmap_mode="r"),
        train_step=train_step,
    )
