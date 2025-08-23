import dataclasses
import torch
from pathlib import Path
import torch.nn as nn
from collections import deque, OrderedDict


_MODEL_KEY = "model"
_OPTIMIZER_KEY = "optimizer"
_STEP_KEY = "step"


@dataclasses.dataclass(frozen=True)
class CheckpointSerializable:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    step: int

    def serialize(self) -> dict[str, OrderedDict[str, torch.Tensor] | int]:
        return {
            _MODEL_KEY: self.model.state_dict(),
            _OPTIMIZER_KEY: self.optimizer.state_dict(),
            _STEP_KEY: self.step,
        }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    out: Path,
) -> None:
    serializable = CheckpointSerializable(
        model=model,
        optimizer=optimizer,
        step=step,
    )
    torch.save(serializable.serialize(), out)


def load_checkpoint(
    src: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state_dicts = torch.load(src, weights_only=False)
    if _MODEL_KEY not in state_dicts:
        raise KeyError(f"Key '{_MODEL_KEY}' not in state dicts.")
    if _OPTIMIZER_KEY not in state_dicts:
        raise KeyError(f"Key '{_OPTIMIZER_KEY}' not in state dicts.")
    if _STEP_KEY not in state_dicts:
        raise KeyError(f"Key '{_STEP_KEY}' not in state dicts.")
    new_model_state_dict = {}
    for key, value in state_dicts[_MODEL_KEY].items():
        if key.startswith("_orig_mod."):
            new_model_state_dict[key[10:]] = value
        else:
            new_model_state_dict[key] = value
    model.load_state_dict(new_model_state_dict)
    optimizer.load_state_dict(state_dicts[_OPTIMIZER_KEY])
    return state_dicts[_STEP_KEY]


class Checkpointer:
    def __init__(
        self,
        save_dir: Path,
        keep_last_n: int = 10,
    ):
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.last_n_checkpoints = deque()
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> None:
        save_path = self.save_dir / f"step_{step:08d}.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            out=save_path,
        )
        self.last_n_checkpoints.append(save_path)
        if len(self.last_n_checkpoints) > self.keep_last_n:
            self.last_n_checkpoints.popleft().unlink()
