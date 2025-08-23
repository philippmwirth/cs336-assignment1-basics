import torch
from jaxtyping import Int
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[torch.Tensor, "b s"], Int[torch.Tensor, "b s"]]:
    n_start_indices = len(dataset) - context_length
    indices = np.concatenate(
        [np.arange(i, i + context_length)[None, :] for i in np.random.randint(0, n_start_indices, size=batch_size)],
        axis=0,
    )
    data = torch.from_numpy(dataset[indices])
    targets = torch.from_numpy(dataset[indices + 1])
    return data.long().to(device), targets.long().to(device)
