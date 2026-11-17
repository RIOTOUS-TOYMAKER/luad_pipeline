from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PatchDataset
from .metadata import PatchSample
from .split import SplitIndices


@dataclass(frozen=True)
class DataLoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: PatchDataset
    val_dataset: PatchDataset


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    samples: Sequence[PatchSample],
    split: SplitIndices,
    *,
    train_transform,
    val_transform,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoaderBundle:
    train_dataset = PatchDataset(
        samples=samples,
        indices=split.train_indices,
        transform=train_transform,
    )
    val_dataset = PatchDataset(
        samples=samples,
        indices=split.val_indices,
        transform=val_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )

    return DataLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
