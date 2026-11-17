from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .settings import EXPECTED_CLASSES


class ImageFolderWithPaths(ImageFolder):
    """ImageFolder that also returns each sample's path and stable index."""

    def __getitem__(self, index: int) -> tuple[Any, int, str, int]:
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, int(label), str(path), int(index)


@dataclass(frozen=True)
class TestDataBundle:
    dataset: ImageFolderWithPaths
    loader: DataLoader
    classes: tuple[str, ...]


def build_test_data(
    test_dir: Path,
    *,
    transform,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> TestDataBundle:
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Patch test directory not found: {test_dir}")

    dataset = ImageFolderWithPaths(test_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No supported image files were found under {test_dir}.")

    classes = tuple(dataset.classes)
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            "The test directory must contain the two class folders in the "
            f"expected ImageFolder order {EXPECTED_CLASSES}; found {classes}. "
            "This check prevents silent inversion of the model's class indices."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    return TestDataBundle(dataset=dataset, loader=loader, classes=classes)
