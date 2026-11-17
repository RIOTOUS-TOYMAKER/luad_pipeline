from __future__ import annotations

from collections.abc import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .metadata import PatchSample


class PatchDataset(Dataset):
    """Dataset backed by a shared list of patch metadata and selected indices."""

    def __init__(
        self,
        samples: Sequence[PatchSample],
        indices: Sequence[int],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.indices = [int(index) for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample = self.samples[self.indices[item]]
        with Image.open(sample.path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label
