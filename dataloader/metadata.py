from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
CLASS_TO_LABEL = {"in_situ": 0, "invasion": 1}
PATIENT_PATTERN = re.compile(r"^p\d{2}$", flags=re.IGNORECASE)


class FilenameFormatError(ValueError):
    """Raised when a patch filename does not follow STAIN_pNN_... ."""


@dataclass(frozen=True)
class PatchSample:
    path: Path
    stain: str
    patient_id: str
    class_name: str
    label: int
    subtype: str

    @property
    def binary_stratum(self) -> str:
        return self.class_name

    @property
    def binary_subtype_stratum(self) -> str:
        # The binary class is included because subtype folder names can overlap.
        return f"{self.class_name}::{self.subtype}"


def parse_patch_filename(filename: str, expected_stain: str) -> tuple[str, str]:
    """
    Parse a filename whose first two underscore-delimited fields are:

        <stain>_<patient>_...

    The patient field is required to be exactly p followed by two digits.
    Examples: HE_p01_..., CK7+EVG_p59_...
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise FilenameFormatError(
            f"{filename!r}: expected at least three underscore-delimited fields."
        )

    stain, patient_id = parts[0], parts[1]

    if stain != expected_stain:
        raise FilenameFormatError(
            f"{filename!r}: stain field is {stain!r}, expected {expected_stain!r}."
        )

    if PATIENT_PATTERN.fullmatch(patient_id) is None:
        raise FilenameFormatError(
            f"{filename!r}: patient field must match pNN, for example p01."
        )

    return stain, patient_id.lower()


def scan_patch_samples(
    data_dir: Path,
    expected_stain: str,
    *,
    strict_filenames: bool = True,
) -> list[PatchSample]:
    """
    Scan this directory layout:

        <data_dir>/in_situ/<histologic_subtype>/*.png
        <data_dir>/invasion/<histologic_subtype>/*.png

    All samples from both binary classes are returned in one list so that a
    single patient-level split can be made across the entire experiment.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    samples: list[PatchSample] = []
    invalid: list[str] = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required class directory not found: {class_dir}"
            )

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative = image_path.relative_to(class_dir)
            if len(relative.parts) < 2:
                invalid.append(
                    f"{image_path}: image must be inside a subtype directory."
                )
                continue

            subtype = relative.parts[0]

            try:
                stain, patient_id = parse_patch_filename(
                    image_path.name,
                    expected_stain=expected_stain,
                )
            except FilenameFormatError as exc:
                invalid.append(str(exc))
                continue

            samples.append(
                PatchSample(
                    path=image_path,
                    stain=stain,
                    patient_id=patient_id,
                    class_name=class_name,
                    label=label,
                    subtype=subtype,
                )
            )

    if invalid:
        preview = "\n".join(f"  - {item}" for item in invalid[:20])
        remaining = len(invalid) - min(len(invalid), 20)
        suffix = f"\n  ... and {remaining} more" if remaining else ""
        message = (
            "Invalid patch filenames or locations were found:\n"
            f"{preview}{suffix}"
        )
        if strict_filenames:
            raise FilenameFormatError(
                f"{message}\n"
                "No split was created. Fix the names or use "
                "--no-strict-filenames only when intentionally excluding "
                "these files."
            )
        warnings.warn(
            f"{message}\nThese {len(invalid)} file(s) were excluded.",
            stacklevel=2,
        )

    if not samples:
        raise ValueError(f"No valid image files were found under {data_dir}.")

    return samples
