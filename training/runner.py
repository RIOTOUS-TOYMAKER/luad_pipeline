from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from architectures import architecture_display_name
from config import ExperimentConfig
from dataloader import (
    build_dataloaders,
    resolve_split,
    save_patient_split,
    save_sample_manifest,
    save_split_summary,
    scan_patch_samples,
)
from preprocess import build_transforms
from .callbacks import BestMetricsAtBestValLoss
from .model import LitClassifier, build_image_classifier
from .results import (
    append_training_result,
    save_best_network_state_dict,
    save_run_config,
)


def _print_split_summary(summary: dict) -> None:
    train = summary["train"]
    validation = summary["validation"]
    print(
        "Training: "
        f"{train['patches']} patches / {train['patients']} patients / "
        f"classes={train['binary_class_counts']}"
    )
    print(
        "Validation: "
        f"{validation['patches']} patches / {validation['patients']} patients / "
        f"classes={validation['binary_class_counts']}"
    )
    print(f"Patient overlap: {summary['patient_overlap']}")


def run_experiment(config: ExperimentConfig) -> None:
    torch.set_float32_matmul_precision("high")
    L.seed_everything(config.seed, workers=True)

    print(f"Experiment: {config.experiment_name}")
    print(
        f"Architecture: {architecture_display_name(config.architecture)} "
        f"({config.architecture})"
    )
    print(f"Source patch size: {config.source_patch_size} x {config.source_patch_size}")
    print(
        f"Network input size: {config.resolved_model_input_size} x "
        f"{config.resolved_model_input_size}"
    )
    if config.architecture == "vit_b_16":
        print("ViT spatial preprocessing: Resize(246) -> CenterCrop(224)")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Data directory: {config.data_dir}")

    samples = scan_patch_samples(
        config.data_dir,
        expected_stain=config.stain,
        strict_filenames=config.strict_filenames,
    )
    print(
        f"Found {len(samples)} patches from "
        f"{len({sample.patient_id for sample in samples})} patients."
    )

    with Image.open(samples[0].path) as image_file:
        first_patch_size = image_file.size
    expected_size = (config.source_patch_size, config.source_patch_size)
    if first_patch_size != expected_size:
        raise ValueError(
            "The first source patch has size "
            f"{first_patch_size}, but --source-patch-size expects "
            f"{expected_size}."
        )

    split = resolve_split(
        samples,
        n_splits=config.n_splits,
        fold=config.fold,
        seed=config.seed,
        stratify_mode=config.stratify_mode,
        split_file=config.split_file,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(
        config.output_dir / "run_config.json",
        config.to_serializable_dict(),
    )
    save_sample_manifest(
        config.output_dir / "sample_split_manifest.csv",
        samples,
        split,
    )
    save_patient_split(
        config.output_dir / "patient_split.csv",
        samples,
        split,
    )
    summary = save_split_summary(
        config.output_dir / "split_summary.json",
        samples,
        split,
    )
    _print_split_summary(summary)

    if config.dry_run:
        print("Dry run completed; no model was trained.")
        return

    train_transform, val_transform = build_transforms(
        use_reinhard=config.use_reinhard,
        target_image=config.resolved_target_image,
        architecture=config.architecture,
    )
    loaders = build_dataloaders(
        samples,
        split,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    logger = TensorBoardLogger(
        save_dir=str(config.log_dir),
        name=f"{config.experiment_name}/{config.architecture}",
        version=f"seed_{config.seed:02d}_fold_{config.fold}",
    )

    best_metrics = BestMetricsAtBestValLoss()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.patience,
        min_delta=config.min_delta,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=config.output_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )

    model = LitClassifier(
        build_image_classifier(
            config.architecture,
            num_classes=2,
            pretrained=True,
            freeze_backbone=True,
            input_size=config.resolved_model_input_size,
        ),
        learning_rate=config.learning_rate,
    )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[early_stopping, checkpoint, best_metrics],
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(
        model,
        train_dataloaders=loaders.train_loader,
        val_dataloaders=loaders.val_loader,
    )

    best_checkpoint_path = Path(checkpoint.best_model_path)
    best_state_dict_path = config.output_dir / "best_net_state_dict.pth"
    save_best_network_state_dict(
        best_checkpoint_path,
        best_state_dict_path,
    )

    message = getattr(early_stopping, "stopping_reason_message", None)
    if message:
        print(f"Early stopping details: {message}")

    train_summary = summary["train"]
    val_summary = summary["validation"]
    append_training_result(
        config.results_csv,
        {
            "experiment": config.experiment_name,
            "architecture": config.architecture,
            "source_patch_size": config.source_patch_size,
            "model_input_size": config.resolved_model_input_size,
            "stain": config.stain,
            "reinhard": config.use_reinhard,
            "seed": config.seed,
            "fold": config.fold,
            "n_splits": config.n_splits,
            "stratify_mode": config.stratify_mode,
            "split_source": split.source,
            "train_patches": train_summary["patches"],
            "validation_patches": val_summary["patches"],
            "train_patients": train_summary["patients"],
            "validation_patients": val_summary["patients"],
            "patient_overlap_count": len(summary["patient_overlap"]),
            "best_epoch_at_val_loss": best_metrics.best_epoch,
            "best_val_loss": best_metrics.best_val_loss,
            "train_acc_at_best_val_loss": best_metrics.best_train_acc,
            "val_acc_at_best_val_loss": best_metrics.best_val_acc,
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_state_dict_path": str(best_state_dict_path),
        },
    )

    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Best network state_dict: {best_state_dict_path}")
    print(f"Results CSV: {config.results_csv}")
