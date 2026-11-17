from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from architectures import (
    Architecture,
    VIT_CENTER_CROP_SIZE,
    normalize_architecture_name,
)


def _classification_head(in_features: int, num_classes: int) -> nn.Sequential:
    """Common trainable head used by every supported architecture."""
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )


def _freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def build_image_classifier(
    architecture: Architecture | str,
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    input_size: int = 512,
) -> nn.Module:
    """Build a classifier with the common 512→128→class head.

    ViT-B/16 is intentionally restored to its historical 224-pixel network
    input. Its preprocessing is handled by preprocess.transforms as
    Resize(246) → CenterCrop(224); no 512-position-embedding interpolation is
    performed.
    """
    architecture = normalize_architecture_name(str(architecture))
    assert architecture != "auto"
    if input_size < 1:
        raise ValueError("input_size must be positive.")

    if architecture == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_small(weights=weights)
        in_features = net.classifier[2].in_features
        if freeze_backbone:
            _freeze_all_parameters(net)
        net.classifier[2] = _classification_head(in_features, num_classes)
        return net

    if architecture == "swin_s":
        weights = models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.swin_s(weights=weights)
        in_features = net.head.in_features
        if freeze_backbone:
            _freeze_all_parameters(net)
        net.head = _classification_head(in_features, num_classes)
        return net

    if architecture == "resnet152":
        weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet152(weights=weights)
        in_features = net.fc.in_features
        if freeze_backbone:
            _freeze_all_parameters(net)
        net.fc = _classification_head(in_features, num_classes)
        return net

    if architecture == "efficientnet_b6":
        weights = models.EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b6(weights=weights)
        in_features = net.classifier[1].in_features
        if freeze_backbone:
            _freeze_all_parameters(net)
        # Replace the whole native classifier, removing EfficientNet's extra
        # leading dropout so every architecture has exactly the common head.
        net.classifier = _classification_head(in_features, num_classes)
        return net

    if architecture == "vit_b_16":
        if input_size != VIT_CENTER_CROP_SIZE:
            raise ValueError(
                "ViT-B/16 requires input_size=224 because preprocessing is "
                "Resize(246) followed by CenterCrop(224); "
                f"got {input_size}."
            )
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.vit_b_16(weights=weights, image_size=VIT_CENTER_CROP_SIZE)
        in_features = net.heads.head.in_features
        if freeze_backbone:
            _freeze_all_parameters(net)
        net.heads = _classification_head(in_features, num_classes)
        return net

    raise AssertionError("unreachable")


def build_convnext_small(num_classes: int = 2) -> nn.Module:
    return build_image_classifier(
        "convnext_small",
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        input_size=512,
    )


class LitClassifier(L.LightningModule):
    def __init__(self, net: nn.Module, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["net"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=x.size(0),
        )
        self.log(
            "train_acc", accuracy, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=x.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=x.size(0),
        )
        self.log(
            "val_acc", accuracy, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=x.size(0),
        )

    def configure_optimizers(self):
        trainable_parameters = [
            parameter for parameter in self.net.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters were found.")
        return optim.SGD(
            trainable_parameters,
            lr=self.learning_rate,
            momentum=0.9,
        )
