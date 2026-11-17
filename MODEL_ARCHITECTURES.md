# Model architecture selection

The pipeline supports the following canonical values for `--architecture`:

| CLI value | Model | TorchVision pretrained weights | Network input |
|---|---|---|---:|
| `convnext_small` | ConvNeXt-Small | `ConvNeXt_Small_Weights.IMAGENET1K_V1` | source patch size |
| `swin_s` | Swin Transformer-S | `Swin_S_Weights.IMAGENET1K_V1` | source patch size |
| `resnet152` | ResNet152 | `ResNet152_Weights.IMAGENET1K_V2` | source patch size |
| `efficientnet_b6` | EfficientNet-B6 | `EfficientNet_B6_Weights.IMAGENET1K_V1` | source patch size |
| `vit_b_16` | Vision Transformer-B/16 | `ViT_B_16_Weights.IMAGENET1K_V1` | 224 × 224 |

Aliases accepted by the CLI include `efficientnet`, `visiontransformer`,
`vision_transformer`, and `vit`.

## Common classification head

All architectures use the same trainable head:

```text
Linear(backbone_features, 512)
ReLU
Dropout(0.5)
Linear(512, 128)
ReLU
Dropout(0.3)
Linear(128, 2)
```

For EfficientNet-B6, the entire native `model.classifier` is replaced. Its
native leading `Dropout(p=0.5)` is not retained, so EfficientNet has the same
two Dropout layers as the other architectures.

## Architecture-specific spatial preprocessing

All models share optional Reinhard normalization and ImageNet-1K channel
normalization. Spatial preprocessing differs only for ViT-B/16:

```text
ConvNeXt / Swin / ResNet152 / EfficientNet-B6
512 × 512 source patch
→ no resize or center crop
→ tensor
→ ImageNet mean/std
```

```text
ViT-B/16
512 × 512 source patch
→ Resize(246)
→ CenterCrop(224)
→ tensor
→ ImageNet mean/std
```

Training-only `RandomHorizontalFlip` and `RandomErasing` are still applied.
The deterministic validation, independent-test, Integrated Gradients, normal
WSI, and post-Xenium WSI paths all use the same ViT spatial preprocessing.

## ViT-B/16 at 224 × 224

ViT-B/16 is constructed at its native ImageNet input size of 224 × 224. The
ImageNet positional embedding therefore remains a 14 × 14 patch grid plus one
class token:

```text
14 × 14 + 1 = 197 tokens
```

## Training examples

```bash
python HE_cross.py --architecture resnet152 --seed 1 --fold 0
python HE_cross.py --architecture efficientnet_b6 --seed 1 --fold 0
python HE_cross.py --architecture vit_b_16 --batch-size 32 --seed 1 --fold 0
```

`--source-patch-size 512` is the preferred name for the original patch size.
`--model-input-size 512` remains as a compatibility alias. For ViT, the actual
network input is automatically resolved to 224.

Outputs remain separated by architecture:

```text
outputs/HE/resnet152/seed_01_fold_0/
outputs/HE/efficientnet_b6/seed_01_fold_0/
outputs/HE/vit_b_16/seed_01_fold_0/
```

## Patch evaluation examples

```bash
python HE_cross_test.py \
    --seed 1 \
    --fold 0 \
    --architecture vit_b_16 \
    --source-patch-size 512 \
    --checkpoint outputs/HE/vit_b_16/seed_01_fold_0
```

## WSI and post-Xenium examples

`--tile-size 512` continues to mean the source patch field used during
training. For ViT, each extracted 512-pixel tile is then resized to 246 and
center-cropped to 224 before network inference.

```bash
python HE_invasion_size_post_xenium.py \
    --case 18 \
    --architecture vit_b_16 \
    --tile-size 512 \
    --wsi-path /path/to/post_xenium_HE.ome.tif \
    --roi-geojson /path/to/tumor_roi.geojson \
    --checkpoint outputs/HE/vit_b_16/seed_01_fold_0 \
    --input-mpp 0.2738
```
