# Patch-level performance evaluation

The independent patch-test scripts are modularized under `patch_evaluation/`.
All eight stain/Reinhard entry points share the same model loading,
preprocessing, prediction, metrics, plotting, CSV output, and optional
Integrated Gradients implementation.

## Directory structure

```text
luad_pipeline/
├── evaluate_patches.py
├── HE_cross_test.py
├── HE_nr_cross_test.py
├── EVG_cross_test.py
├── EVG_nr_cross_test.py
├── CK7+EVG_cross_test.py
├── CK7+EVG_nr_cross_test.py
├── CK7+VB_cross_test.py
├── CK7+VB_nr_cross_test.py
├── modeling/
│   └── checkpoints.py
└── patch_evaluation/
    ├── settings.py
    ├── cli.py
    ├── data.py
    ├── inference.py
    ├── metrics.py
    ├── output.py
    ├── integrated_gradients.py
    └── runner.py
```

Each condition-specific script is only an entry point. For example,
`HE_cross_test.py` contains:

```python
from patch_evaluation.cli import main

if __name__ == "__main__":
    main(default_experiment="HE")
```

## Architecture selection

The evaluator supports the same five models as training:

```text
convnext_small
swin_s
resnet152
efficientnet_b6
vit_b_16
```

Use `--architecture auto` to infer the model from checkpoint keys, or specify it
explicitly. The aliases `efficientnet` and `visiontransformer` are accepted.
All architectures use the same 512→128→2 classification head. ViT-B/16 receives the
original 512 × 512 patch through `Resize(246) → CenterCrop(224)` before the
network; the other architectures keep the 512 × 512 tensor.

## Preprocessing

Patch training, patch evaluation, and WSI inference import the same deterministic
transform from `preprocess/transforms.py`:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

For ViT-B/16, the original 512 x 512 field is resized to 246 and center-cropped
to 224. Other architectures retain 512 x 512. For a Reinhard experiment,
torchstain Reinhard normalization is applied after the ViT spatial operations and before
ImageNet channel normalization. For an `_nr` experiment, the RGB image is
converted directly to a tensor before ImageNet normalization.

The test data must have this layout:

```text
/media/owner/Data/deep_learning_data/test_sorted/
└── HE/
    ├── in_situ/
    │   └── *.png
    └── invasion/
        └── *.png
```

The evaluator checks that `ImageFolder.classes` is exactly:

```python
["in_situ", "invasion"]
```

This prevents accidental inversion of class 0 and class 1. The default positive
class for ROC, sensitivity, and specificity is `invasion`.

## Checkpoint selection

`--checkpoint` is required. It may point to either:

```text
outputs/HE/convnext_small/seed_01_fold_0
```

or a specific file:

```text
outputs/HE/convnext_small/seed_01_fold_0/best_net_state_dict.pth
outputs/HE/convnext_small/seed_01_fold_0/best-epoch=...ckpt
```

When a directory is supplied, `best_net_state_dict.pth` is preferred; otherwise
one `best-*.ckpt` is selected. ConvNeXt-Small, Swin-S, ResNet152,
EfficientNet-B6, and ViT-B/16 are detected from checkpoint keys unless
`--architecture` is specified.

If an adjacent `run_config.json` explicitly records preprocessing different
from ImageNet-1K mean/std, evaluation stops rather than silently combining an
incompatible checkpoint and transform. If no metadata exists, a warning is
printed because preprocessing cannot be verified.

## Ordinary performance test: Integrated Gradients off

Integrated Gradients is disabled by default. This command calculates the full
performance test without attribution computation:

```bash
python HE_cross_test.py \
    --seed 1 \
    --fold 0 \
    --architecture convnext_small \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0
```

For HE without Reinhard normalization:

```bash
python HE_nr_cross_test.py \
    --seed 1 \
    --fold 0 \
    --checkpoint outputs/HE_nr/convnext_small/seed_01_fold_0
```

For names containing `+`, quote the script or checkpoint path in the shell:

```bash
python 'CK7+EVG_cross_test.py' \
    --seed 1 \
    --checkpoint 'outputs/CK7+EVG/convnext_small/seed_01_fold_0'
```

The generic entry point can replace all eight wrappers:

```bash
python evaluate_patches.py \
    --experiment 'CK7+VB_nr' \
    --seed 1 \
    --checkpoint 'outputs/CK7+VB_nr/convnext_small/seed_01_fold_0'
```

## EfficientNet-B6 and ViT-B/16 examples

```bash
python HE_cross_test.py \
    --seed 1 \
    --fold 0 \
    --architecture efficientnet_b6 \
    --checkpoint outputs/HE/efficientnet_b6/seed_01_fold_0

python HE_cross_test.py \
    --seed 1 \
    --fold 0 \
    --architecture vit_b_16 \
    --model-input-size 512 \
    --batch-size 8 \
    --checkpoint outputs/HE/vit_b_16/seed_01_fold_0
```

## Enable Integrated Gradients explicitly

Add `--integrated-gradients` or its alias `--ig`:

```bash
python HE_cross_test.py \
    --seed 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --integrated-gradients
```

IG can be limited while checking the pipeline:

```bash
python HE_cross_test.py \
    --seed 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --integrated-gradients \
    --ig-max-images 10 \
    --ig-n-steps 200
```

Generate IG only for incorrectly classified patches:

```bash
python HE_cross_test.py \
    --seed 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --integrated-gradients \
    --ig-only-misclassified
```

Use a zero baseline or the true class as the attribution target:

```bash
--ig-baseline zero
--ig-target true
```

Ordinary prediction can optionally use `--amp` on CUDA. Integrated Gradients is
always run outside inference mode and autocast because input gradients are
required.

## Outputs

A standard run creates:

```text
patch_test_results/HE/convnext_small/seed_01_fold_0/
├── run_config.json
├── evaluation_summary.json
├── metrics.json
├── patch_predictions.csv
├── confusion_matrix.csv
├── confusion_matrix.tiff
├── classification_report.csv
├── classification_report.txt
├── roc_curve.csv
└── ROC_curve.tiff
```

`patch_predictions.csv` contains the source path, true label, predicted label,
correct/incorrect status, and both class probabilities for every patch.

When IG is enabled, the following is added:

```text
integrated_gradients/
├── convergence_delta.csv
└── 00000_<patch-name>_true-..._pred-..._target-....tiff
```

The convergence CSV is opened once and flushed after each image, so it contains
all processed patches and retains completed rows if a later image fails.

One summary row per run is appended to:

```text
patch_test_results/HE_convnext_small_patch_test_results.csv
patch_test_results/HE_nr_convnext_small_patch_test_results.csv
...
```

## Seeds 1 through 10

For ten trained seeds without IG:

```bash
set -euo pipefail

for seed in $(seq 1 10); do
    python HE_cross_test.py \
        --seed "$seed" \
        --checkpoint "outputs/HE/convnext_small/seed_$(printf '%02d' "$seed")_fold_0"
done
```

Because IG is opt-in, the loop above performs only prediction and metrics.

## Validate paths and checkpoint first

```bash
python HE_cross_test.py \
    --seed 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --validate-only
```

This checks the test directory, class folders, target image, checkpoint,
architecture, and recorded preprocessing without running prediction or IG.
