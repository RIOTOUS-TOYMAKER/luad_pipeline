# LUAD patch training, patch evaluation, and WSI inference pipeline

This project modularizes patient-grouped patch training, independent patch-level performance evaluation, and WSI-level invasion-size inference. Training, testing, Integrated Gradients visualization, and WSI inference share the same ImageNet-normalized preprocessing implementation.

## Research-use notice

This software was developed for research purposes and has not been
validated, certified, or authorized for clinical diagnosis, treatment
selection, or other patient-care decision-making. The authors make no
representation or warranty regarding its clinical performance, safety,
or fitness for medical use.

## Availability of trained models and datasets

The trained model weights are not distributed as part of this repository.

The histopathological images and associated training, validation, and test
datasets are not publicly available because they contain data derived from
human pathological specimens and are subject to institutional, ethical,
legal, and data-governance restrictions.

Requests for access should be directed to the corresponding author.
Requests will be considered for research and verification purposes,
subject to approval by the relevant institution, execution of an
appropriate data-use agreement, and any other applicable requirements.

The GNU General Public License applicable to the source code in this
repository does not grant any right to access or use the trained model
weights or datasets.

## License

Copyright (C) 2026 Ryutaro Sekimoto

Except where otherwise noted, the original source code in this repository
is licensed under the GNU General Public License version 3 only
(GPL-3.0-only). Commercial and non-commercial use, modification, and
redistribution of the source code are permitted subject to the terms of
the GPL-3.0-only license. See the LICENSE file for details.

## Third-party pretrained weights

This repository does not distribute ImageNet, ImageNet-1K images, or the
original pretrained model-weight files.

When pretrained initialization is enabled, the software downloads or loads
third-party ImageNet-1K-pretrained weights provided through TorchVision.
These pretrained weights are not licensed by the authors of this repository
under the GPL-3.0-only license.

The pretrained weights may be subject to separate licenses, terms, or
restrictions derived from their original providers or training datasets.
Users are responsible for determining whether their intended use, including
commercial use, is permitted under all applicable third-party terms.

In particular, this repository uses the following TorchVision weight
variants:

- ConvNeXt_Small_Weights.IMAGENET1K_V1
- Swin_S_Weights.IMAGENET1K_V1
- ResNet152_Weights.IMAGENET1K_V2
- EfficientNet_B6_Weights.IMAGENET1K_V1
- ViT_B_16_Weights.IMAGENET1K_V1

# Environment Setup

The Python packages required to run the scripts are listed in requirements.txt. Install the dependencies using the following command:

pip install -r requirements.txt

# Path Configuration

Before running the scripts, update the following paths to match your local directory structure and dataset.

config.py
-Set training_validation_dataset to the directory containing the patch images used for training and validation.
-Set target_image_folder to the directory containing the reference images for Reinhard color normalization. A reference image should be provided for each staining method (H&E, EVG, CK7+EVG, and CK7+VB).

patch_evaluation/settings.py
-Set the path to the directory containing the patch images used for testing.

wsi_inference/settings.py
-Set the path to the directory containing the whole-slide images (WSIs).
-Set the path to the directory containing the corresponding GeoJSON files defining the annotated tumor regions.

# Preparation for WSI-Level Inference
Before performing WSI-level inference, open each whole-slide image (WSI) in QuPath and annotate the tumor region(s). Assign the classification name `tumor` to each annotation and export the annotations in GeoJSON format.


## Selectable model architectures

Every training, patch-test, conventional-WSI, and post-Xenium command accepts
`--architecture`. The canonical choices are:

```text
convnext_small   ConvNeXt-Small, ImageNet-1K V1
swin_s           Swin Transformer-S, ImageNet-1K V1
resnet152        ResNet152, ImageNet-1K V2
efficientnet_b6  EfficientNet-B6, ImageNet-1K V1
vit_b_16         Vision Transformer-B/16, ImageNet-1K V1
```

The aliases `efficientnet`, `visiontransformer`, `vision_transformer`, and
`vit` are accepted and stored as the canonical names. All five backbones use
the same trainable classification head:

```text
backbone features
→ Linear(..., 512) → ReLU → Dropout(0.5)
→ Linear(512, 128) → ReLU → Dropout(0.3)
→ Linear(128, 2)
```

For EfficientNet-B6 the entire native classifier is replaced, so its original
leading Dropout is not retained. ViT-B/16 uses the historical project spatial
preprocessing `Resize(246) → CenterCrop(224)` and is constructed at its native
224 × 224 ImageNet input size. The other four architectures retain the original
512 × 512 source patch without resize or center crop.

Examples:

```bash
python HE_cross.py --architecture efficientnet_b6 --seed 1 --fold 0
python HE_cross.py --architecture vit_b_16 --seed 1 --fold 0 --batch-size 8
```

For ViT-B/16, `--source-patch-size 512` describes the original patch field;
the network input is resolved automatically to 224 × 224 after Resize(246) and
CenterCrop(224).

## Directory structure

```text
luad_pipeline/
├── architectures.py
├── config.py
├── MODEL_ARCHITECTURES.md
├── train.py
├── HE_cross.py
├── HE_nr_cross.py
├── EVG_cross.py
├── EVG_nr_cross.py
├── CK7+EVG_cross.py
├── CK7+EVG_nr_cross.py
├── CK7+VB_cross.py
├── CK7+VB_nr_cross.py
├── make_shared_splits.py
├── dataloader/
│   ├── metadata.py
│   ├── dataset.py
│   ├── split.py
│   └── factory.py
├── preprocess/
│   └── transforms.py
├── training/
│   ├── model.py
│   ├── callbacks.py
│   ├── results.py
│   ├── runner.py
│   └── cli.py

```

## ImageNet-1K input normalization

Both Reinhard-normalized and non-Reinhard images are converted to RGB tensors
in `[0, 1]` and then normalized with the statistics expected by the
`torchvision` ImageNet-1K pretrained weights used by all supported backbones:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

The original 512 x 512 patch size is retained. The resize and center-crop steps
from the standard ImageNet inference preset are intentionally not applied.

`RandomErasing(value=0)` is applied after normalization. Therefore, an erased
region is filled with normalized zero, which corresponds to the ImageNet RGB
channel means rather than raw black. For visualization, use the shared inverse
normalization helper:

```python
from preprocess import denormalize_imagenet1k

image_for_display = denormalize_imagenet1k(normalized_image)
```

## Expected data layout

```text
path/to/the/training_validation_dataset/
└── HE/
    ├── in_situ/
    │   └── lepidic/
    │       └── HE_p01_...png
    └── invasion/
        ├── acinar/
        │   └── HE_p01_...png
        └── solid/
            └── HE_p02_...png
```

The first two underscore-delimited filename fields must be:

```text
<stain>_<patient ID>
```

Examples:

```text
HE_p01_lepidic_0001_x31451_y49858.png
CK7+EVG_p59_acinar_0006_x43583_y32079.png
```

The patient field is checked strictly against `pNN`. This prevents silent group
parsing errors.

## All samples are split together

This pipeline first pools all patches and then calls
`StratifiedGroupKFold` once:

- `groups`: patient ID parsed from the second filename field
- default `y`: composite label `binary_class::histologic_subtype`
- `n_splits=5`, `fold=0`: approximately an 80/20 grouped split
- `shuffle=True`, `random_state=seed`: reproducible repeated partitions

The code asserts that the train/validation patient intersection is empty.

Because entire patients are indivisible, patch counts and subtype proportions
will not necessarily be exactly 80/20. A warning is emitted when a subtype occurs
in fewer patient groups than the number of folds.


## Audit the split before training

```bash
python HE_cross.py --seed 1 --dry-run
```

This creates:

```text
outputs/HE/convnext_small/seed_01_fold_0/
├── run_config.json
├── patient_split.csv
├── sample_split_manifest.csv
└── split_summary.json
```

Check that `patient_overlap` is an empty list.

## Train one condition

Reinhard-normalized HE:

```bash
python HE_cross.py --seed 1
```

HE without Reinhard normalization:

```bash
python HE_nr_cross.py --seed 1
```

Equivalent generic command:

```bash
python train.py --stain HE --reinhard --architecture convnext_small --seed 1
python train.py --stain HE --no-reinhard --architecture convnext_small --seed 1
```


## Outputs

Each run stores the best Lightning checkpoint and extracts the network weights
from that best checkpoint:

```text
best-epoch=...ckpt
last.ckpt
best_net_state_dict.pth
```

This differs from saving the in-memory model after training, which can represent
the last epoch rather than the minimum-validation-loss epoch.

Results are separated by architecture and appended to:

```text
HE_convnext_small_sgkf_training_results.csv
HE_nr_convnext_small_sgkf_training_results.csv
...
```


# Independent patch-level performance evaluation

The eight `*_cross_test.py` scripts are short entry points backed by
`patch_evaluation/`. Accuracy, confusion matrix, classification report, ROC/AUC,
sensitivity, specificity, and per-patch probabilities are always calculated.
Integrated Gradients is disabled by default and runs only when
`--integrated-gradients` (or `--ig`) is supplied.

See [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md) for model-specific details and
[PATCH_TEST_README.md](PATCH_TEST_README.md) for commands, IG options, and output files.

# WSI-level invasion-size inference

The same project contains a modular WSI inference pipeline. 

```text
luad_pipeline/
├── infer_wsi.py
├── HE_invasion_size.py
├── HE_nr_invasion_size.py
├── EVG_invasion_size.py
├── EVG_nr_invasion_size.py
├── CK7+EVG_invasion_size.py
├── CK7+EVG_nr_invasion_size.py
├── CK7+VB_invasion_size.py
├── CK7+VB_nr_invasion_size.py
└── wsi_inference/
    ├── settings.py       # paths and numerical parameters
    ├── cli.py            # command-line interface
    ├── geojson_io.py     # QuPath/standard GeoJSON loading
    ├── model_io.py       # checkpoint resolution and architecture detection
    ├── tiling.py         # ROI filtering, tissue filtering, batched inference
    ├── measurement.py    # thresholding, morphology, connected components
    ├── output.py         # NPY/CSV/GeoJSON/JSON output
    └── runner.py         # end-to-end orchestration
```

Each condition-specific entry point contains only the selected experiment name:

```python
from wsi_inference.cli import main

if __name__ == "__main__":
    main(default_experiment="HE")
```

## Training/inference preprocessing parity

WSI inference imports `build_inference_transform()` from the same `preprocess`
package used by patch training. Both Reinhard and non-Reinhard conditions end
with:

```python
transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
```

WSI source tiles remain 512 x 512 by default. ViT-B/16 alone applies
Resize(246) and CenterCrop(224) immediately before the network; all other
architectures retain 512 x 512. Reinhard inference uses the same torchstain-based implementation as training.

## Run one case

For a newly trained HE + Reinhard model:

```bash
python HE_invasion_size.py \
    --case 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0
```

HE without Reinhard normalization:

```bash
python HE_nr_invasion_size.py \
    --case 1 \
    --checkpoint outputs/HE_nr/convnext_small/seed_01_fold_0
```

A generic command can replace all eight wrappers:

```bash
python infer_wsi.py \
    --experiment 'CK7+EVG' \
    --case 1 \
    --checkpoint outputs/CK7+EVG/convnext_small/seed_01_fold_0
```

Please modify the following paths to match your local directory structure and dataset before running the scripts.
scripts:

```text
WSI root:
path/to/the/WSI_folder

GeoJSON root:
path/to/the/geojson_folder
```

Thus, case 1 HE resolves to:

```text
path/to/the/WSI_folder/p01_HE.ndpi
path/to/the/geojson_folder/HE/p01_HE.ndpi.geojson
```

Override either path when necessary:

```bash
python HE_invasion_size.py \
    --case 1 \
    --checkpoint /path/to/model.pth \
    --wsi-path /path/to/p01_HE.ndpi \
    --geojson-path /path/to/p01_HE.ndpi.geojson
```

## Validate before the full WSI run

This checks the WSI, ROI, target image, checkpoint, model architecture, and WSI
MPP without classifying all tiles:

```bash
python HE_invasion_size.py \
    --case 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --validate-only
```

Checkpoint architecture is inferred from its state-dict keys for all five
supported models. Specify one of `convnext_small`, `swin_s`, `resnet152`,
`efficientnet_b6`, or `vit_b_16` only when automatic detection cannot resolve
the checkpoint.


## Default WSI parameters

```text
tile_size               512 pixels
stride                   256 pixels
batch_size               128
invasion_threshold       0.50
ROI overlap threshold    0.30
HSV saturation threshold 20
minimum tissue ratio     0.10
minimum component area   0.58 mm²
invasion class index     1
```

All are command-line options, for example:

```bash
python HE_invasion_size.py \
    --case 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --batch-size 64 \
    --invasion-threshold 0.6 \
    --min-area-mm2 0.58
```

## Output files

For case 1 HE, the default directory is:

```text
1_HE_tumor_roi_invasion_result/
├── run_config.json
├── inference_summary.json
├── invasion_prob_map.npy
├── evaluated_tile_mask.npy
├── invasion_binary_mask.npy
├── grid_coordinates.npz
├── invasion_components.csv
└── invasion_regions.geojson
```

`evaluated_tile_mask.npy` is saved separately because zero in the filled
probability map can otherwise mean either a true near-zero invasive probability
or a tile omitted because it was outside the ROI/background.

For compatibility with the original downstream analysis, one row is also
appended to:

```text
HE_invasion_size_inference.csv
```

using the original two-column schema:

```csv
case,invasion_size
1,12.345
```

The `invasion_size` value retains the original scripts' definition: the largest
`farthest_mm` among connected positive grid components. `major_axis_mm` and
`bbox_long_mm` are recorded separately in `invasion_components.csv` and
`inference_summary.json`.

# Unified conventional and post-Xenium WSI input

WSI inference uses one shared runner and one generic entry point:

```bash
python infer_wsi.py --input-type standard --experiment HE ...
python infer_wsi.py --input-type post-xenium --experiment HE ...
```

The post-Xenium mode reads an OME-TIFF at its source MPP, adapts the source tile
footprint to the physical field of view used during patch training, and exports
`invasion_regions_xenium.geojson` in the original post-Xenium H&E level-0 pixel
coordinate system. Conventional WSI behavior remains available through the same
CLI and the existing eight short stain/normalization wrappers.

See [WSI_README.md](WSI_README.md) for complete commands and output rules.
