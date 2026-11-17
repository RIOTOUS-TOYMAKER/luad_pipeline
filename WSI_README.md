# Unified conventional and post-Xenium WSI invasion-size inference

The same `wsi_inference/` package handles two input types:

- `standard`: conventional NDPI/WSI inference using the existing level-0
  512-pixel tile and 256-pixel stride behavior.
- `post-xenium`: post-Xenium H&E OME-TIFF inference. The source-image MPP is
  used to read the same physical field of view as the model saw during
  training, and the result is returned to the original post-Xenium level-0
  pixel coordinate system.

Both modes use the same:

- ImageNet-1K normalization
- optional torchstain Reinhard normalization
- checkpoint loader and architecture detection
- ROI filtering and background filtering
- batched inference
- connected-component measurement
- array, CSV, JSON, and GeoJSON output code

## Architecture selection

Conventional and post-Xenium inference support:

```text
convnext_small
swin_s
resnet152
efficientnet_b6
vit_b_16
```

`--architecture auto` detects the architecture from checkpoint keys. Aliases
`efficientnet` and `visiontransformer` are accepted. WSI extraction continues
to use a 512 × 512 source patch by default. ViT-B/16 then applies
`Resize(246) → CenterCrop(224)` and uses a 224 × 224 network input; the other
architectures retain 512 × 512.

## Generic entry point

### Conventional WSI

```bash
python infer_wsi.py \
    --input-type standard \
    --experiment HE \
    --case 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0
```

The existing short wrappers remain available:

```bash
python HE_invasion_size.py \
    --case 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0
```

### Post-Xenium H&E

```bash
python infer_wsi.py \
    --input-type post-xenium \
    --experiment HE \
    --case 1 \
    --wsi-path /path/to/post_xenium_HE.ome.tif \
    --geojson-path /path/to/tumor_roi.geojson \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --input-mpp 0.2738
```

Equivalent short wrapper:

```bash
python HE_invasion_size_post_xenium.py \
    --case 1 \
    --wsi-path /path/to/post_xenium_HE.ome.tif \
    --roi-geojson /path/to/tumor_roi.geojson \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --input-mpp 0.2738
```

For an HE model trained without Reinhard normalization:

```bash
python HE_nr_invasion_size_post_xenium.py \
    --case 1 \
    --wsi-path /path/to/post_xenium_HE.ome.tif \
    --roi-geojson /path/to/tumor_roi.geojson \
    --checkpoint outputs/HE_nr/convnext_small/seed_01_fold_0 \
    --input-mpp 0.2738
```

`--input-mpp` can be omitted when the OME-TIFF metadata is correct. It accepts
one isotropic value or two values in X/Y order.

## EfficientNet-B6 and ViT-B/16 examples

```bash
python HE_invasion_size_post_xenium.py \
    --case 18 \
    --architecture efficientnet_b6 \
    --wsi-path /path/to/post_xenium_HE.ome.tif \
    --roi-geojson /path/to/tumor_roi.geojson \
    --checkpoint outputs/HE/efficientnet_b6/seed_01_fold_0 \
    --input-mpp 0.2738

python HE_invasion_size_post_xenium.py \
    --case 18 \
    --architecture vit_b_16 \
    --tile-size 512 \
    --batch-size 8 \
    --wsi-path /path/to/post_xenium_HE.ome.tif \
    --roi-geojson /path/to/tumor_roi.geojson \
    --checkpoint outputs/HE/vit_b_16/seed_01_fold_0 \
    --input-mpp 0.2738
```

## Scale handling

The default classifier geometry is:

```text
model_mpp = 0.2296 µm/pixel
model tile = 512 × 512 pixels
model stride = 256 pixels
```

For conventional WSI, the pipeline retains the original behavior: it reads a
512 × 512 level-0 tile and advances 256 level-0 pixels.

For post-Xenium H&E, it calculates the source-image footprint from the physical
model field of view:

```text
source tile width  = tile_size × model_mpp ÷ input_mpp_x
source tile height = tile_size × model_mpp ÷ input_mpp_y
source stride x    = stride × model_mpp ÷ input_mpp_x
source stride y    = stride × model_mpp ÷ input_mpp_y
```

The source tile is then resized to 512 × 512 before ImageNet-normalized model
inference. GeoJSON coordinates are never converted to model pixels; they stay
in the original input-image level-0 coordinate system.

## GeoJSON profiles

`--geojson-profile auto` is the default:

- conventional WSI → `invasion_regions.geojson`
- post-Xenium H&E → `invasion_regions_xenium.geojson`

The post-Xenium file includes annotation/classification properties and preserves
coordinates in the original post-Xenium H&E level-0 pixel system. By default,
post-Xenium output polygons are clipped to the tumor ROI and simplified by
1 µm with topology preservation.

Override these behaviors with:

```bash
--geojson-profile standard
--geojson-profile xenium
--clip-output-to-roi
--no-clip-output-to-roi
--geojson-simplify-um 2.0
--classification-name "AI invasion"
--classification-color 255 0 0
```

## ImageNet normalization

All WSI modes share `preprocess/transforms.py` with training and patch testing:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

Use checkpoints trained with the same ImageNet normalization.

## Validate before full inference

Conventional WSI:

```bash
python HE_invasion_size.py \
    --case 1 \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --validate-only
```

Post-Xenium H&E:

```bash
python HE_invasion_size_post_xenium.py \
    --case 1 \
    --wsi-path /path/to/post_xenium_HE.ome.tif \
    --roi-geojson /path/to/tumor_roi.geojson \
    --checkpoint outputs/HE/convnext_small/seed_01_fold_0 \
    --input-mpp 0.2738 \
    --validate-only
```

## Output files

Each run creates:

```text
run_config.json
inference_summary.json
invasion_prob_map.npy
evaluated_tile_mask.npy
invasion_binary_mask.npy
grid_coordinates.npz
invasion_components.csv
```

The final annotation file is selected by the GeoJSON profile:

```text
invasion_regions.geojson
# or
invasion_regions_xenium.geojson
```

