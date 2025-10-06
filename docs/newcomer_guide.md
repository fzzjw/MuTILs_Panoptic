# MuTILs newcomer guide

This guide introduces the MuTILs panoptic segmentation codebase, highlights key 
modules to explore first, and outlines how to prepare custom data that matches 
the expected training and inference formats.

## Repository layout

- **Top-level usage docs** – `README.md` walks through the published model, Docker 
  workflow, expected host/container mounts, and the default inference command, 
  which are the fastest way to spin up an environment before reading code.
- **`configs/`** – centralizes configuration assets. In addition to the runtime 
  YAML (`MuTILsWSIRunConfigs.yaml`), there are Python helpers that enumerate 
  region and nucleus taxonomies (`panoptic_model_configs.py`), link raw label 
  names to the standardized MuTILs codes, and expose visualization defaults.
- **`mutils_panoptic/`** – contains the core model, training, and inference code. 
  Start with `MuTILs.py` (model definition and evaluation helpers), 
  `MuTILsTrainer.py` (orchestrates cross-validation, loss computation, and 
  logging), and `MuTILsWSIRunner.py` (whole-slide inference pipeline).
- **`utils/`** – shared utilities used across training/inference: GPU allocation, 
  torchvision-style transforms, visualization helpers, and numpy/pandas tooling.
- **`tests/`** – provides an automated smoke test (`tests/test_inference.py`) for 
  verifying Dockerized inference and output directory integrity.

## Key runtime components

- **MuTILsWSIRunner** (`mutils_panoptic/MuTILsWSIRunner.py`) splits WSI processing 
  into dedicated preprocess, inference, and post-process workers. The runner 
  loads an ensemble of model checkpoints, extracts top-ranked ROIs, dispatches 
  them across GPUs, and consolidates slide-level metrics before saving masks, 
  annotations, and feature tables.
- **MuTILs model stack** (`mutils_panoptic/MuTILs.py`) defines a multiresolution 
  UNet backbone plus heads for region segmentation, nucleus classification, and 
  Computational TILs Assessment (CTA) scoring. The file also houses 
  `MuTILsTransform` (batch/normalize inputs) and `MuTILsEvaluator` (aggregates ROI 
  and HPF metrics, CTA numerators/denominators, and acceptable misclassifications).
- **Training entry point** (`mutils_panoptic/MuTILsTrainer.py`) manages cross-fold 
  sampling, data loaders, class balancing, optimizer/scheduler setup, epoch loops, 
  visualization checkpoints, and evaluation exports for later analysis.

## Training data expectations

`RegionDatasetLoaders.MuTILsDataset` is the canonical dataset used during 
training.

- **Directory structure** – the loader expects `root/tcga` and `root/acs` 
  subfolders, each containing parallel `rgbs/` and `masks/` directories with ROI 
  patches saved as PNG files. File stems follow the pattern `SLIDENAME_*.png` so 
  that the dataset can map ROIs back to slides.
- **Mask encoding** – each mask is a three-channel PNG. Channel 0 stores region 
  super-class codes, channel 1 stores nucleus classes, and channel 2 stores 
  contour pixels. During loading, nuclei in noisy regions (`OTHER`, `WHITE`) are 
  zeroed out and boundary pixels are reassigned to the nucleus background class 
  to enforce separation before augmentation.
- **Patch scaling** – ROIs are resized to the requested HPF magnification, random 
  crops/scale jitter are applied during training, and both high-res (HPF) and 
  low-res (ROI scale) copies of RGB + mask tensors are returned to the model.
- **Class balancing** – when `training=True`, the loader builds `region_summary.csv`
  and `nuclei_summary.csv` caches if they do not exist, then computes ROI sample 
  weights that simultaneously balance slides and rare tissue regions. These 
  weights feed a `WeightedRandomSampler` in `MuTILsTrainer`.
- **Train/test splits** – `get_cv_fold_slides` reads CSVs named 
  `fold_{k}_train.csv` and `fold_{k}_test.csv` from `train_test_splits/`. Provide 
  these files for your dataset to control cross-validation membership.

## Label taxonomy and constraints

The MuTILs label space is defined in `configs/panoptic_model_configs.py`:

- `RegionCellCombination.REGION_CODES` lists the region super-classes (`TUMOR`, 
  `STROMA`, `TILS`, `NORMAL`, `OTHER`, `WHITE`, `BLOOD`, etc.) used for ROI-level 
  segmentation.
- `RegionCellCombination.NUCLEUS_CODES` enumerates the nine nucleus classes (with 
  explicit `BACKGROUND` and `EXCLUDE` codes) and maps external NuCLS labels to 
  this standardized set.
- `RegionCellCombination.nuclei_regions_codes` restricts which nucleus classes are 
  allowed inside each region super-class (e.g., epithelial nuclei must appear in 
  `TUMOR`, stromal cells in `STROMA`/`TILS`, normal epithelium in `NORMAL`). The 
  loader enforces these constraints when building training masks and during 
  evaluation.
- Combined region+nucleus masks reuse these codes so that visualization and 
  panoptic metrics remain consistent across outputs.

When curating your own data, align region and nucleus annotations to these codes. 
If you must introduce new classes, extend the dictionaries and update masks, 
loss functions, and visualization color maps accordingly.

## Practical steps for onboarding and custom data

1. **Replicate inference** – follow the Docker workflow in `README.md` to run 
   `MuTILsWSIRunner` on sample slides. Confirm outputs match the structure 
   validated by `tests/test_inference.py`.
2. **Inspect datasets** – study `RegionDatasetLoaders.MuTILsDataset` to understand 
   ROI sampling, augmentation, and weighting. Prototype a notebook that prints 
   ROI masks to confirm your annotations respect region/nucleus constraints.
3. **Bootstrap splits and caches** – generate train/test CSVs, organize ROIs 
   under `tcga/` and `acs/`, and let the loader create the summary CSVs before 
   launching multi-GPU training via `MuTILsTrainer.py`.
4. **Monitor training** – leverage the loss plots, evaluation exports, and CTA 
   metrics emitted each epoch to diagnose class imbalance or mask alignment 
   issues.
5. **Iterate on label alignment** – use the taxonomy definitions and forced 
   mappings in `panoptic_model_configs.py` to audit how your labels translate into 
   MuTILs codes and adjust preprocessing pipelines or config dictionaries as 
   needed.

With these steps, new contributors can move from familiarizing themselves with 
the repository layout to running experiments on bespoke datasets while staying 
compatible with the MuTILs panoptic segmentation stack.
