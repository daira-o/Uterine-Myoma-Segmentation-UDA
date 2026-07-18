# Cross-Modal Weakly Supervised Uterine Myoma Segmentation

This repository contains the implementation of a research project on uterine
myoma segmentation in ultrasound images. The approach uses sagittal T2-weighted
MRI with pixel-level segmentation masks as the fully supervised source domain
and adapts the model to ultrasound using weak supervision from bounding-box
annotations.

The repository covers the complete experimental workflow: MRI and ultrasound
preprocessing, supervised source training, weakly supervised target adaptation,
checkpoint evaluation, inference, and qualitative visualization. Optional
Domain-Adversarial Neural Network (DANN) components are also included for
experimental comparison.

This project was developed for academic research. It has not been validated for
clinical diagnosis, treatment planning, or autonomous medical decision-making.

## Research Objective

Pixel-level segmentation masks are expensive to create for ultrasound images,
while bounding-box annotations require less annotation effort. This project
investigates whether a segmentation model trained with fully annotated MRI data
can be adapted to ultrasound using only bounding boxes in the target domain.

The main objectives are:

- Train an Attention U-Net on sagittal T2-weighted MRI images with expert
  segmentation masks.
- Transfer the learned representation from MRI to ultrasound.
- Adapt the model using weak supervision derived from ultrasound bounding boxes.
- Preserve source-domain MRI performance during target adaptation.
- Compare weakly supervised adaptation with optional domain-adversarial
  regularization.
- Evaluate ultrasound predictions using the localization information available
  in the target annotations.

## Method Overview

The workflow follows a cross-modal source-to-target strategy:

```text
Sagittal T2 MRI + pixel-level masks
                │
                ▼
       MRI preprocessing
                │
                ▼
 Supervised Attention U-Net training
                │
                ▼
       Pretrained MRI model
                │
                ├──────────────────────────┐
                ▼                          ▼
 Ultrasound preprocessing       MRI validation monitoring
 + bounding-box conversion
                │
                ▼
 Weakly supervised target adaptation
                │
                ▼
 Checkpoint comparison and selection
                │
                ▼
 Ultrasound inference and visualization
```

During target adaptation, training combines fully supervised MRI samples with
weakly supervised ultrasound samples. The ultrasound objective uses the
available bounding boxes to constrain the predicted segmentation without
treating the entire box as a pixel-level ground-truth mask.

Optional DANN components can be enabled to study adversarial alignment between
the MRI and ultrasound feature distributions.

## Datasets

This project uses two publicly available research datasets. The datasets are not
redistributed in this repository and must be downloaded from their original
sources.

### Uterine Myoma MRI Dataset

The source domain uses the **Uterine Myoma MRI Dataset (UMD)**. It contains
sagittal T2-weighted MRI studies from 300 uterine myoma cases together with
pixel-level annotation files.

- [Download the UMD dataset from Figshare](https://figshare.com/articles/dataset/UMD_zip/23541312)
- [Read the dataset publication in Scientific Data](https://www.nature.com/articles/s41597-024-03170-x)

The MRI annotations are used for fully supervised source-domain segmentation
training and validation.

### Uterine Fibroid Ultrasound Images

The target domain uses **Uterine Fibroid Ultrasound Images, Version 2**, made
available through Mendeley Data.

- [Download the ultrasound dataset from Mendeley Data](https://data.mendeley.com/datasets/n2zcmcypgb/2)

The ultrasound images include bounding-box annotations rather than pixel-level
segmentation masks. These annotations are used to construct the weak
target-domain supervision.

Users are responsible for reviewing and following the licenses, access
conditions, and citation requirements of both datasets.

## Data Selection

The MRI dataset provides the fully supervised source samples.

For the ultrasound domain, the original collection was manually reviewed to
exclude images that did not meet the project criteria, including unsuitable
views, pregnancy-related images, and samples with excessive visual noise. The
final curated ultrasound subset contains 288 images.

Because the ultrasound dataset does not provide acquisition metadata for every
sample, the target data is described as **2D pelvic ultrasound** rather than
being restricted to one probe or acquisition route.

## Preprocessing Pipeline

### MRI preprocessing

The MRI pipeline prepares sagittal 2D slices from the original NIfTI volumes.

Main steps:

1. Load the MRI volume and its corresponding segmentation mask.
2. Convert the volume to a canonical orientation.
3. extract sagittal slices.
4. Match each image slice with its mask.
5. Normalize intensities at the volume level.
6. Resample to approximately 0.8 mm per pixel.
7. Resize or pad each sample to a final input size of 256 × 256 pixels.
8. Export the processed images and masks for training.

Run the MRI preprocessing pipeline with:

```bash
python scripts/data_preparation/mri_pipeline.py
```

### Ultrasound preprocessing

The ultrasound pipeline standardizes images from different acquisition
field-of-view groups and converts the available bounding boxes to the final
model coordinates.

Main steps:

1. Remove text and non-image overlays where applicable.
2. Apply image enhancement and normalization.
3. Detect and crop the relevant ultrasound region.
4. Use visible scale information when available to standardize physical size.
5. Resample to approximately 0.8 mm per pixel.
6. Resize or pad the image to 256 × 256 pixels.
7. Transform the original bounding box into the processed image coordinates.
8. Export the processed image and its weak annotation.

Run the main ultrasound preprocessing pipeline with:

```bash
python scripts/data_preparation/us_pipeline.py
```

Review annotation organization before applying file changes:

```bash
python scripts/data_preparation/organize_us_annotations.py --dry-run
```

Apply the annotation organization:

```bash
python scripts/data_preparation/organize_us_annotations.py
```

For manually curated ultrasound data, rebuild the final splits with:

```bash
python scripts/evaluation/review_us_dataset.py --help
python scripts/data_preparation/build_us_splits_from_clean.py --help
```

## Model Architecture

The segmentation model is based on an Attention U-Net for binary uterine myoma
segmentation.

The architecture includes:

- An encoder with progressively deeper feature maps.
- A bottleneck representation.
- A decoder with skip connections.
- Attention gates that filter encoder features before concatenation.
- A single-channel output containing segmentation logits.

The repository also includes optional components for domain-adversarial
experiments:

- `AttentionUNetDANN`, which exposes encoder features.
- `GradientReversalLayer`, which reverses domain-classification gradients.
- `DomainDiscriminator`, which predicts whether extracted features belong to
  MRI or ultrasound.
- `DANNUNet`, which combines the segmenter and domain branch.

## Training Workflow

### Stage 1: supervised MRI training

The first stage trains the Attention U-Net using MRI images and pixel-level
segmentation masks.

The supervised objective combines binary cross-entropy with Dice loss.

Run source training with:

```bash
python scripts/training/train_source.py
```

The source model is evaluated with:

- Dice coefficient.
- HD95.
- Object-level precision.
- Training and validation losses.

### Stage 2: weakly supervised ultrasound adaptation

The pretrained MRI model is then adapted to ultrasound. Each training step
combines a supervised MRI batch and a weakly supervised ultrasound batch.

The weak ultrasound objective contains three main components:

- **Outside-box loss:** penalizes predicted foreground outside the expanded
  bounding box.
- **Inside-box loss:** encourages the model to predict foreground within the
  annotated region.
- **Area constraint:** discourages masks with implausibly small or large areas
  relative to the bounding box.

Run target adaptation with:

```bash
python scripts/training/train_target.py
```

MRI validation is monitored during adaptation to detect source-domain
performance degradation.

DANN is available as an optional regularizer, but the main target workflow can
run using only the weak ultrasound supervision.

## Checkpoint Evaluation

Target checkpoints are compared using both source-domain segmentation metrics
and target-domain weak localization metrics.

Run the comparison tool with:

```bash
python scripts/evaluation/compare_checkpoints_us.py --help
```

The main evaluation signals include:

- MRI validation Dice.
- MRI validation HD95.
- MRI object-level precision.
- Ultrasound bounding-box inside ratio.
- Ultrasound centroid-inside ratio.
- Predicted mask area ratio.
- Weak-supervision losses.
- Domain loss and domain accuracy when DANN is enabled.

Because pixel-level ultrasound masks are unavailable, the ultrasound metrics are
indirect. They measure consistency with the available bounding boxes and should
not be interpreted as direct segmentation accuracy.

## Main Results

Selected results from the final weakly supervised target-adaptation experiment:

| Metric | Result |
|---|---:|
| MRI validation Dice | 0.9075 |
| MRI validation HD95 | 5.33 px |
| MRI object-level precision | 0.8918 |
| Ultrasound bounding-box inside ratio | 0.6409 |
| Ultrasound centroid-inside ratio | 0.7000 |
| Target checkpoint score | 0.7541 |

For comparison, the supervised MRI source model reached a validation Dice of
0.9126 before target adaptation.

The final adapted model retained most of the MRI segmentation performance while
improving its consistency with the weak ultrasound annotations. However,
expert-reviewed ultrasound masks are still required for direct target-domain
segmentation evaluation.

## Inference

Run ultrasound inference with:

```bash
python scripts/inference/infer_us_production.py --help
```

The inference pipeline:

1. Loads a selected model checkpoint.
2. Applies the ultrasound preprocessing used during training.
3. Generates a probability map.
4. Applies the selected segmentation threshold.
5. Exports the predicted mask and visualization outputs.

## Visualization

MRI Streamlit dashboard:

```bash
streamlit run scripts/visualization/visualizar_modelo.py
```

Ultrasound Streamlit dashboard:

```bash
streamlit run scripts/visualization/visualizar_us_modelo.py
```

Additional visualization scripts:

```bash
python scripts/visualization/visualizador_mri.py
python scripts/visualization/visualizador_us.py
python scripts/visualization/visualizar_epocas_target.py
```

The visualization tools support qualitative inspection of:

- MRI images and ground-truth masks.
- MRI model predictions.
- Ultrasound images and bounding boxes.
- Ultrasound probability maps.
- Thresholded segmentation predictions.
- Prediction changes across target-training epochs.

## Repository Structure

```text
Cross-Modal-Myoma-Segmentation/
├── config.py
├── models/
│   ├── attention_unet.py
│   ├── attention_unet_dann.py
│   ├── dann_unet.py
│   ├── domain_discriminator.py
│   └── grl.py
├── scripts/
│   ├── data_preparation/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── docs/
│   └── dann_implementation.md
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/daira-o/Cross-Modal-Myoma-Segmentation.git
cd Cross-Modal-Myoma-Segmentation
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Before running the scripts, review `config.py` and update the dataset paths and
experiment settings for your local environment.

## Limitations

- The ultrasound dataset does not provide pixel-level segmentation masks.
- Ultrasound evaluation relies on bounding-box localization and other indirect
  metrics.
- Bounding-box supervision does not provide enough information to assess exact
  lesion boundaries.
- The MRI and ultrasound datasets differ in modality, annotation type, image
  appearance, and lesion presentation.
- The curated ultrasound dataset is limited to 288 images.
- Ultrasound performance may be affected by acquisition protocol, field of
  view, device settings, image quality, and preprocessing consistency.
- Segmentation outputs are sensitive to the selected probability threshold.
- External validation with independent ultrasound data and expert-reviewed
  pixel-level masks is required before any clinical application.

## Intended Use

This repository is intended for academic research, reproducibility studies,
weakly supervised medical image segmentation experiments, and cross-modal
adaptation research.

It is not intended for clinical diagnosis, treatment planning, patient triage,
autonomous medical decision-making, or use as a certified medical device.

## Citation

A formal citation will be added if the associated research work is published.

Until then, academic use of this repository should acknowledge this project and
cite the original MRI and ultrasound dataset sources.

## Author

**Daira Orlandini**

Computer Engineering student  
Universidad de Palermo