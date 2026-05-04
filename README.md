[![DOI](https://zenodo.org/badge/ID/20022015.svg)](https://doi.org/10.5281/zenodo.20022014)
![MATLAB](https://img.shields.io/badge/MATLAB-required-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
# Hartmann–Shack Artifact Detection

This repository provides a complete framework for detecting artifacts in Hartmann–Shack (HS) wavefront sensor images using both classical image processing methods and deep learning approaches.

## The repository includes

* binary classification methods for **valid / invalid** HS images
* multiclass classification methods for **blinking / corneal / laser reflex / valid**
* folder-based inference scripts for easy use
* training pipelines with k-fold cross-validation for reproducible deep learning models
* optional evaluation using Excel label files

---

## Repository Structure

```text
hartmann-shack-artifact-detection/
│
├── sample_images/
├── sample_labels_2class.xlsx
├── sample_labels_4class.xlsx
│
├── classical_methods/
│   ├── morphological_artifact_detection.m
│   ├── correlation_artifact_detection.m
│   └── reference/
│       └── meanImage_weighted.bmp
│
├── deep_learning/
│   ├── training/
│   │   ├── mobilenet/
│   │   │   ├── train_mobilenet_kfold.m
│   │   │   ├── mobilenet_2class_base.mat
│   │   │   └── mobilenet_4class_base.mat
│   │   │
│   │   ├── shufflenet/
│   │   ├── squeezenet/
│   │   │
│   │   └── create_kfold_datastores.m
│   │
│   └── inference/
│       ├── mobilenet/
│       │   ├── two_class/
│       │   │   ├── mobilenet_2class_inference.m
│       │   │   └── mobilenet_2class_trained.mat
│       │   │
│       │   └── four_class/
│       │       ├── mobilenet_4class_inference.m
│       │       └── mobilenet_4class_trained.mat
│       │
│       ├── shufflenet/
│       │   ├── two_class/
│       │   └── four_class/
│       │
│       └── squeezenet/
│           ├── two_class/
│           └── four_class/
│
└── README.md
```

---

## Methods

### 1. Classical Image Processing Methods

The repository includes two classical approaches for artifact detection in HS images:

**Morphological-based detection**

* pupil detection
* pupil circularity validation
* pupil size validation
* corneal reflection detection

**Correlation-based detection**

* pupil localization
* pupil centering
* image correlation with a reference mean image
* reflection-based rejection

---

### 2. Deep Learning Methods

The repository includes deep learning pipelines based on:

* MobileNet
* ShuffleNet
* SqueezeNet

Each architecture supports:

**Binary classification**

* `valid`
* `invalid`

**Multiclass classification**

* `blinking`
* `corneal`
* `laser reflex`
* `valid`

---

## Training and Inference

Each deep learning architecture follows a unified structure:

```
network_name/
├── training/
└── inference/
```

### Training

* Training is performed using **k-fold cross-validation**
* Scripts:

  * `train_*_kfold.m`
* Base configurations:

  * `*_2class_base.mat`
  * `*_4class_base.mat`

Dataset splitting is handled using:

```
create_kfold_datastores.m
```

---

## Note on Class Labels

The class **"laser reflex"** used in the dataset corresponds to **lens reflection** in the manuscript.

Both terms refer to the same type of optical artifact.

---

## Supported Tasks

### Binary Classification

* `valid`
* `invalid`

### Multiclass Classification

* `blinking`
* `corneal`
* `laser reflex`
* `valid`

---

## Input Data

### Images

Place input images in a folder such as:

```
sample_images/
```

Supported formats:

* `.bmp`
* `.png`
* `.jpg`
* `.jpeg`
* `.tif`
* `.tiff`

---

### Label Files

Excel format:

```
filename | label
```

#### Binary Example

```
img1.bmp | 1
img2.bmp | 0
```

or:

```
img1.bmp | valid
img2.bmp | invalid
```

#### Multiclass Example

```
img1.bmp | blinking
img2.bmp | corneal
img3.bmp | laser reflex
img4.bmp | valid
```

Notes:

* filenames must match exactly
* labels may be numeric or text
* multiclass labels must match training labels

---

## Example Usage

### Binary Classification

```matlab
mobilenet_2class_inference( ...
    'sample_images', ...
    'results', ...
    'mobilenet_2class_trained.mat', ...
    'sample_labels_2class.xlsx')
```

### Multiclass Classification

```matlab
shufflenet_4class_inference( ...
    'sample_images', ...
    'results', ...
    'shufflenet_4class_trained.mat', ...
    'sample_labels_4class.xlsx')
```

---

## Output

### Binary

```
results/
├── valid/
├── invalid/
└── prediction_results_YYYYMMDD_HHMMSS.xlsx
```

### Multiclass

```
results/
├── blinking/
├── corneal/
├── laser reflex/
├── valid/
└── prediction_results_YYYYMMDD_HHMMSS.xlsx
```

Output may include:

* image name
* predicted label
* confidence score
* processing time
* optional ground truth

---

## Evaluation Metrics

### Binary

* Accuracy
* Sensitivity (Recall)
* Specificity
* Precision
* F1-score
* Confusion matrix

### Multiclass

* Overall accuracy
* Confusion matrix
* Per-class precision/recall/F1
* Macro-averaged metrics

---

## Preprocessing

### Classical

* grayscale conversion
* morphological operations
* binarization
* hole filling
* connected components

### Deep Learning

* grayscale → RGB
* resizing (224×224 or 227×227)
* normalization

---

## Sample Data

Included for testing:

* `sample_images/`
* `sample_labels_2class.xlsx`
* `sample_labels_4class.xlsx`

---

## Requirements

* MATLAB
* Deep Learning Toolbox
* Image Processing Toolbox

GPU is optional.

---

## Purpose

* research reproducibility
* comparison of methods
* evaluation of artifact detection pipelines
* support for academic publications

---

## License

This project is released under the MIT License.
