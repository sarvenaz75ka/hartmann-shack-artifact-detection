# Hartmann–Shack Artifact Detection

This repository provides a complete framework for detecting artifacts in Hartmann–Shack (HS) wavefront sensor images using both classical image processing methods and deep learning approaches.

The repository includes:

* binary classification methods for **valid / invalid** HS images
* multiclass classification methods for **blinking / corneal / laser reflex / valid**
* folder-based inference scripts for easy use
* optional evaluation using Excel label files

---

## Repository Structure

```text
hartmann-shack-artifact-detection/
│
├── sample_images/                  # Example HS images
├── sample_labels_2class.xlsx       # Ground-truth labels for binary classification
├── sample_labels_4class.xlsx       # Ground-truth labels for multiclass classification
│
├── classical_methods/              # Classical image-processing approaches
│   ├── morphological_artifact_detection.m
│   ├── correlation_artifact_detection.m
│   └── reference/
│       └── meanImage_weighted_1024_2.bmp
│
├── mobilenet/                      # MobileNet inference scripts and trained models
├── shufflenet/                     # ShuffleNet inference scripts and trained models
├── squeezenet/                     # SqueezeNet inference scripts and trained models
│
└── README.md
```

---

## Methods

### 1. Classical Methods

The repository includes two classical approaches for artifact detection in HS images:

* **Morphological-based detection**

  * pupil detection
  * pupil circularity validation
  * pupil size validation
  * corneal reflection detection

* **Correlation-based detection**

  * pupil localization
  * pupil centering
  * image correlation with a reference mean image
  * reflection-based rejection

### 2. Deep Learning Methods

The repository includes deep learning inference pipelines based on:

* **MobileNet**
* **ShuffleNet**
* **SqueezeNet**

Each architecture is provided for:

* **binary classification**

  * `valid`
  * `invalid`

* **multiclass classification**

  * `blinking`
  * `corneal`
  * `laser reflex`
  * `valid`

---

## Note on Class Labels

The class **"laser reflex"** used in the dataset corresponds to **lens reflection** in the manuscript.

Both terms refer to the same type of optical artifact.

---

## Supported Tasks

### Binary Classification

Binary classification scripts separate images into:

* `valid`
* `invalid`

### Multiclass Classification

Multiclass classification scripts separate images into:

* `blinking`
* `corneal`
* `laser reflex`
* `valid`

---

## Input Data

### Images

Images should be placed inside an input folder such as:

```text
sample_images/
```

Supported image formats include:

* `.bmp`
* `.png`
* `.jpg`
* `.jpeg`
* `.tif`
* `.tiff`

### Label Files

Ground-truth labels are optional. If provided, the scripts compute performance metrics.

The Excel label file must contain two columns:

```text
filename | label
```

#### Binary Example

```text
filename | label
img1.bmp | 1
img2.bmp | 0
```

or equivalently:

```text
filename | label
img1.bmp | valid
img2.bmp | invalid
```

#### Multiclass Example

```text
filename | label
img1.bmp | blinking
img2.bmp | corneal
img3.bmp | laser reflex
img4.bmp | valid
```

Notes:

* `filename` must exactly match the image filename, including the extension.
* Labels may be numeric or text, depending on the script.
* For multiclass scripts, text labels must exactly match the training labels.

---

## Example Usage

### Binary Classification Example

```matlab
mobilenet_2class_inference( ...
    'sample_images', ...
    'results', ...
    'mobilenet_binary_trained.mat', ...
    'sample_labels_2class.xlsx')
```

### Multiclass Classification Example

```matlab
shufflenet_4class_inference( ...
    'sample_images', ...
    'results', ...
    'shufflenet_4class_trained.mat', ...
    'sample_labels_4class.xlsx')
```

You may also run the scripts without a label file. In that case, the code performs inference only and skips performance evaluation.

---

## Output

After execution, results are saved in a user-defined output folder.

### Binary Classification Output

```text
results/
├── valid/
├── invalid/
└── prediction_results_YYYYMMDD_HHMMSS.xlsx
```

### Multiclass Classification Output

```text
results/
├── blinking/
├── corneal/
├── laser reflex/
├── valid/
└── prediction_results_YYYYMMDD_HHMMSS.xlsx
```

The output Excel file may include:

* image name
* predicted class name
* predicted label or label ID
* confidence score
* elapsed processing time
* optional ground-truth label

---

## Evaluation Metrics

If a label file is provided, the scripts compute quantitative evaluation metrics.

### Binary Classification Metrics

* Accuracy
* Sensitivity (Recall)
* Specificity
* Precision
* F1-score
* Confusion matrix

### Multiclass Classification Metrics

* Overall accuracy
* Confusion matrix
* Per-class precision
* Per-class recall
* Per-class F1-score
* Macro-averaged precision
* Macro-averaged recall
* Macro-averaged F1-score

---

## Preprocessing

### Classical Methods

Preprocessing may include:

* grayscale conversion
* morphological opening
* binarization
* hole filling
* connected-component analysis
* pupil-based region-of-interest selection

### Deep Learning Methods

Depending on the architecture, preprocessing may include:

* grayscale to RGB conversion
* removal of extra channels if present
* resizing according to network input size
* normalization

Typical input resolutions:

* **MobileNet**: typically `224 × 224`
* **ShuffleNet**: typically `224 × 224`
* **SqueezeNet**: typically `227 × 227`

For SqueezeNet, ImageNet-style channel normalization is applied to match the training pipeline.

---

## Sample Data

A small set of example Hartmann–Shack images is included in:

```text
sample_images/
```

The corresponding ground-truth label files are:

* `sample_labels_2class.xlsx`
* `sample_labels_4class.xlsx`

These files can be used to test the inference scripts and verify the full pipeline.

---

## Requirements

The code was developed in MATLAB and requires:

* MATLAB
* Deep Learning Toolbox
* Image Processing Toolbox

Depending on your workflow, GPU support may improve performance, but inference can also be executed on CPU.

---

## Purpose

This repository is intended for:

* research reproducibility
* comparison between classical and deep learning methods
* practical testing of trained artifact-detection models
* supporting material for academic publications

---

## Author

**Sarvenaz Kalantarinejad**

---

## License

This repository is intended for academic and research use.
