**Lower-Extremity Kinematic Estimation using Deep Learning and IMU Data**

This repository implements and extends the baseline joint-angle prediction framework from the CMU-MBL JointAnglePrediction\_JOB project ([https://github.com/CMU-MBL/JointAnglePrediction\_JOB](https://github.com/CMU-MBL/JointAnglePrediction_JOB)). We provide both the original CNN baseline and a transformer-based model, along with pretrained weights and scripts for data preprocessing, training, fine-tuning, and evaluation.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Baseline Reference](#baseline-reference)
* [Pretrained Weights](#pretrained-weights)
* [Installation](#installation)
* [Data Preparation & Harmonization](#data-preparation--harmonization)
* [Training & Fine-Tuning](#training--fine-tuning)
* [Model Architectures](#model-architectures)
* [Evaluation](#evaluation)
* [Usage Examples](#usage-examples)
* [Citation](#citation)

---

## Project Overview

Accurate lower-limb kinematic estimation is essential for gait analysis, rehabilitation, and sports biomechanics. Building on the CNN + optimization framework of Rapp et al., we reproduce the Conv1D baseline and develop a lightweight transformer model (with Rotary Positional Embedding) to predict hip, knee, and ankle joint angles from raw IMU signals. Our contributions include:

* **Reproduction** of the Conv1D baseline (Rapp et al.) on Calgary synthetic IMU data.
* **Transformer implementation** (`pure_transformer.py`) with four encoder layers, six attention heads, RoPE, and feedforward blocks.
* **Data harmonization pipeline**: resampling, time synchronization, axis alignment, noise equalization across CMU-AMASS, Calgary, and lab-collected datasets.
* **Fine-tuning scripts** (`finetune.ipynb`) for adapting the transformer to new datasets.
* **Pretrained weights** for both baseline CNN and transformer models.

---

## Features

* **Baseline CNN**: Conv1D architecture matching published performance.
* **Transformer**: Self-attention with RoPE for temporal-spatial modeling.
* **Harmonization**: Unified pipeline to merge synthetic and real IMU sources.
* **Pretrained Weights**: Ready-to-use weights for immediate evaluation.
* **Training Scripts**: Compatible with CMU-MBL baseline; see [Baseline Reference](#baseline-reference).
* **Evaluation Tools**: RMSE computation and plotting utilities.

---

## Repository Structure

```bash
├── data/
│   ├── amass/            # Preprocessed CMU-AMASS synthetic data
│   ├── calgary/          # Calgary Running Clinic synthetic IMU data
│   └── lab/              # Real IMU data (Halilaj lab)
│
├── models/
│   ├── baseline_cnn.py   # Conv1D baseline implementation
│   └── pure_transformer.py # Transformer model with RoPE
│
├── weights/
│   ├── cnn_baseline.pt   # Pretrained CNN weights
│   └── transformer.pt    # Pretrained transformer weights
│
├── scripts/
│   ├── preprocess.py     # Harmonization pipeline functions
│   ├── train.py          # Training wrapper (CNN & transformer)
│   └── finetune.ipynb    # Notebook for fine-tuning on new data
│
├── eval/
│   ├── evaluate.py       # RMSE and plotting utilities
│   └── plots/            # Example result figures
│
└── README.md             # This document
```

---

## Baseline Reference

Our training and evaluation scripts are built upon the CMU-MBL JointAnglePrediction\_JOB repository. You can refer to their data handling and training scripts here:

* [https://github.com/CMU-MBL/JointAnglePrediction\_JOB/tree/master](https://github.com/CMU-MBL/JointAnglePrediction_JOB/tree/master)

We adapt their training loops, hyperparameter settings, and dataset splits for both CNN and transformer models.

---

## Pretrained Weights

We provide pretrained weights for easy evaluation:

* **CNN Baseline**: `weights/cnn_baseline.pt`
* **Transformer**: `weights/transformer.pt`

Load these weights in your own scripts to reproduce the results shown in the report.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jtang3-cmu/Lower-Extremity-Kinematic-Estimation-Using-Deep-Learning-and-IMU-Data.git
   cd Lower-Extremity-Kinematic-Estimation-Using-Deep-Learning-and-IMU-Data
   ```

2. Install dependencies (recommend using a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

Required packages include PyTorch, NumPy, SciPy, pandas, and matplotlib.

---

## Data Preparation & Harmonization

Use `scripts/preprocess.py` to:

1. **Resample** all streams to a common frequency.
2. **Synchronize** IMU and MoCap timelines via calibration peaks.
3. **Align** sensor axes across datasets.
4. **Equalize** noise characteristics.

Example usage:

```bash
python scripts/preprocess.py \
  --input_dirs data/calgary data/amass data/lab \
  --output_dir data/processed --freq 100
```

---

## Training & Fine-Tuning

### Baseline Training

Referencing the baseline repo, launch:

```bash
python scripts/train.py \
  --model cnn --data_dir data/processed --epochs 50 --batch_size 32 \
  --out weights/cnn_baseline.pt
```

### Transformer Training

```bash
python scripts/train.py \
  --model transformer --data_dir data/processed --epochs 50 --batch_size 32 \
  --out weights/transformer.pt
```

### Fine-Tuning

Open `scripts/finetune.ipynb` to run fine-tuning on external datasets. It loads `weights/transformer.pt`, performs harmonization, and retrains on a subset of new data.

---

## Model Architectures

* **Baseline CNN**: 1D convolutional layers with padding, hyperparameter search via Hyperopt.
* **Transformer**: 4 layers, 6 attention heads, RoPE, feedforward dimension 256, dropout 0.1. Implemented in `models/pure_transformer.py`.

---

## Evaluation

Use `eval/evaluate.py` to compute RMSE over held-out test sets and generate plots:

```bash
python eval/evaluate.py --model transformer --weights weights/transformer.pt --data_dir data/processed
```

---

## Usage Examples

See `eval/plots/` for example figures comparing ground truth vs. predictions and RMSE bar charts.

---

## Citation

If you use this code, please cite:

Tang, J., Chen, C.-Y., Kwon, M., Mahajan, A. "Kinematic Estimation Using Deep Learning and IMU Data" (2025).

Baseline: Rapp, E., Shin, S., Thomsen, W., Ferber, R., Halilaj, E. (2021). *Estimation of kinematics from inertial measurement units using a combined deep learning and optimization framework*, Journal of Biomechanics 116, 110229.

---

For questions or issues, contact Jonathan Tang ([jtang3@andrew.cmu.edu](mailto:jtang3@andrew.cmu.edu)).
