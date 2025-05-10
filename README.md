**Lower-Extremity Kinematic Estimation using Deep Learning and IMU Data**

This repository reproduces and extends the CMU-MBL baseline for lower-extremity joint-angle prediction from IMU signals, adding a lightweight transformer model and harmonization tools. The work builds upon the [CMU-MBL JointAnglePrediction\_JOB](https://github.com/CMU-MBL/JointAnglePrediction_JOB) baseline.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Baseline Reference](#baseline-reference)
* [Data Harmonization](#data-harmonization)
* [Baseline Training](#baseline-training)
* [Transformer Model & Weights](#transformer-model--weights)
* [Reports](#reports)
* [Installation](#installation)
* [Usage Examples](#usage-examples)
* [Citation](#citation)

---

## Project Overview

Accurate estimation of hip, knee, and ankle angles from wearable IMU data is critical for gait analysis and rehabilitation. We replicate the Conv1D baseline from Rapp et al. on synthetic and lab-collected IMU datasets, then introduce a transformer-based predictor with Rotary Positional Embeddings.

Key contributions:

* **Reproduced** the Conv1D baseline (CNN) from the CMU-MBL project.
* **Implemented** a transformer architecture (`pure_transformer.py`) for sequential modeling.
* **Pretrained weights** for both CNN and transformer models.
* **Data harmonization** scripts to merge CMU-AMASS, Calgary, and lab IMU sources.

---

## Repository Structure

```
├── Data_Harmonization/      # Scripts to preprocess and align IMU and MoCap streams
├── Train_baseline/          # Baseline CNN training code and configs
├── trained_transformer/     # Transformer implementation, architecture, and pretrained weights
├── Reports/                 # Report files (PDF)
└── README.md                # Project overview and instructions
```

---

## Baseline Reference

Training and evaluation scripts follow the structure of the [JointAnglePrediction\_JOB](https://github.com/CMU-MBL/JointAnglePrediction_JOB) repository. We adapt their:

* Data loaders and preprocessing routines
* Conv1D architecture and hyperparameters
* Training loop and evaluation metrics

---

## Data Harmonization

In `Data_Harmonization/`, use the provided scripts to:

1. **Resample** all sensor streams to a common frequency
2. **Synchronize** IMU and MoCap timelines
3. **Align** sensor axes across datasets
4. **Equalize** noise profiles

Example:

```bash
python Data_Harmonization/preprocess.py \
  --input amass_dir calgary_dir lab_dir \
  --output processed_data --freq 100
```

---

## Baseline Training

In `Train_baseline/`, run:

```bash
python train_baseline.py \
  --data_dir processed_data --epochs 50 --batch_size 32 \
  --out cnn_baseline_weights.pt
```

This reproduces the CNN baseline results reported in our study.

---

## Transformer Model & Weights

The `trained_transformer/` folder contains:

* `pure_transformer.py`: transformer model with four encoder layers, six attention heads, and RoPE.
* `train_transformer.py`: training script mirroring the baseline interface.
* `transformer_weights.pt`: pretrained transformer weights.

To evaluate:

```bash
python trained_transformer/train_transformer.py \
  --data_dir processed_data --out transformer_weights.pt
```

---

## Reports

All written analyses and figures are in `Reports/`. Open `Reports/Report.pdf` for detailed methodology and results.

---

## Installation

Clone and install dependencies:

```bash
git clone https://github.com/jtang3-cmu/Lower-Extremity-Kinematic-Estimation-Using-Deep-Learning-and-IMU-Data.git
cd Lower-Extremity-Kinematic-Estimation-Using-Deep-Learning-and-IMU-Data
pip install -r requirements.txt
```

---

## Usage Examples

After training or loading pretrained weights:

```python
from models.pure_transformer import TransformerModel
import torch
model = TransformerModel(...)
model.load_state_dict(torch.load('trained_transformer/transformer_weights.pt'))
# run evaluation on test set
```

---

## Citation

Tang, J., Chen, C.-Y., Kwon, M., Mahajan, A. "Kinematic Estimation Using Deep Learning and IMU Data" (2025).

Baseline: Rapp, E., Shin, S., Thomsen, W., Ferber, R., Halilaj, E. (2021). Estimation of kinematics from inertial measurement units using a combined deep learning and optimization framework, *Journal of Biomechanics*, 116, 110229.
