# Data Harmonization Toolkit

This folder contains tools to harmonize multi-sensor datasets (e.g., IMU and MoCap) by resampling, time-synchronizing, and applying orientation transformations. It also provides an interactive Jupyter notebook for experimentally aligning sensor axes and extracting the corresponding transform matrices.

## Folder Structure

```
Data_Harmonization/
├── data_harmonization.py      # Command-line script for batch processing CSV files
├── manual_alignment.ipynb     # Jupyter notebook for manual/experimental axis alignment
└── transforms.npz             # (Optional) Precomputed orientation matrices
```

## Requirements

* Python 3.7+
* NumPy
* SciPy
* pandas
* Jupyter Notebook (for `manual_alignment.ipynb`)

Install dependencies via pip:

```bash
pip install numpy scipy pandas jupyter
```

## `data_harmonization.py`

A CLI tool to process all CSV files in an input directory:

```bash
python data_harmonization.py \
  --input PATH/TO/RAW_CSVs \
  --output PATH/TO/PROCESSED_CSVs \
  --freq 100 \
  [--orientation-file PATH/TO/transforms.npz]
```

* `--input` / `-i`: folder of raw sensor CSVs (each with a `time` column).
* `--output` / `-o`: folder where harmonized CSVs will be saved.
* `--freq` / `-f`: target sampling frequency (Hz).
* `--orientation-file` / `-m`: optional `.npz` file mapping sensor names (or `default`) to 3×3 rotation matrices. If omitted, identity (`I₃`) is used.

### Example: Creating `transforms.npz`

```python
import numpy as np
# Define per-sensor rotation matrices
transforms = {
    'sensor1': np.array([[1,0,0],[0,0,-1],[0,1,0]]),
    'sensor2': np.array([[0,-1,0],[1,0,0],[0,0,1]]),
    'default': np.eye(3),
}
np.savez('transforms.npz', **transforms)
```

## `manual_alignment.ipynb`

Use this notebook to **manually align IMU axes** by plotting raw signals and experimenting with only axis swaps and sign flips until they match reference motion patterns or MoCap data. Follow these steps:

1. **Load sample IMU data**:

   * Read a CSV for one sensor triplet (columns: `time`, `acc_x`, `acc_y`, `acc_z`, etc.).
2. **Visualize raw axes**:

   * Plot each of the three accelerometer or gyroscope channels against time.
3. **Experiment with rotations**:

   * Apply axis **swaps** (e.g., swap X↔Y) and/or **sign flips** (multiply an axis by –1).
   * Re-plot after each adjustment to see which combination best aligns with known movements (or MoCap axes).
4. **Quantify alignment** *(optional)*:

   * Compute correlation or RMSE between IMU axes and reference signals to select the optimal transform.
5. **Record the transform matrix**:

   * For the chosen combination of swaps/flips, assemble a 3×3 rotation matrix.
   * Example: swapping X↔Y and flipping Z gives

     ```python
     R = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]])
     ```
6. **Save to `transforms.npz`**:

   * Add an entry under the sensor’s key:

     ```python
     np.savez('transforms.npz', sensor1=R, default=np.eye(3))
     ```

Once each sensor’s transform matrix is obtained, run `data_harmonization.py` to batch-process all files using:

```bash
python data_harmonization.py \
  --input raw_data/ \
  --output processed_data/ \
  --freq 200 \
  --orientation-file transforms.npz
```

---

Feel free to extend with YAML/JSON support or automate axis calibration based on statistical metrics.
