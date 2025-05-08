import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def resample_signal(timestamps: np.ndarray, data: np.ndarray, target_freq: float) -> (np.ndarray, np.ndarray):
    """
    Resample a signal to the target frequency using linear interpolation.

    Args:
        timestamps: Original time axis (seconds).
        data: Array of shape (N, D) where D is number of channels.
        target_freq: Desired sampling frequency in Hz.

    Returns:
        new_times: Uniformly spaced time axis at target_freq.
        new_data: Resampled data of shape (M, D).
    """
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(np.ceil(duration * target_freq)) + 1
    new_times = np.linspace(timestamps[0], timestamps[-1], num=num_samples)
    interp_func = interp1d(timestamps, data, axis=0, kind='linear', fill_value='extrapolate')
    new_data = interp_func(new_times)
    return new_times, new_data


def time_synchronize(imu_times: np.ndarray, imu_signal: np.ndarray,
                     mocap_times: np.ndarray, mocap_signal: np.ndarray,
                     hop_peaks: int = 3) -> np.ndarray:
    """
    Align IMU timeline to MoCap by detecting calibration hop peaks and shifting.

    Returns shifted IMU timestamps to match MoCap.
    """
    imu_peaks = np.argsort(imu_signal)[-hop_peaks:]
    mocap_peaks = np.argsort(mocap_signal)[-hop_peaks:]
    t_offset = imu_times[imu_peaks[0]] - mocap_times[mocap_peaks[0]]
    return imu_times - t_offset


def apply_orientation_transform(data: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply orientation alignment using provided 3x3 transform matrices.
    """
    if data.ndim == 2 and data.shape[1] % 3 == 0:
        num_sensors = data.shape[1] // 3
        out = np.zeros_like(data)
        for i in range(num_sensors):
            block = data[:, i*3:(i+1)*3]
            out[:, i*3:(i+1)*3] = block.dot(transform.T)
        return out
    return data.dot(transform.T)


def harmonize_dataset(input_folder: str,
                      output_folder: str,
                      target_freq: float,
                      orientation_matrices: dict):
    """
    Process all CSVs in input, apply resampling, sync, orientation, and save processed CSVs.
    """
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(input_folder, fname))
        t = df['time'].values
        data = df.drop(columns=['time']).values

        # Resample
        t_res, data_res = resample_signal(t, data, target_freq)

        # Orientation: choose sensor-specific or default
        transform = orientation_matrices.get(fname.rstrip('.csv'),
                                             orientation_matrices.get('default', np.eye(3)))
        data_oriented = apply_orientation_transform(data_res, transform)

        out_df = pd.DataFrame(data_oriented, columns=df.columns.drop('time'))
        out_df.insert(0, 'time', t_res)
        out_df.to_csv(os.path.join(output_folder, fname), index=False)
        print(f"Processed and saved: {fname}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Harmonize IMU/MoCap datasets: resampling, sync, orientation.'
    )
    parser.add_argument('--input', '-i', required=True, help='Input folder of CSV files')
    parser.add_argument('--output', '-o', required=True, help='Output folder for harmonized CSVs')
    parser.add_argument('--freq', '-f', type=float, default=100.0, help='Target sampling frequency (Hz)')
    parser.add_argument('--orientation-file', '-m', 
                        help='Path to .npz file containing orientation matrices mapping sensor names to 3x3 arrays')
    args = parser.parse_args()

    # Load orientations
    if args.orientation_file:
        npz = np.load(args.orientation_file)
        orientation_matrices = {k: npz[k] for k in npz.files}
    else:
        # Default: identity
        orientation_matrices = {'default': np.eye(3)}

    harmonize_dataset(args.input, args.output, args.freq, orientation_matrices)
