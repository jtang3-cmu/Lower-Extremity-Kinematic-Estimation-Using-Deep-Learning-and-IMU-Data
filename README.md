# Lower-Extremity-Kinematic-Estimation-Using-Deep-Learning-and-IMU-Data

## Project Overview
This repository implements deep‐learning‐based estimation of lower‐extremity joint kinematics from inertial measurement unit (IMU) signals. Compared to expensive, lab‐bound motion‐capture systems, our approach leverages IMUs plus neural networks to deliver low‐cost, portable gait analysis.

We re-implement the baseline 1D-CNN from the CMU-MBL JointAnglePrediction_JOB repository, then introduce a Transformer with Rotary Positional Embedding (RoPE) for improved temporal modeling. Pre-trained weights for both CNN and Transformer models are provided—no retraining is required to reproduce our main results.
