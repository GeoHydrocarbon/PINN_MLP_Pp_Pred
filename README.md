PINN-MLP Pore Pressure Prediction

📦 Repository: GeoHydrocarbon/PINN\_MLP\_Pp\_Pred

🎯 Task: Well-log–based pore pressure (Pp) prediction using Physics-Informed Neural Network (PINN)–enhanced MLP

1\. Overview

This repository implements a Physics-Informed Multi-Layer Perceptron (PINN-MLP) framework for pore pressure (Pp) prediction from well-logging data.

Unlike purely data-driven machine learning models, this framework introduces physics-based constraints into the training process to regularize the neural network and improve generalization, especially in data-limited scenarios (e.g., single-well training).

The workflow is designed for:

Training on one reference well

Predicting pore pressure in unseen wells

Optional Bayesian optimization of hyperparameters

2\. Key Features

✅ MLP-based supervised pore pressure prediction

✅ Physics-informed loss term (PINN regularization)

✅ Single-well training → multi-well prediction

✅ Feature standardization with persistent scaler

✅ Bayesian optimization for hyperparameter tuning

✅ Publication-ready visualization outputs

3\. Repository Structure

PINN\_MLP\_Pp\_Pred/

├── train.py # Main training script

├── predict.py # Prediction / inference script

├── bayes\_opt.py # Bayesian hyperparameter optimization

├── config.yaml # Central configuration file

│

├── models/

│ ├── mlp\_phy.py # Physics-informed MLP model

│ ├── lstm\_phy.py # (Optional) Physics-informed LSTM (not used by default)

│ ├── registry.json # Model registry

│ └── scaler.pkl # Saved feature scaler

│

├── data/

│ └── processed/

│ ├── N873\_fortrain\_Ip.csv # Training well

│ ├── N871\_forpred.csv # Prediction well

│ └── \*\_filtered.csv # Filtered prediction data

│

├── checkpoints/

│ └── best\_mlp\_model.pth # Best trained model

│

└── runs/

 └── 20251201\_212542\_N873\_fortrain\_Ip\_mlp/

 ├── best\_model.pth

 ├── scaler.pkl

 ├── config.yaml

 ├── actual\_vs\_pred.png

 └── pred\_vs\_target\_depth.png

4\. Data Format

4.1 Input Features

The model uses four logging-based input features:

Feature Description

Depth Measured depth (m)

AC Acoustic slowness (μs/ft)

DEN Bulk density (g/cm³)

Ip P-wave impedance

These are defined in config.yaml:

input\_features:

\- Depth

\- AC

\- DEN

\- Ip

4.2 Target Variable

target: Pp

Pp = pore pressure (MPa)

5\. Model Architecture

5.1 MLP Backbone

Defined in models/mlp\_phy.py:

Fully connected feed-forward network

Configurable hidden layers

ReLU activation

Output: scalar pore pressure prediction

MLP(input\_dim=4, hidden\_dims=\[64, 64, 64\], output\_dim=1)

5.2 Physics-Informed Constraint (PINN)

The total loss is:

𝐿=𝜆data⋅𝐿data+𝜆phys⋅physics

Where:

Data loss: Mean Squared Error (MSE) between predicted and measured Pp

Physics loss: Regularization term enforcing physically reasonable gradients with respect to depth

This is implemented directly in train.py.

6\. Configuration (config.yaml)

Key parameters:

model:

 name: mlp

 hidden\_dims: \[64, 64, 64\]

training:

 epochs: 2000

 batch\_size: 256

 learning\_rate: 0.001

 weight\_decay: 0.0

loss:

 lambda\_data: 1.0

 lambda\_phys: 1.0

device: cuda

7\. Training

7.1 Run Training

python train.py

The script will:

Load training data

Standardize features (saved as scaler.pkl)

Train PINN-MLP

Save best model to checkpoints/

Generate prediction figures

7.2 Outputs

Saved under runs/&lt;experiment\_name&gt;/:

best\_model.pth – trained network

actual\_vs\_pred.png – Pp comparison

pred\_vs\_target\_depth.png – depth-domain visualization

config.yaml – snapshot of configuration

scaler.pkl – feature scaler

8\. Prediction / Inference

python predict.py

The script:

Loads trained model + scaler

Predicts pore pressure for unseen wells

Outputs depth-aligned prediction curves

9\. Bayesian Hyperparameter Optimization

python bayes\_opt.py

Optimized parameters include:

Learning rate

Hidden layer size

Physics-loss weight (lambda\_phys)

Objective:

Minimize validation RMSE

10\. Scientific Significance

This framework is particularly suitable for:

Single-well training scenarios

Physics-constrained ML in geophysics

Seismic–well integration workflows

(features chosen to be compatible with impedance-based inversion)
