# PINN-MLP Pore Pressure Prediction

Repository: GeoHydrocarbon/PINN_MLP_Pp_Pred

## 1. Overview

This repository implements a Physics-Informed Multi-Layer Perceptron (PINN-MLP) framework for pore pressure (Pp) prediction using well-log data.

The model combines:
- A data-driven MLP
- Physics-based regularization (PINN)

to improve generalization performance, especially in data-limited scenarios such as single-well training.

---

## 2. Repository Structure

PINN_MLP_Pp_Pred/
├── train.py
├── predict.py
├── bayes_opt.py
├── config.yaml
├── models/
│ ├── mlp_phy.py
│ ├── lstm_phy.py
│ ├── registry.json
│ └── scaler.pkl
├── data/
│ └── processed/
├── checkpoints/
└── runs/


---

## 3. Data Description

### 3.1 Input Features

The following well-log features are used as model inputs:

| Feature | Description |
|--------|------------|
| Depth  | Measured depth (m) |
| AC     | Acoustic slowness (μs/ft) |
| DEN    | Bulk density (g/cm³) |
| Ip     | P-wave impedance |

### 3.2 Target Variable

- **Pp**: Pore pressure (MPa)

---

## 4. Model Description

### 4.1 MLP Backbone

The base network is a fully connected multi-layer perceptron with ReLU activation.

### 4.2 Physics-Informed Loss

The total loss function is defined as:

L = λ_data · L_data + λ_phys · L_phys

where:
- L_data is the mean squared error between predicted and measured pore pressure
- L_phys is the physics-informed regularization term

---

## 5. Configuration

Model and training parameters are defined in `config.yaml`.

Key parameters include:

```yaml
model:
  name: mlp
  hidden_dims: [64, 64, 64]

training:
  epochs: 2000
  batch_size: 256
  learning_rate: 0.001

loss:
  lambda_data: 1.0
  lambda_phys: 1.0
6. Training
Run the training script:

python train.py
Training outputs are saved in the runs/ directory.

7. Prediction
Run inference on unseen wells:

python predict.py
Predicted pore pressure curves are generated along depth.

8. Bayesian Optimization
Hyperparameters can be optimized using Bayesian optimization:

python bayes_opt.py
