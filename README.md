# PINN_MLP_Pp_Pred  
Physics-Informed Neural Network and MLP for Pore Pressure Prediction Using Eaton’s Equation  

This repository provides a hybrid **MLP + Physics-Informed Neural Network (PINN)** framework for predicting pore pressure (Pp) from well logs by embedding **Eaton’s effective stress relationship** as a physical constraint.

The method combines:

• Data-driven regression via multilayer perceptron (MLP)  
• Physics-based regularization using Eaton’s pore pressure formulation  
• Bayesian hyperparameter optimization  

This approach improves prediction robustness in scenarios with sparse pressure measurements, which are common in subsurface engineering.

---

## 🔬 Method Overview

The neural network learns the mapping:

\[
\hat{P}_p = f_\theta(z, AC, \rho, I_p)
\]

where:

- \(z\): depth  
- \(AC\): acoustic slowness  
- \(\rho\): density  
- \(I_p\): P-impedance  

### Physics-Informed Constraint (Eaton Equation)

Eaton’s pore pressure model is expressed as:

\[
P_p = P_o - (P_o - P_n) \left(\frac{X}{X_n}\right)^m
\]

where:

- \(P_o\): overburden stress  
- \(P_n\): normal hydrostatic pressure  
- \(X\): observed log value (e.g., sonic slowness)  
- \(X_n\): normal compaction trend (NCT)  
- \(m\): Eaton exponent  

The PINN loss function becomes:

\[
\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{Eaton}
\]

forcing the neural network to honor geomechanical consistency while learning from data.

---

## 📁 Repository Structure
PINN_MLP_Pp_Pred/
│
├── data/
│ └── processed/ # Preprocessed well log & pressure datasets
│
├── models/ # Neural network architectures
│
├── utils/ # Physics loss & helper functions
│
├── checkpoints/ # Trained model weights
│
├── runs/ # Training logs & metrics
│
├── bayes_opt.py # Bayesian hyperparameter tuning
├── train.py # Model training
├── predict.py # Inference and prediction
├── config.yaml # Model and data configuration
└── README.md

---

## 📥 Input & Output

### Input Features

Default input vector:
[Depth, Acoustic Slowness, Density, P-impedance]
### Output

Predicted pore pressure (Pp)


Feature combinations can be modified in `config.yaml`.

---

## ⚙️ Installation

```bash
git clone https://github.com/GeoHydrocarbon/PINN_MLP_Pp_Pred.git
cd PINN_MLP_Pp_Pred

pip install -r requirements.txt
Recommended environment:

Python ≥ 3.9

PyTorch ≥ 2.0

🧠 Training
Basic training:

python train.py
Using custom config:

python train.py --config config.yaml
🔎 Prediction
python predict.py --model checkpoints/best_model.pt
📈 Bayesian Optimization
To automatically tune hyperparameters:

python bayes_opt.py
Optimized parameters include:

Learning rate

Network depth and width

Physics loss weight λ

Eaton exponent (optional)

🛢 Applications
Abnormal pore pressure prediction

Drilling safety and well planning

Geomechanical modeling

Data-scarce basin analysis

📚 Citation
If this repository contributes to your research, please cite:

@misc{Chen2026PINNPressure,
  title={Physics-Informed Neural Network for Pore Pressure Prediction Using Eaton’s Equation},
  author={Chen, Junlin},
  year={2026},
  note={GitHub repository}
}
🤝 Contributing
Contributions are welcome:

New physics constraints

Additional well log features

Improved optimization strategies

Please open an issue or submit a pull request.

📬 Contact
Junlin Chen
Geophysics & AI for Energy
GitHub: https://github.com/GeoHydrocarbon



