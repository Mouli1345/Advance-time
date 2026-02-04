Advanced Time Series Forecasting with Transformer Attention

Features

Synthetic non-stationary time series with multiple seasonalities

Transformer Encoder for sequence forecasting

Attention mechanism for interpretability

Optuna for hyperparameter optimization

Baseline comparison using SARIMA

Evaluation using RMSE, MAE, and MAPE

 Dataset

The dataset is synthetically generated and includes:

Trend component

Daily and weekly seasonality

Random noise

Exogenous variables

Lag features (lag_1, lag_2, lag_24)

This setup mimics real-world forecasting challenges like demand or energy consumption forecasting.

Model Architecture
Transformer Forecast Model

Input embedding layer

Multi-head self-attention encoder

Feed-forward output layer

Sequence-to-one prediction

Input → Linear Embedding → Transformer Encoder → Final Dense Layer → Forecast

Hyperparameter Optimization

Optuna is used to tune:

d_model

n_heads

n_layers

dropout

learning rate

The objective function minimizes Mean Squared Error (MSE) on the validation set.

 Evaluation Metrics

Both models are evaluated using:

RMSE – Root Mean Squared Error

MAE – Mean Absolute Error

MAPE – Mean Absolute Percentage Error

Results are printed for:

Transformer model

SARIMA baseline

Attention Analysis

The attention weights are analyzed to understand which timesteps influence predictions the most.

Key findings:

Recent timesteps receive higher attention

Daily seasonal patterns (lag-24) show strong influence

Confirms learning of both short-term dynamics and seasonality

Installation
pip install numpy pandas torch scikit-learn optuna statsmodels


The script automatically installs Optuna if missing.

Usage

Simply run:

python advance.py


The script will:

Generate data

Train the Transformer

Tune hyperparameters

Compare with SARIMA

Output metrics and attention insights

Project Structure
.
├── advance.py        # Main forecasting script
└── README.md         # Project documentation

Technologies Used

Python

PyTorch

Optuna

NumPy & Pandas

scikit-learn

statsmodel
