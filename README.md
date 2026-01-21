# MNIST Training with MLflow & PyTorch

This project demonstrates a complete MLOps lifecycle using **PyTorch** for model training and **MLflow** for experiment tracking, model registration, and alias-based deployment.

It serves as a reference implementation for integrating the American Science Cloud MLflow server into research workflows.

## Key Features

* **PyTorch CNN:** trains a Convolutional Neural Network on the MNIST dataset.
* **MLflow Tracking:** Automates logging of hyperparameters, loss metrics, and confusion matrix plots.
* **Automatic Tagging:** Captures Git commit hash, branch, and user info for full reproducibility.
* **Model Registry:** Automatically registers trained models to the central MLflow Model Registry.
* **Alias Management:** Demonstrates how to programmatically promote a model version to `@production` and perform inference using that alias.
* **Security:** Includes authentication patching for API Keys and SSL certificate handling.

## Prerequisites

* Access to the MLflow Tracking Server
* An API Key for authentication


## Configuration

Before running the script, you must export your API key as an environment variable. The script expects the key in `AM_SC_API_KEY`.

**Linux/Mac:**

export AM_SC_API_KEY="your-secret-token-here"

**Windows (PowerShell):**

$env:AM_SC_API_KEY="your-secret-token-here"

## Usage

### 1. Development Run (Tracking Only)
Run a simple training session. This logs metrics and params but **does not** register the model.

```bash
python train_mnist_mlflow.py --epochs 5 --phase development --register
```

### 2. Production Run (Register & Promote)
Train the model, register it to the Model Registry, promote it to the `@production` alias, and run an inference sanity check.

```bash
python train_mnist_mlflow_v2.py --register --epochs 5 --phase production --note "Baseline CNN model"
```

### Available Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--batch_size` | 64 | Batch size for training. |
| `--epochs` | 1 | Number of training epochs. |
| `--lr` | 0.001 | Learning rate (Adam optimizer). |
| `--phase` | development | Tag for project phase (`development`, `tuning`, `production`). |
| `--register` | False | **Flag.** If present, registers model and promotes to `@production`. |
| `--note` | "" | Optional text note to attach to the run. |


##  MLOps Workflow Explained

This script implements the following workflow:

1.  **Setup:** Patches `mlflow` to inject the `X-Api-Key` header for authentication.
2.  **Train:** Standard PyTorch training loop.
3.  **Log:** Sends metrics (accuracy, loss) and artifacts (confusion matrix PNG) to the remote server.
4.  **Register (Optional):**
    * Saves the model as a new **Version** in the Model Registry.
    * Assigns the **Alias** `@production` to this new version.
    * Loads the model using the URI `models:/MNIST_CNN@production` to verify it works.

## Project Structure

```text
.
├── train_mnist_mlflow_v2.py   # Main training script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Requirements

```text
mlflow>=3.0.0
torch>=1.13.0
torchvision>=0.14.0
matplotlib
numpy
scikit-learn
urllib3
