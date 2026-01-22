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
python train_mnist_mlflow.py --epochs 5 --phase development
```

### 2. Production Run (Register & Promote)
Train the model, register it to the Model Registry, promote it to the `@production` alias, and run an inference sanity check.

```bash
python train_mnist_mlflow.py --register --epochs 5 --phase production --note "Baseline CNN model"
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
```

## 🧠 Core MLflow Concepts

### 1. Experiments
High-level logical groupings of related runs. This helps separate different projects or major architectural changes.

mlflow.set_experiment("mnist_cnn_pytorch")


2. **Runs**: A single execution of training code. Each run captures hyperparameters and performance metrics.

   # Starts a new tracking context & Tracks training configuration and performance
   with mlflow.start_run():
       mlflow.log_param("batch_size", batch_size)
       mlflow.log_param("epochs", epochs)
       mlflow.log_param("learning_rate", lr)
       mlflow.log_param("optimizer", "Adam")

       mlflow.log_metric("train_loss", train_loss, step=epoch)
       mlflow.log_metric("val_accuracy", acc, step=epoch)




## 🔎 How to Search Runs in MLflow

You can filter experiments in the MLflow UI using a SQL-like syntax. Here are the specific search patterns relevant to this project.

### 1. Filtering by Tags (Metadata)
Use `tags.<key>` to filter by the organizational data we added.

| Goal | Syntax | Example |
| :--- | :--- | :--- |
| **Project Phase** | `tags.project.phase = '...'` | `tags.project.phase = 'production'` |
| **Owner** | `tags.owner = '...'` | `tags.owner = '8cf'` |
| **Model Type** | `tags.model.type = '...'` | `tags.model.type = 'Computer Vision'` |
| **Git Commit** | `tags.git.commit = '...'` | `tags.git.commit = 'a1b2c3d'` |

### 2. Filtering by Metrics (Performance)
Use `metrics.<name>` to find your best-performing models.

| Goal | Syntax | Example |
| :--- | :--- | :--- |
| **High Accuracy** | `metrics.val_accuracy > ...` | `metrics.val_accuracy > 0.95` |
| **Low Loss** | `metrics.train_loss < ...` | `metrics.train_loss < 0.1` |

### 3. Filtering by Parameters (Configuration)
Use `params.<name>` to find runs with specific hyperparameters.

| Goal | Syntax | Example |
| :--- | :--- | :--- |
| **Batch Size** | `params.batch_size = '...'` | `params.batch_size = '64'` |
| **Optimizer** | `params.optimizer = '...'` | `params.optimizer = 'Adam'` |

### 4. Searching by Attributes (System Fields)
Attributes are built-in MLflow fields (not custom tags) that track the run's metadata.

| Goal | Syntax | Example |
| :--- | :--- | :--- |
| **Run Name** | `attributes.run_name = '...'` | `attributes.run_name = 'MNIST_production_8cf'` |
| **Run Name (Partial)** | `attributes.run_name LIKE '...'` | `attributes.run_name LIKE 'MNIST%'` |
| **Run Status** | `attributes.status = '...'` | `attributes.status = 'FINISHED'` (or `FAILED`) |
| **Time Created** | `attributes.start_time > ...` | `attributes.start_time > 1672531200000` (Unix Timestamp) |

### 5. Complex Queries (Advanced)
You can combine multiple filters using `AND`.

**Example: Find the best production candidate by Ming**
```sql
tags.owner = '8cf' AND tags.project.phase = 'development' AND metrics.val_accuracy > 0.95
