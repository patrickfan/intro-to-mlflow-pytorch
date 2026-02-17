# MNIST Training with MLflow & PyTorch

This project provides a standardized MLOps lifecycle using **PyTorch** for model training and **MLflow** for experiment tracking, model registration, and alias-based deployment.

It serves as a reference implementation for integrating the American Science Cloud MLflow server into research workflows.

## Key Features

* **PyTorch CNN:** trains a Convolutional Neural Network on the MNIST dataset.
* **MLflow Tracking:** Automates logging of hyperparameters, loss metrics, and confusion matrix plots.
* **Automatic Tagging:** Captures Git commit hash, branch, and user info for full reproducibility.
* **Model Registry:** Automatically registers trained models to the central MLflow Model Registry.
* **Alias Management:** Demonstrates how to programmatically promote a model version to `@production` and perform inference using that alias.
* **Security:** Includes authentication patching for API Keys and SSL certificate handling.
* **Large Model Workflow:** Supports a "Zero-Copy" workflow for large models (GBs/TBs). Metadata is logged to the remote server, while heavy artifacts are stored on high-speed shared filesystems to avoid network bottlenecks during training and inference.
  
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
      
### 2. Production Run (Tracking & Register & Promote)
Train the model, register it to the Model Registry, promote it to the `@production` alias, and run an inference sanity check.

```bash
python train_mnist_mlflow.py --epochs 5 --register --phase production --note "Baseline CNN model"
```

This script implements the following workflow:

1.  **Setup:** Patches `mlflow` to inject the `X-Api-Key` header for authentication.
2.  **Train:** Standard PyTorch training loop.
3.  **Log:** Sends metrics (accuracy, loss) and artifacts (confusion matrix PNG) to the remote server.
4.  **Register (Optional):**
    * Saves the model as a new **Version** in the Model Registry.
    * Assigns the **Alias** `@production` to this new version.
    * Loads the model using the URI `models:/MNIST_CNN@production` to verify it works.
      
### 3. Register & Inference
Register the model to the Model Registry, promote it to the `@production` alias, and run an inference sanity check.

```bash
python register_and_inference.py --run_id d87bbf048599435f9798501d6ba3d3d6
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

## Large Model Workflow (Shared Filesystem)
Standard MLflow operations upload model artifacts to the tracking server. However, for large models (GBs/TBs), uploading to a remote server may fail due to timeouts, and downloading the full model for every inference job is computationally infeasible.

* This workflow solves that by decoupling metadata from storage:

* Metadata (metrics, params) is logged to the remote MLflow server.

* Heavy Artifacts (model weights) are saved directly to a high-speed shared filesystem. 

* Inference is performed via "Zero-Copy" loading, reading directly from the storage path without network transfer. 

### 1. Training with Custom Artifact Storage
Runs the training session but forces the artifacts to be stored in a specific local or shared directory (--target_path) instead of uploading them to the server.

```bash
python Modify_artifact_location.py \
  --epochs 5 \
  --phase development \
  --target_path "/lustre/orion/atm112/scratch/patrickfan/MLFlow/mlflow_artifacts"
```

### 2. Zero-Copy Register & Inference
Registers the model using the existing storage path (no file movement) and performs an inference sanity check. This script resolves the absolute path from the MLflow metadata and loads the model directly from the filesystem.

```bash
python Modify_artifact_location_register_inference.py \
  --run_id <YOUR_RUN_ID> \
  --model_name <model name>
```


## Project Structure

```text
.
├── train_mnist_mlflow.py                     # Standard training script (Logs artifacts to MLflow server)
├── register_and_inference.py                 # Standard registration & inference (Downloads model from server)
├── upload_external_model.py                  # Utility to upload an existing local model to MLflow
├── Modify_artifact_location.py               # [Large Model] Training script that saves artifacts to a local shared path 
├── Modify_artifact_location_register_inference.py # [Large Model] Zero-copy registration & inference from shared storage
└── README.md                  # Project documentation
```

## Requirements

```text
mlflow>=3.0.0
torch>=1.13.0
torchvision>=0.14.0
Python 3.8+
matplotlib
numpy
scikit-learn
urllib3
```

## Core MLflow Concepts

### 1. Experiments
High-level logical groupings of related runs. This helps separate different projects or major architectural changes.

```
mlflow.set_experiment("mnist_cnn_pytorch")
```


### 2: Runs
A single execution of training code. Each run captures hyperparameters and performance metrics.

```
#Starts a new tracking context & Tracks training configuration and performance
with mlflow.start_run():

    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_accuracy", acc, step=epoch)
```

### 3:Artifacts 
Output files logged during a run, such as plot and configuration files.

```
mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
```

### 4:Registered Models
Named entities in the Model Registry. Once a model is "Registered," it is treated as an asset that can be versioned (v1, v2, v3).

```
# Log and register the model in one step
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    registered_model_name="MNIST_CNN"
)
```

### 5:Model Aliases 
Mutable, user-defined labels (like production or champion) used to point to specific model versions.
```
# Assign an alias to a specific version
client.set_registered_model_alias(name="MNIST_CNN", alias="production", version="1")

# Load a model using its alias
model = mlflow.pytorch.load_model("models:/MNIST_CNN@production")
```



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

```sql
tags.owner = '8cf' AND tags.project.phase = 'development' AND metrics.val_accuracy > 0.95
