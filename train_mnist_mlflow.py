import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient  # Added for Alias management
import os
import subprocess
import getpass
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# --- FIX FOR DATASET DOWNLOAD ---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --------------------------------------------------
# 1. MLflow Connection Setup
# --------------------------------------------------
AMSC_API_KEY_ENV = "AM_SC_API_KEY"   # token lives here

if AMSC_API_KEY_ENV not in os.environ:
    print(f"Warning: {AMSC_API_KEY_ENV} not found in environment.")

# This tells MLflow to ignore SSL certificate errors (common with self-signed internal servers).
# Without this, the connection to 'mlflow.american-science-cloud.org' will fail.
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
urllib3.disable_warnings(InsecureRequestWarning)

# Inject X-Api-Key into all MLflow REST calls
def enable_amsc_x_api_key():

    """
    MLFLOW AUTHENTICATION HELPER:
    Standard MLflow does not automatically inject custom headers like 'X-Api-Key'.
    We patch the http_request function to ensure every request to the server 
    includes our security token.
    """

    import mlflow.utils.rest_utils as rest_utils

    if AMSC_API_KEY_ENV in os.environ:
        api_key = os.environ[AMSC_API_KEY_ENV]
        _orig = rest_utils.http_request

        def patched(host_creds, endpoint, method, *args, **kwargs):
            if "headers" in kwargs and kwargs["headers"] is not None:
                h = dict(kwargs["headers"])
                h["X-Api-Key"] = api_key
                kwargs["headers"] = h
            else:
                h = dict(kwargs.get("extra_headers") or {})
                h["X-Api-Key"] = api_key
                kwargs["extra_headers"] = h
            return _orig(host_creds, endpoint, method, *args, **kwargs)

        rest_utils.http_request = patched

enable_amsc_x_api_key()

# [MLFLOW SETUP 1] set_tracking_uri
# This connects your local script to the centralized MLflow server.
# If you don't set this, MLflow will save everything to a local './mlruns' folder on your laptop.
mlflow.set_tracking_uri("https://mlflow.american-science-cloud.org")

# [MLFLOW SETUP 2] set_experiment
# An "Experiment" is a folder/bucket that groups related runs together.
# If "mnist_cnn_pytorch_v1" doesn't exist, MLflow creates it.
# If it does exist, we just start logging new runs into it.
experiment_name = "mnist_cnn_pytorch_exp"
mlflow.set_experiment(experiment_name)

# --------------------------------------------------
# 2. Helper: Automatic Metadata Extraction
# --------------------------------------------------
def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
        return commit, branch
    except Exception:
        return "unknown", "unknown"

# --------------------------------------------------
# 3. Configuration & Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST CNN with MLflow Tags")
    
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs") # Reduced default for speed
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    parser.add_argument("--phase", type=str, default="development", 
                        choices=["development", "tuning", "production"],
                        help="Project phase")
    parser.add_argument("--owner", type=str, default=getpass.getuser(),
                        help="User responsible for this run")
    parser.add_argument("--note", type=str, default="", 
                        help="Optional text description")

    parser.add_argument("--register", action="store_true",
                        help="If set, registers the model, promotes to 'production', and runs inference")

    return parser.parse_args()

# --------------------------------------------------
# 4. Model Definition
# --------------------------------------------------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --------------------------------------------------
# 5. Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    
    args = parse_args()
    git_commit, git_branch = get_git_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # [MLFLOW TRACKING 1] start_run
    # This initiates a new "Run" (a single execution of your code).
    # The 'with' statement ensures the run is automatically closed if the script crashes.
    # We name the run dynamically (e.g., "MNIST_development_8cf") so it's easy to spot in the UI.
    with mlflow.start_run(run_name=f"MNIST_{args.phase}_{args.owner}") as run:
        
        # Capture Run ID for reference
        run_id = run.info.run_id

        # ==========================================
        # [MLFLOW TRACKING 2] set_tags
        # Tags are METADATA used for organization and filtering.
        # They don't affect the model, but they let you search later.
        # Example query in UI: tags.owner = 'ming' and tags.project.phase = 'production'
        # ==========================================
        tags = {
            "owner": args.owner,
            "project.phase": args.phase,
            "git.commit": git_commit,
            "git.branch": git_branch,
            "model.class": "MNISTCNN",
            "model.type": "Computer Vision",
        }
        if args.note:
            tags["mlflow.note.content"] = args.note
        mlflow.set_tags(tags)

        # ==========================================
        # Training Logic
        # ==========================================
        model = MNISTCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # [MLFLOW TRACKING 3] log_params
        # Params are inputs/configuration settings (Hyperparameters).
        # Logging these lets you compare runs to see "Did increasing learning_rate improve accuracy?"
        mlflow.log_params({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "optimizer": "Adam"
        })

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            correct = 0
            total = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            acc = correct / total


            # [MLFLOW TRACKING 4] log_metrics
            # Metrics are numeric outputs that change over time.
            # MLflow automatically plots these as line charts (Step on X-axis, Value on Y-axis).
            mlflow.log_metrics({"train_loss": train_loss, "val_accuracy": acc}, step=epoch)
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Acc: {acc:.4f}")

        # ==========================================
        # Artifact Logging
        # ==========================================
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"MNIST CM ({args.phase})")
        plt.savefig("confusion_matrix.png")
        plt.close()

        # [MLFLOW TRACKING 5] log_artifact
        # Artifacts are output FILES (images, dataframes, parquet files, etc.).
        # This uploads the 'confusion_matrix.png' to the artifact store (e.g., S3) associated with this run.
        mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

        # ==========================================
        # Registration & Versioning
        # ==========================================
        # Create a sample input. MLflow needs this to infer the schema (Input/Output shapes).
        example_input = torch.randn(1, 1, 28, 28)
        example_output = model(example_input.to(device)).detach().cpu().numpy()
        signature = infer_signature(example_input.numpy(), example_output)

        reg_name = "MNIST_CNN" if args.register else None

        # [MLFLOW MODEL REGISTRY 1] log_model
        # This is the most powerful function. It does three things:
        # 1. Serializes the model (pickle/pytorch format).
        # 2. Captures the environment (conda.yaml, requirements.txt) so it can be served later.
        # 3. If 'registered_model_name' is provided, it adds this model to the Central Model Registry.
        #    (It will create Version 1 if new, or Version N+1 if it already exists).
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            input_example=example_input.numpy(),
            registered_model_name=reg_name 
        )

    # ==========================================
    # Post-Run: Alias Promotion & Inference
    # ==========================================
    if args.register and model_info.registered_model_version:
        
        print("\n" + "="*40)
        print("  MODEL REGISTRY & INFERENCE")
        print("="*40)

        # 1. Promote to Alias "production"
        # [MLFLOW CLIENT]
        # The MlflowClient allows us to interact with the Registry API programmatically
        # (e.g., creating versions, transitioning stages, setting aliases) without being inside a 'run'.
        client = MlflowClient()
        version = model_info.registered_model_version
        print(f"Model registered as '{reg_name}' Version: {version}")
        
        # [MLFLOW ALIASES] set_registered_model_alias
        # Aliases are pointers to specific versions. 
        # Instead of saying "Deploy Version 5", we say "Deploy @production".
        # This command moves the "@production" alias to the version we just trained.
        client.set_registered_model_alias(
            name=reg_name,
            alias="production",
            version=version
        )
        print(f"Successfully assigned alias 'production' to Version {version}.")

        # 2. Load Model via Alias
        # [MLFLOW INFERENCE] load_model with Alias URI
        # We load the model using the "models:/" URI scheme.
        # Notice we use '@production' instead of a specific version number.
        # This decouples the inference code from the training code. 
        # The inference script always loads '@production', regardless of which version is currently valid.
        production_uri = f"models:/{reg_name}@production"
        print(f"Loading model from URI: {production_uri} ...")
        
        try:
            loaded_model = mlflow.pytorch.load_model(production_uri).to(device)
            loaded_model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
            exit(1)

        # 3. Perform Inference on One Sample
        # We reuse test_ds from earlier
        x_sample, y_true = test_ds[0]
        
        # Prepare input (add batch dim)
        input_tensor = x_sample.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_logits = loaded_model(input_tensor)
            pred_label = output_logits.argmax(dim=1).item()

        print(f"\n--- Inference Result ---")
        print(f"True Label: {y_true}")
        print(f"Predicted : {pred_label}")
        print("="*40)
    
    else:
        print("Run Complete. Model Logged (Not Registered). Use --register to register and test.")

