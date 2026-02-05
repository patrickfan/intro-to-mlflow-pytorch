import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# --- Security Configuration ---
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
urllib3.disable_warnings(InsecureRequestWarning)

# Function to inject API Key (Reuse your existing logic here)
def enable_amsc_x_api_key():
    import mlflow.utils.rest_utils as rest_utils
    api_key = os.environ.get("AM_SC_API_KEY")
    if api_key:
        _orig = rest_utils.http_request
        def patched(host_creds, endpoint, method, *args, **kwargs):
            h = dict(kwargs.get("headers") or kwargs.get("extra_headers") or {})
            h["X-Api-Key"] = api_key
            kwargs["headers" if "headers" in kwargs else "extra_headers"] = h
            return _orig(host_creds, endpoint, method, *args, **kwargs)
        rest_utils.http_request = patched

enable_amsc_x_api_key()
mlflow.set_tracking_uri("https://mlflow.american-science-cloud.org")

# 1. Re-define or Import the model class structure
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1); self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2); self.fc1 = nn.Linear(9216, 128); self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x)); x = torch.relu(self.conv2(x))
        x = self.pool(x); x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x)); return self.fc2(x)

def upload_local_model():
    model = MNISTCNN()
    # 2. Load the local weights
    model.load_state_dict(torch.load("my_local_mnist_model.pth"))
    model.eval()

    mlflow.set_experiment("mnist_cnn_pytorch_exp")

    # 3. Log and Register in one go
    with mlflow.start_run(run_name="Manual_Upload") as run:
        print(f"Uploading local model to MLflow...")
        
        # We use log_model to package the weights + environment together
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="MNIST_CNN_Official" # It will create a new version here
        )
        print(f"✅ Successfully uploaded! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    upload_local_model()

