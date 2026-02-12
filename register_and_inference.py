import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from torchvision import datasets, transforms
import os
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# --- SSL & Security Configuration ---
# Ignore SSL certificate errors for internal servers
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
urllib3.disable_warnings(InsecureRequestWarning)

AMSC_API_KEY_ENV = "AM_SC_API_KEY"   # token lives here

if AMSC_API_KEY_ENV not in os.environ:
    print(f"Warning: {AMSC_API_KEY_ENV} not found in environment.")
    
# Inject X-Api-Key into all MLflow REST calls
def enable_amsc_x_api_key():
    """
    MLFLOW AUTHENTICATION HELPER:
    Standard MLflow does not automatically inject custom headers like 'X-Api-Key'.
    We patch the http_request function to ensure every request to the server 
    includes our security token.
    """
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

# --- MLflow Setup ---
mlflow.set_tracking_uri("https://mlflow.american-science-cloud.org")

def register_and_serve(run_id):
    client = MlflowClient()
    model_name = "MNIST_CNN_Official"
    
    # 1. REGISTER THE MODEL
    # We point to the artifact location from the training run
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model version from Run: {run_id}...")
    mv = mlflow.register_model(model_uri, model_name)
    
    # 2. ASSIGN ALIAS (Promotion)
    # Instead of manual stages, we use Aliases to identify the 'production' version
    client.set_registered_model_alias(model_name, "production", mv.version)
    print(f"✅ Model '{model_name}' Version {mv.version} promoted to @production.")

    # 3. INFERENCE SERVICE LOADING
    # The inference code doesn't need to know the Version Number or Run ID.
    # It simply asks for the '@production' alias.
    prod_uri = f"models:/{model_name}@production"
    print(f"Inference Service: Loading model from {prod_uri}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Inference Device: {device}")
    
    loaded_model = mlflow.pytorch.load_model(prod_uri)
    loaded_model.to(device)
    loaded_model.eval()

    # Test inference with one sample from the test set
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    sample, label = test_ds[0]

    input_tensor = sample.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Add batch dimension and predict
        prediction = loaded_model(input_tensor).argmax(dim=1).item()
    
    print(f"\n--- Inference Result ---")
    print(f"Actual Label: {label}")
    print(f"Model Prediction: {prediction}")
    print(f"------------------------")

if __name__ == "__main__":
    # In the meeting, copy the Run ID from the first script output and paste it here
    target_run_id = input("Please enter the Run ID to register: ").strip()
    if target_run_id:
        register_and_serve(target_run_id)
    else:
        print("Error: A valid Run ID is required.")

