import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from torchvision import datasets, transforms
import os
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# --- Security Configuration ---
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
urllib3.disable_warnings(InsecureRequestWarning)

AMSC_API_KEY_ENV = "AM_SC_API_KEY"   # token lives here

if AMSC_API_KEY_ENV not in os.environ:
    print(f"Warning: {AMSC_API_KEY_ENV} not found in environment.")
    
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

def inference_with_version(model_name, version):
    client = MlflowClient()

    print(f"\n" + "="*40)
    print(f"🔍 INSPECTING MODEL VERSION: {version}")
    print(f"\n" + "="*40)

    try:
        # 1. Get Metadata using MlflowClient
        # This proves the link between the Registry and the original Training Run
        mv = client.get_model_version(model_name, version)
        print(f"Successfully retrieved metadata for '{model_name}'")
        print(f"  - Source Run ID: {mv.run_id}")
        print(f"  - Current Status: {mv.status}")
        print(f"  - Created At: {mv.creation_timestamp}")

        # 2. Construct the URI for a specific version
        # Syntax: models:/<model_name>/<version_number>
        version_uri = f"models:/{model_name}/{version}"
        print(f"\n🚀 Loading model from URI: {version_uri}")

        # 3. Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    Inference Device: {device}")
        
        model = mlflow.pytorch.load_model(version_uri)
        model.to(device)
        model.eval()

        # 4. Perform a test prediction
        # (Using a simple test sample from MNIST)
        transform = transforms.Compose([transforms.ToTensor()])
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        sample, label = test_ds[0]

        input_tensor = sample.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(input_tensor).argmax(dim=1).item()

        print(f"\n--- Result for Version {version} ---")
        print(f"Target Digit: {label}")
        print(f"Predicted   : {prediction}")
        print(f"Match       : {'✅ YES' if label == prediction else '❌ NO'}")

    except Exception as e:
        print(f"❌ Error retrieving Version {version}: {e}")

if __name__ == "__main__":
    MODEL_NAME = "MNIST_CNN_Official"
    
    # You can change this number during the meeting to show different results
    TARGET_VERSION = 8
    
    inference_with_version(MODEL_NAME, TARGET_VERSION)


