import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# --- Security & Setup ---
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
urllib3.disable_warnings(InsecureRequestWarning)

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
mlflow.set_experiment("mnist_cnn_pytorch_exp")

def evaluate_production_model():
    model_name = "MNIST_CNN_Official"
    alias = "production"
    model_uri = f"models:/{model_name}@{alias}"

    # local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    # full_path = os.path.abspath(local_path)
    # print(f"📍 The model is actually stored here: {full_path}")

    print(f"🚀 Loading model: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    # --- 1. Prepare Test Data ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)

    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # --- 2. Start MLflow Evaluation Run ---
    with mlflow.start_run(run_name=f"mnist_evaluation_v_{alias}"):
        
        # Log Metrics
        accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        mlflow.log_metric("eval_accuracy", accuracy)
        print(f"✅ Accuracy: {accuracy:.4f}")

        # --- 3. Create & Log Confusion Matrix Plot ---
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}@{alias}')
        
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
        plt.close()

        # --- 4. Log Detailed Classification Report as CSV ---
        report = classification_report(all_labels, all_preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = "classification_report.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path, artifact_path="evaluation_results")

        print(f"📊 Evaluation artifacts logged to MLflow successfully.")

if __name__ == "__main__":
    evaluate_production_model()


