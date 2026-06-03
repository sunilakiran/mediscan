import shutil
from huggingface_hub import hf_hub_download

print("Downloading model from HF Hub...")
path = hf_hub_download(
    repo_id="sunilakiran56/mediscan-model",
    filename="mediscan_model.pt",
    repo_type="model",
)
shutil.copy(path, "mediscan_model.pt")
print("Model downloaded successfully!")