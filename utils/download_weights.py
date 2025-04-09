from huggingface_hub import snapshot_download

# Download the model to the specified directory
model_path = snapshot_download(
    repo_id="RobbieHolland/RetinaVLM",
    local_dir="/home/jack/Projects/yixin-llm/yixin-llm-data/SpecialistVLMs",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {model_path}")

