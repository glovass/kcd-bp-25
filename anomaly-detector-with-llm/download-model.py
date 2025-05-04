from huggingface_hub import snapshot_download
from huggingface_hub import login

snapshot_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct", local_dir="./model/")
print("download complete")