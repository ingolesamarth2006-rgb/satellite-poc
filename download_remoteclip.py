from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="chendelong/RemoteCLIP",
    filename="RemoteCLIP-RN50.pt",
    cache_dir="checkpoints"
)

print("RemoteCLIP downloaded successfully!")
print("Checkpoint:", checkpoint_path)