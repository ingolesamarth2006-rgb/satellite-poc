import torch
import open_clip
from huggingface_hub import hf_hub_download

model_name = "RN50"

print("Creating RN50 model...")

model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

print("Finding RemoteCLIP checkpoint...")

checkpoint_path = hf_hub_download(
    repo_id="chendelong/RemoteCLIP",
    filename="RemoteCLIP-RN50.pt",
    cache_dir="checkpoints"
)

print("Loading trained RemoteCLIP weights...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

result = model.load_state_dict(checkpoint)

model = model.eval()

print("\nREMOTECLIP MODEL LOADED SUCCESSFULLY!")
print(result)