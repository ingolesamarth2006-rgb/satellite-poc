import os
import torch
import open_clip
from PIL import Image
from huggingface_hub import hf_hub_download

# ==============================
# SETTINGS
# ==============================
IMAGE_FOLDER = "images"
QUERY = "a satellite image of a dense green forest"

# RemoteCLIP lightweight model
MODEL_NAME = "RN50"

# ==============================
# LOAD MODEL
# ==============================
print("Loading RemoteCLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

checkpoint_path = hf_hub_download(
    repo_id="chendelong/RemoteCLIP",
    filename="RemoteCLIP-RN50.pt",
    cache_dir="checkpoints"
)

checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

print("RemoteCLIP loaded successfully!")

# ==============================
# LOAD ALL IMAGES FROM FOLDER
# ==============================
image_files = []
for file in os.listdir(IMAGE_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files.append(file)

if len(image_files) == 0:
    print("No images found in 'images' folder.")
    exit()

print("\nImages found:")
for file in image_files:
    print("-", file)

# ==============================
# ENCODE QUERY TEXT
# ==============================
text = tokenizer([QUERY])

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ==============================
# COMPARE QUERY WITH EACH IMAGE
# ==============================
results = []

with torch.no_grad():
    for file_name in image_files:
        image_path = os.path.join(IMAGE_FOLDER, file_name)

        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0)

        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).item()

        results.append((file_name, similarity))

# ==============================
# SORT RESULTS
# ==============================
results.sort(key=lambda x: x[1], reverse=True)

# ==============================
# PRINT FINAL RESULTS
# ==============================
print("\n========== SEMANTIC SEARCH RESULT ==========")
print("Query:", QUERY)
print()

for file_name, score in results:
    print(f"{file_name} -> {score:.2f}")

print("\nBEST MATCH:")
print(results[0][0], "->", f"{results[0][1]:.2f}")
print("============================================")