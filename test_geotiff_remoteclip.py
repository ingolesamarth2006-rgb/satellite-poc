import torch
from PIL import Image

from core.semantic_search import load_remoteclip


IMAGE_PATH = "integration_data/geotiff_preview.png"

QUERIES = [
    "urban residential area",
    "forest area",
    "river surrounded by vegetation",
    "industrial area",
    "highway"
]


print("\n========== GEOTIFF → REMOTECLIP TEST ==========")

model, preprocess, tokenizer = load_remoteclip()

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = (
    preprocess(image)
    .unsqueeze(0)
)

with torch.no_grad():

    image_features = model.encode_image(
        image_tensor
    )

    image_features = (
        image_features /
        image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    text = tokenizer(
        QUERIES
    )

    text_features = model.encode_text(
        text
    )

    text_features = (
        text_features /
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    similarities = (
        text_features @ image_features.T
    ).squeeze(1)


results = list(
    zip(
        QUERIES,
        similarities.tolist()
    )
)

results.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nSemantic Ranking:")

for index, (query, score) in enumerate(
    results,
    start=1
):
    print(
        f"{index}. {query} | "
        f"Score: {score:.4f}"
    )


print("\nBEST MATCH:")
print(results[0][0])

print("\nGEOTIFF → REMOTECLIP SUCCESS")