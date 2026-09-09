from core.geospatial import create_rgb_preview

preview = create_rgb_preview(
    "RGB.byte.tif",
    "integration_data/geotiff_preview.png"
)

print("\nGEOTIFF PREVIEW SUCCESS")
print(f"Preview: {preview}")