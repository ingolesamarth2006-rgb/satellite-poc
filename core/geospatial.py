from pathlib import Path

import rasterio
import numpy as np
from PIL import Image
def read_geotiff_metadata(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"GeoTIFF not found: {image_path}"
        )

    with rasterio.open(image_path) as src:

        metadata = {
            "path": str(image_path),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top
            },
            "transform": str(src.transform),
            "driver": src.driver,
            "dtype": str(src.dtypes[0])
        }

    return metadata
def create_rgb_preview(image_path, output_path):

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"GeoTIFF not found: {image_path}"
        )

    with rasterio.open(image_path) as src:

        if src.count < 3:
            raise ValueError(
                "GeoTIFF must contain at least 3 bands for RGB preview."
            )

        # Read RGB bands
        rgb = src.read([1, 2, 3])

    # Rasterio format:
    # (bands, height, width)
    #
    # PIL expects:
    # (height, width, bands)

    rgb = np.moveaxis(
        rgb,
        0,
        -1
    )

    # Convert to uint8 if needed
    if rgb.dtype != np.uint8:

        rgb_float = rgb.astype(np.float32)

        output = np.zeros(
            rgb.shape,
            dtype=np.uint8
        )

        for band in range(3):

            channel = rgb_float[:, :, band]

            low = np.percentile(channel, 2)
            high = np.percentile(channel, 98)

            if high > low:

                channel = (
                    (channel - low) /
                    (high - low)
                )

                channel = np.clip(
                    channel,
                    0,
                    1
                )

                output[:, :, band] = (
                    channel * 255
                ).astype(np.uint8)

        rgb = output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        rgb
    ).save(output_path)

    return str(output_path)