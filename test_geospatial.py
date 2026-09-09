from core.geospatial import read_geotiff_metadata

metadata = read_geotiff_metadata(
    "RGB.byte.tif"
)

print("\n========== GEOTIFF METADATA ==========")

for key, value in metadata.items():
    print(f"{key}: {value}")

print("\nGEOSPATIAL MODULE SUCCESS")

