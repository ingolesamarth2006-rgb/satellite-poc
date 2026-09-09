import rasterio

IMAGE_PATH = r"C:\Users\samar\Downloads\RSITMD\images\viaduct_4576.tif"

with rasterio.open(IMAGE_PATH) as src:

    print("\n========== RSITMD GEOREFERENCE TEST ==========")

    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)

    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Transform:", src.transform)

    print("Driver:", src.driver)

    print("==============================================")