import rasterio

file_path = "RGB.byte.tif"

with rasterio.open(file_path) as src:
    print("\n========== GEOTIFF TEST ==========")

    print("\nFILE")
    print(src.name)

    print("\nIMAGE SIZE")
    print("Width:", src.width)
    print("Height:", src.height)

    print("\nBANDS")
    print("Bands:", src.count)

    print("\nCRS")
    print(src.crs)

    print("\nBOUNDS")
    print(src.bounds)

    print("\nRESOLUTION")
    print(src.res)

    print("\nDATA TYPES")
    print(src.dtypes)

    print("\n==================================")