import os
import json
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "pca_cube.npy"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "frontend",
    "public"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "pca_map.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


pca_cube = np.load(
    INPUT_FILE
)

height, width, components = \
    pca_cube.shape


data = []


for row in range(height):

    for col in range(width):

        # Geographic coordinate
        lon = 80.0000 + (
            col * 0.0001
        )

        lat = 16.0000 - (
            row * 0.0001
        )

        data.append({

            "lat": float(lat),

            "lon": float(lon),

            "pc1": float(
                pca_cube[row, col, 0]
            ),

            "pc2": float(
                pca_cube[row, col, 1]
            ),

            "pc3": float(
                pca_cube[row, col, 2]
            )

        })


with open(
    OUTPUT_FILE,
    "w"
) as file:

    json.dump(
        data,
        file
    )


print(
    "PCA JSON created successfully!"
)

print(
    "Pixels:",
    len(data)
)

print(
    "Output:",
    OUTPUT_FILE
)
