import json
import numpy as np

INPUT_PATH = "dataset/processed/pca_hyperspectral.npy"
OUTPUT_PATH = "dataset/processed/pca_map.json"

# Load PCA result
pca_cube = np.load(INPUT_PATH)

height, width, components = pca_cube.shape

# Use the first PCA component for visualization
component = pca_cube[:, :, 0]

# Normalize values to 0–1 for map visualization
minimum = component.min()
maximum = component.max()

normalized = (component - minimum) / (maximum - minimum)

# Create map points
points = []

# Example farm location around Visakhapatnam
base_longitude = 83.2185
base_latitude = 17.6868

for row in range(height):
    for col in range(width):

        longitude = base_longitude + (col - width / 2) * 0.00005
        latitude = base_latitude + (row - height / 2) * 0.00005

        points.append({
            "longitude": float(longitude),
            "latitude": float(latitude),
            "value": float(normalized[row, col])
        })

output = {
    "width": width,
    "height": height,
    "components": components,
    "component_used": 1,
    "points": points
}

with open(OUTPUT_PATH, "w") as file:
    json.dump(output, file)

print("PCA map data created successfully!")
print("Shape:", pca_cube.shape)
print("Component used: PCA 1")
print("Number of map points:", len(points))
print("Saved:", OUTPUT_PATH)