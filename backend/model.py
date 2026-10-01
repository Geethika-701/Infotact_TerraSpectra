import torch
import torch.nn as nn


class Hybrid3DCNN(nn.Module):
    """
    Hybrid 3D-CNN for hyperspectral pixel classification.

    Input:
        [batch, 1, spectral_bands, height, width]

    Example:
        [64, 1, 224, 5, 5]

    Output:
        [batch, 2]

    Classes:
        0 = Healthy
        1 = Chemically Stressed
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ---------------------------------------------------------
        # 3D CNN
        # Learns spectral-spatial features from hyperspectral data
        # ---------------------------------------------------------
        self.spectral_spatial_features = nn.Sequential(

            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(7, 3, 3),
                padding=(3, 1, 1)
            ),

            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(2, 1, 1)
            ),

            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(5, 3, 3),
                padding=(2, 1, 1)
            ),

            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            # Reduce the spectral dimension to 1
            nn.AdaptiveAvgPool3d((1, 5, 5))
        )

        # ---------------------------------------------------------
        # 2D CNN
        # Learns spatial information after spectral extraction
        # ---------------------------------------------------------
        self.spatial_features = nn.Sequential(

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ---------------------------------------------------------
        # Classification layer
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32, 16),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):

        # Input:
        # [batch, 1, bands, height, width]

        x = self.spectral_spatial_features(x)

        # After AdaptiveAvgPool3d:
        # [batch, 16, 1, 5, 5]

        # Remove spectral depth dimension
        x = x.squeeze(2)

        # Now:
        # [batch, 16, 5, 5]

        x = self.spatial_features(x)

        # Now:
        # [batch, 32, 1, 1]

        x = self.classifier(x)

        # Final:
        # [batch, 2]

        return x


if __name__ == "__main__":

    # Simple architecture test
    model = Hybrid3DCNN()

    dummy_input = torch.randn(
        2,
        1,
        224,
        5,
        5
    )

    output = model(dummy_input)

    print("Hybrid 3D-CNN test successful")
    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Model:")
    print(model)