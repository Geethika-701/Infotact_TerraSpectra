import torch
import torch.nn as nn


class Hybrid3DCNN(nn.Module):
    """
    Hybrid hyperspectral classifier.

    Branch 1:
        3D CNN processes the original 224-band
        hyperspectral pixel patch.

    Branch 2:
        2D CNN processes the PCA-reduced
        10-component representation.

    The two feature representations are fused
    for healthy vs chemically stressed classification.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # -------------------------------------------------
        # 3D CNN BRANCH
        # Input:
        # [batch, 1, 5, 5, 224]
        # -------------------------------------------------

        self.spectral_branch = nn.Sequential(

            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3, 7),
                padding=(1, 1, 3)
            ),

            nn.BatchNorm3d(8),
            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=(2, 2, 4)
            ),

            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3, 5),
                padding=(1, 1, 2)
            ),

            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=(2, 2, 4)
            ),

            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding=1
            ),

            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            )
        )

        # -------------------------------------------------
        # PCA 2D BRANCH
        # Input:
        # [batch, 10, 5, 5]
        # -------------------------------------------------

        self.pca_branch = nn.Sequential(

            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        # -------------------------------------------------
        # FUSION CLASSIFIER
        # 3D branch = 32
        # PCA branch = 32
        # Total = 64
        # -------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                num_classes
            )
        )

    def forward(
        self,
        hyperspectral,
        pca
    ):

        # 3D CNN features
        spectral_features = self.spectral_branch(
            hyperspectral
        )

        spectral_features = spectral_features.view(
            spectral_features.size(0),
            -1
        )

        # PCA features
        pca_features = self.pca_branch(
            pca
        )

        pca_features = pca_features.view(
            pca_features.size(0),
            -1
        )

        # Hybrid feature fusion
        combined = torch.cat(
            [
                spectral_features,
                pca_features
            ],
            dim=1
        )

        output = self.classifier(
            combined
        )

        return output


if __name__ == "__main__":

    model = Hybrid3DCNN()

    print(model)

    # Test tensor
    hyperspectral = torch.randn(
        2,
        1,
        5,
        5,
        224
    )

    pca = torch.randn(
        2,
        10,
        5,
        5
    )

    output = model(
        hyperspectral,
        pca
    )

    print(
        "\nTest output shape:",
        output.shape
    )