import torch
import torch.nn as nn


class TerraSpectra3DCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3, 7),
                padding=(1, 1, 3)
            ),

            nn.BatchNorm3d(8),

            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
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
                kernel_size=(2, 2, 2)
            ),

            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(32),

            nn.ReLU(),

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x