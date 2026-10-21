import torch
import torch.nn as nn


class SpectralPatchEmbedding(nn.Module):
    """
    Converts a hyperspectral spatial-spectral patch into
    a sequence of spectral tokens.

    Input:
        [batch, 1, bands, height, width]

    Example:
        [16, 1, 224, 7, 7]

    224 bands are divided into spectral groups.
    """

    def __init__(
        self,
        bands=224,
        patch_size=7,
        spectral_group_size=16,
        embedding_dim=128
    ):
        super().__init__()

        self.bands = bands
        self.patch_size = patch_size
        self.spectral_group_size = spectral_group_size

        self.num_tokens = bands // spectral_group_size

        input_features = (
            spectral_group_size
            * patch_size
            * patch_size
        )

        self.projection = nn.Linear(
            input_features,
            embedding_dim
        )

    def forward(self, x):

        batch_size = x.shape[0]

        # [B, 1, Bands, H, W]
        x = x.squeeze(1)

        tokens = []

        for start in range(
            0,
            self.bands,
            self.spectral_group_size
        ):

            end = start + self.spectral_group_size

            spectral_group = x[
                :,
                start:end,
                :,
                :
            ]

            spectral_group = spectral_group.reshape(
                batch_size,
                -1
            )

            token = self.projection(
                spectral_group
            )

            tokens.append(token)

        tokens = torch.stack(
            tokens,
            dim=1
        )

        return tokens


class TerraSpectraViT(nn.Module):

    def __init__(
        self,
        bands=224,
        patch_size=7,
        spectral_group_size=16,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=2
    ):

        super().__init__()

        self.patch_embedding = SpectralPatchEmbedding(
            bands=bands,
            patch_size=patch_size,
            spectral_group_size=spectral_group_size,
            embedding_dim=embedding_dim
        )

        num_tokens = (
            bands // spectral_group_size
        )

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_tokens + 1,
                embedding_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                64
            ),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        tokens = self.patch_embedding(x)

        batch_size = tokens.shape[0]

        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1
        )

        tokens = torch.cat(
            [class_token, tokens],
            dim=1
        )

        tokens = (
            tokens
            + self.position_embedding[:, :tokens.shape[1], :]
        )

        encoded = self.transformer(
            tokens
        )

        representation = self.norm(
            encoded[:, 0]
        )

        output = self.classifier(
            representation
        )

        return output