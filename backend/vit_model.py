import torch
import torch.nn as nn


class SpectralPatchEmbedding(nn.Module):
    """
    Converts a 7x7 spatial neighborhood containing 16
    spectral bands into a 128-dimensional token.

    Input per token:
        7 x 7 x 16 = 784 features

    Output:
        128-dimensional embedding
    """

    def __init__(
        self,
        input_features=784,
        embed_dim=128
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_features,
            embed_dim
        )

    def forward(self, x):
        return self.projection(x)


class TerraSpectraViT(nn.Module):
    """
    TerraSpectra Vision Transformer.

    Input:
        [batch, 7, 7, 224]

    Processing:
        224 spectral bands
             ↓
        14 spectral groups
             ↓
        7x7x16 = 784 features per token
             ↓
        128-dimensional embeddings
             ↓
        Multi-head self-attention
             ↓
        Transformer encoder
             ↓
        Healthy / Chemically Stressed
    """

    def __init__(
        self,
        bands=224,
        spectral_group_size=16,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=2
    ):
        super().__init__()

        self.bands = bands
        self.spectral_group_size = spectral_group_size
        self.embed_dim = embed_dim

        if bands % spectral_group_size != 0:
            raise ValueError(
                "Number of spectral bands must be divisible "
                "by spectral_group_size."
            )

        self.num_tokens = (
            bands // spectral_group_size
        )

        self.patch_embedding = SpectralPatchEmbedding(
            input_features=(
                7 * 7 * spectral_group_size
            ),
            embed_dim=embed_dim
        )

        self.class_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens + 1,
                embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
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
            embed_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                embed_dim,
                64
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                64,
                num_classes
            )
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )

        nn.init.xavier_uniform_(
            self.patch_embedding.projection.weight
        )

        if (
            self.patch_embedding.projection.bias
            is not None
        ):
            nn.init.zeros_(
                self.patch_embedding.projection.bias
            )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                "TerraSpectraViT expects input "
                "with shape [batch, 7, 7, 224]. "
                f"Received {tuple(x.shape)}"
            )

        batch_size = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]
        bands = x.shape[3]

        if height != 7 or width != 7:
            raise ValueError(
                "TerraSpectraViT expects 7x7 "
                "spatial patches. "
                f"Received {height}x{width}."
            )

        if bands != self.bands:
            raise ValueError(
                f"TerraSpectraViT expects "
                f"{self.bands} spectral bands. "
                f"Received {bands}."
            )

        # -------------------------------------------------
        # Step 1
        # Split 224 spectral bands into 14 groups.
        #
        # 224 / 16 = 14
        #
        # [B, 7, 7, 224]
        #
        # becomes
        #
        # [B, 7, 7, 14, 16]
        # -------------------------------------------------

        x = x.reshape(
            batch_size,
            height,
            width,
            self.num_tokens,
            self.spectral_group_size
        )

        # -------------------------------------------------
        # Step 2
        # Move spectral tokens before spatial dimensions.
        #
        # [B, 7, 7, 14, 16]
        #
        # becomes
        #
        # [B, 14, 7, 7, 16]
        # -------------------------------------------------

        x = x.permute(
            0,
            3,
            1,
            2,
            4
        )

        # -------------------------------------------------
        # Step 3
        # Flatten each spectral token.
        #
        # 7 x 7 x 16 = 784
        #
        # [B, 14, 7, 7, 16]
        #
        # becomes
        #
        # [B, 14, 784]
        # -------------------------------------------------

        x = x.reshape(
            batch_size,
            self.num_tokens,
            7 * 7 * self.spectral_group_size
        )

        # -------------------------------------------------
        # Step 4
        # Spectral patch embedding.
        #
        # [B, 14, 784]
        #
        # becomes
        #
        # [B, 14, 128]
        # -------------------------------------------------

        x = self.patch_embedding(x)

        # -------------------------------------------------
        # Step 5
        # Add classification token.
        #
        # [B, 14, 128]
        #
        # becomes
        #
        # [B, 15, 128]
        # -------------------------------------------------

        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [
                class_token,
                x
            ],
            dim=1
        )

        # -------------------------------------------------
        # Step 6
        # Add positional embeddings.
        # -------------------------------------------------

        x = (
            x +
            self.position_embedding
        )

        # -------------------------------------------------
        # Step 7
        # Multi-head self-attention.
        # -------------------------------------------------

        x = self.transformer(x)

        # -------------------------------------------------
        # Step 8
        # Normalize transformer representation.
        # -------------------------------------------------

        x = self.norm(x)

        # -------------------------------------------------
        # Step 9
        # Take classification token.
        # -------------------------------------------------

        x = x[:, 0]

        # -------------------------------------------------
        # Step 10
        # Binary classification.
        # -------------------------------------------------

        x = self.classifier(x)

        return x