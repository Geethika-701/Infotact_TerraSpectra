import numpy as np
import torch

from vit_model import TerraSpectraViT


CUBE_PATH = "dataset/raw/mock_hyperspectral_cube.npy"

MODEL_PATH = (
    "models/terraspectra_vit.pth"
)


def load_vit_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = TerraSpectraViT(
        bands=224,
        patch_size=7,
        spectral_group_size=16,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model, device


def predict_vit_patch(
    model,
    device,
    patch
):

    tensor = np.transpose(
        patch,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        tensor,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    return probabilities[0, 1].item()