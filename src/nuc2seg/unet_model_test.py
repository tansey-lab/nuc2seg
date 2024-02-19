from nuc2seg.unet_model import SparseUNet, angle_loss
import torch


def test_model():
    unet = SparseUNet(
        n_channels=3,
        n_classes=3,
        tile_height=64,
        tile_width=64,
        tile_overlap=0.0,
        celltype_criterion_weights=torch.tensor([1, 1, 1]),
    )

    pred = unet.forward(
        x=torch.tensor([[0, 1, 0, 1]]),
        y=torch.tensor([[0, 0, 1, 1]]),
        z=torch.tensor([[0, 1, 2, 2]]),
    )

    assert pred.shape == (1, 64, 64, 5)
