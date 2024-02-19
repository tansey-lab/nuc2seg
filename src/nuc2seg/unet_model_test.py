from nuc2seg.unet_model import SparseUNet, Nuc2SegDataModule
import torch
import shutil
import numpy as np
from pytorch_lightning import Trainer
from nuc2seg.data import Nuc2SegDataset
import tempfile
import os


def test_model_predict():
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


def test_model():
    model = SparseUNet(
        n_channels=3,
        n_classes=3,
        tile_height=64,
        tile_width=64,
        tile_overlap=0.0,
        celltype_criterion_weights=torch.tensor([1, 1, 1]),
    )

    trainer = Trainer(fast_dev_run=True, accelerator="cpu")

    ds = Nuc2SegDataset(
        labels=np.ones((100, 100)),
        angles=np.zeros((100, 100)),
        classes=np.ones((100, 100), dtype=int),
        transcripts=np.array([[0, 0, 0], [5, 5, 1], [10, 10, 2]]),
        bbox=np.array([100, 100, 200, 200]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    tmpdir = tempfile.mkdtemp()

    output_path = os.path.join(tmpdir, "test.h5")

    ds.save_h5(output_path)

    dm = Nuc2SegDataModule(preprocessed_data_path=output_path, val_percent=0.5)

    try:
        trainer.fit(model, dm)
    finally:
        shutil.rmtree(tmpdir)
