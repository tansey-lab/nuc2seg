import logging
import torch

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from nuc2seg.data import Nuc2SegDataset

logger = logging.getLogger(__name__)


def xenium_collate_fn(data):
    outputs = {key: [] for key in data[0].keys()}
    for sample in data:
        for key, val in sample.items():
            outputs[key].append(val)
    outputs["X"] = pad_sequence(outputs["X"], batch_first=True, padding_value=-1)
    outputs["Y"] = pad_sequence(outputs["Y"], batch_first=True, padding_value=-1)
    outputs["gene"] = pad_sequence(outputs["gene"], batch_first=True, padding_value=-1)
    outputs["labels"] = torch.stack(outputs["labels"])
    outputs["angles"] = torch.stack(outputs["angles"])
    outputs["classes"] = torch.stack(outputs["classes"])
    outputs["label_mask"] = torch.stack(outputs["label_mask"]).type(torch.bool)
    outputs["nucleus_mask"] = torch.stack(outputs["nucleus_mask"]).type(torch.bool)
    outputs["location"] = torch.tensor(np.stack(outputs["location"])).type(torch.long)

    # Edge case: pad_sequence will squeeze tensors if there are no entries.
    # In that case, we just need to add the dimension back.
    if len(outputs["gene"].shape) == 1:
        outputs["X"] = outputs["X"][:, None]
        outputs["Y"] = outputs["Y"][:, None]
        outputs["gene"] = outputs["gene"][:, None]

    return outputs


def generate_tiles(x_extent, y_extent, tile_size, overlap_fraction, tile_ids=None):
    """
    A generator function to yield overlapping tiles from a 2D NumPy array (image).

    Parameters:
    - image: 2D NumPy array representing the image.
    - tile_size: Tuple of (tile_height, tile_width), the size of each tile.
    - overlap_fraction: Fraction of overlap between tiles (0 to 1).
    - tile_ids: List of tile IDs to generate. If None, all tiles are generated.

    Yields:
    - BBox extent in pixels for each tile (non inclusive end) x1, y1, x2, y2
    """
    # Calculate stride for moving the window based on overlap
    stride_y = int(tile_size[0] * (1 - overlap_fraction))
    stride_x = int(tile_size[1] * (1 - overlap_fraction))

    # Ensure stride is at least 1 to avoid infinite loops
    stride_y = max(1, stride_y)
    stride_x = max(1, stride_x)

    # Generate tiles
    tile_id = 0
    for y in range(0, y_extent - tile_size[0] + 1, stride_y):
        for x in range(0, x_extent - tile_size[1] + 1, stride_x):
            if tile_ids is not None and tile_id not in tile_ids:
                continue
            else:
                yield x, y, x + tile_size[1], y + tile_size[0]
            tile_id += 1


class XeniumDataset(Dataset):
    def __init__(
        self,
        dataset: Nuc2SegDataset,
        tile_height: int,
        tile_width: int,
        tile_overlap: float = 0.25,
    ):
        self.ds = dataset
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.tile_overlap = tile_overlap
        self.n_tiles = sum(
            1
            for _ in generate_tiles(
                x_extent=dataset.x_extent_pixels,
                y_extent=dataset.y_extent_pixels,
                tile_size=(tile_height, tile_width),
                overlap_fraction=tile_overlap,
            )
        )

    def __len__(self):
        return self.n_tiles

    def __getitem__(self, idx):
        x1, y1, x2, y2 = next(
            generate_tiles(
                x_extent=self.ds.x_extent_pixels,
                y_extent=self.ds.y_extent_pixels,
                tile_size=(self.tile_height, self.tile_width),
                overlap_fraction=self.tile_overlap,
                tile_ids=[idx],
            )
        )
        transcripts = self.ds.transcripts
        labels = self.ds.labels
        angles = self.ds.angles
        classes = self.ds.classes

        selection_criteria = transcripts[:, 0].between(x1, x2) & transcripts[
            :, 1
        ].between(y1, y2)
        tile_transcripts = transcripts[selection_criteria]
        tile_labels = labels[y1:y2, x1:x2]

        local_ids = np.unique(tile_labels)
        local_ids = local_ids[local_ids > 0]
        for i, c in enumerate(local_ids):
            tile_labels[tile_labels == c] = i + 1

        tile_angles = angles[y1:y2, x1:x2]

        tile_angles[tile_labels == -1] = -1

        tile_classes = classes[y1:y2, x1:x2]

        labels_mask = tile_labels > -1
        nucleus_mask = tile_labels > 0

        return {
            "X": torch.as_tensor(tile_transcripts[:, 0]).long().contiguous(),
            "Y": torch.as_tensor(tile_transcripts[:, 1]).long().contiguous(),
            "gene": torch.as_tensor(tile_transcripts[:, 2]).long().contiguous(),
            "labels": torch.as_tensor(tile_angles).long().contiguous(),
            "angles": torch.as_tensor(angles).float().contiguous(),
            "classes": torch.as_tensor(tile_classes).long().contiguous(),
            "label_mask": torch.as_tensor(labels_mask).bool().contiguous(),
            "nucleus_mask": torch.as_tensor(nucleus_mask).bool().contiguous(),
            "location": np.array([x1, y1]),
        }
