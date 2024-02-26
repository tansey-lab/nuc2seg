import h5py
import logging
import torch

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from blended_tiling import TilingModule

logger = logging.getLogger(__name__)


class CelltypingResults:
    def __init__(
        self,
        aic_scores,
        bic_scores,
        final_expression_profiles,
        final_prior_probs,
        final_cell_types,
        relative_expression,
        min_n_components,
        max_n_components,
    ):
        self.aic_scores = aic_scores
        self.bic_scores = bic_scores
        self.final_expression_profiles = final_expression_profiles
        self.final_prior_probs = final_prior_probs
        self.final_cell_types = final_cell_types
        self.relative_expression = relative_expression
        self.n_component_values = np.arange(min_n_components, max_n_components + 1)

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("aic_scores", data=self.aic_scores, compression="gzip")
            f.create_dataset("bic_scores", data=self.bic_scores, compression="gzip")
            f.create_dataset(
                "n_component_values", data=self.n_component_values, compression="gzip"
            )
            for idx, k in self.n_component_values:
                f.create_group(str(idx))
                f[str(idx)].create_dataset(
                    "final_expression_profiles",
                    data=self.final_expression_profiles[k],
                    compression="gzip",
                )
                f[str(idx)].create_dataset(
                    "final_prior_probs",
                    data=self.final_prior_probs[k],
                    compression="gzip",
                )
                f[str(idx)].create_dataset(
                    "final_cell_types",
                    data=self.final_cell_types[k],
                    compression="gzip",
                )
                f[str(idx)].create_dataset(
                    "relative_expression",
                    data=self.relative_expression[k],
                    compression="gzip",
                )
                f[str(idx)].attrs["n_components"] = k

    @staticmethod
    def load_h5(path):
        with h5py.File(path, "r") as f:
            aic_scores = f["aic_scores"][:]
            bic_scores = f["bic_scores"][:]
            final_expression_profiles = []
            final_prior_probs = []
            final_cell_types = []
            relative_expression = []
            n_component_values = []
            for idx, k in f["n_component_values"][:]:
                aic_scores.append(f[str(idx)]["aic_scores"][:])
                bic_scores.append(f[str(idx)]["bic_scores"][:])
                final_expression_profiles.append(
                    f[str(idx)]["final_expression_profiles"][:]
                )
                final_prior_probs.append(f[str(idx)]["final_prior_probs"][:])
                final_cell_types.append(f[str(idx)]["final_cell_types"][:])
                relative_expression.append(f[str(idx)]["relative_expression"][:])
                n_component_values.append(f.attrs["n_components"])
        return CelltypingResults(
            aic_scores=aic_scores,
            bic_scores=bic_scores,
            final_expression_profiles=final_expression_profiles,
            final_prior_probs=final_prior_probs,
            final_cell_types=final_cell_types,
            relative_expression=relative_expression,
            min_n_components=min(n_component_values),
            max_n_components=max(n_component_values),
        )


class Nuc2SegDataset:
    def __init__(
        self, labels, angles, classes, transcripts, bbox, n_classes, n_genes, resolution
    ):
        self.labels = labels
        self.angles = angles
        self.classes = classes
        self.transcripts = transcripts
        self.bbox = bbox
        self.n_classes = n_classes
        self.n_genes = n_genes
        self.resolution = resolution

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=self.labels, compression="gzip")
            f.create_dataset("angles", data=self.angles, compression="gzip")
            f.create_dataset("classes", data=self.classes, compression="gzip")
            f.create_dataset("transcripts", data=self.transcripts, compression="gzip")
            f.create_dataset("bbox", data=self.bbox)
            f.attrs["n_classes"] = self.n_classes
            f.attrs["n_genes"] = self.n_genes
            f.attrs["resolution"] = self.resolution

    @property
    def x_extent_pixels(self):
        return self.labels.shape[0]

    @property
    def y_extent_pixels(self):
        return self.labels.shape[1]

    @staticmethod
    def load_h5(path):
        with h5py.File(path, "r") as f:
            labels = f["labels"][:]
            angles = f["angles"][:]
            classes = f["classes"][:]
            transcripts = f["transcripts"][:]
            bbox = f["bbox"][:]
            n_classes = f.attrs["n_classes"]
            n_genes = f.attrs["n_genes"]
            resolution = f.attrs["resolution"]
        return Nuc2SegDataset(
            labels=labels,
            angles=angles,
            classes=classes,
            transcripts=transcripts,
            bbox=bbox,
            n_classes=n_classes,
            n_genes=n_genes,
            resolution=resolution,
        )


def generate_tiles(
    tiler: TilingModule, x_extent, y_extent, tile_size, overlap_fraction, tile_ids=None
):
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
    # Generate tiles
    tile_id = 0
    for x in tiler._calc_tile_coords(x_extent, tile_size[0], overlap_fraction)[0]:
        for y in tiler._calc_tile_coords(y_extent, tile_size[1], overlap_fraction)[0]:
            if tile_ids is not None:
                if tile_id in tile_ids:
                    yield x, y, x + tile_size[0], y + tile_size[1]
            else:
                yield x, y, x + tile_size[0], y + tile_size[1]
            tile_id += 1


def collate_tiles(data):
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


class TiledDataset(Dataset):
    def __init__(
        self,
        dataset: Nuc2SegDataset,
        tile_height: int,
        tile_width: int,
        tile_overlap: float = 0.25,
    ):
        self.ds = dataset
        self.tile_height = tile_width
        self.tile_width = tile_height
        self.tile_overlap = tile_overlap

        self._tiler = TilingModule(
            tile_size=(tile_width, tile_height),
            tile_overlap=(tile_overlap, tile_overlap),
            base_size=(dataset.x_extent_pixels, dataset.y_extent_pixels),
        )

    def __len__(self):
        return self._tiler.num_tiles()

    @property
    def tiler(self):
        return self._tiler

    @property
    def per_tile_class_histograms(self):
        class_tiles = (
            self._tiler.split_into_tiles(torch.tensor(self.ds.classes[None, None, ...]))
            .squeeze()
            .detach()
            .numpy()
            .astype(int)
        )

        class_tiles_flattened = class_tiles.reshape(
            (self._tiler.num_tiles(), class_tiles.shape[1] * class_tiles.shape[2])
        )

        return np.apply_along_axis(
            np.bincount, 1, class_tiles_flattened + 1, minlength=self.ds.n_classes + 2
        )

    @property
    def celltype_criterion_weights(self):
        weights = torch.Tensor(
            self.per_tile_class_histograms[:, 2:].mean()
            / self.per_tile_class_histograms[:, 2:].mean(axis=0)
        )

        if not torch.all(torch.isfinite(weights)):
            logger.warning("Non-finite weights found. Replacing with epsilon")
            weights[~torch.isfinite(weights)] = 1e-6

        return weights

    def __getitem__(self, idx):
        x1, y1, x2, y2 = next(
            generate_tiles(
                tiler=self.tiler,
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

        selection_criteria = (
            (transcripts[:, 0] < x2)
            & (transcripts[:, 0] > x1)
            & (transcripts[:, 1] < y2)
            & (transcripts[:, 1] > y1)
        )
        tile_transcripts = transcripts[selection_criteria]

        tile_transcripts[:, 0] = tile_transcripts[:, 0] - x1
        tile_transcripts[:, 1] = tile_transcripts[:, 1] - y1

        tile_labels = labels[x1:x2, y1:y2]

        local_ids = np.unique(tile_labels)
        local_ids = local_ids[local_ids > 0]
        for i, c in enumerate(local_ids):
            tile_labels[tile_labels == c] = i + 1

        tile_angles = angles[x1:x2, y1:y2]

        tile_angles[tile_labels == -1] = -1

        tile_classes = classes[x1:x2, y1:y2]

        labels_mask = tile_labels > -1
        nucleus_mask = tile_labels > 0

        return {
            "X": torch.as_tensor(tile_transcripts[:, 0]).long().contiguous(),
            "Y": torch.as_tensor(tile_transcripts[:, 1]).long().contiguous(),
            "gene": torch.as_tensor(tile_transcripts[:, 2]).long().contiguous(),
            "labels": torch.as_tensor(tile_labels).long().contiguous(),
            "angles": torch.as_tensor(tile_angles).float().contiguous(),
            "classes": torch.as_tensor(tile_classes).long().contiguous(),
            "label_mask": torch.as_tensor(labels_mask).bool().contiguous(),
            "nucleus_mask": torch.as_tensor(nucleus_mask).bool().contiguous(),
            "location": np.array([x1, y1]),
        }


class ModelPredictions:
    def __init__(self, angles, classes, foreground):
        self.angles = angles
        self.classes = classes
        self.foreground = foreground

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("angles", data=self.angles, compression="gzip")
            f.create_dataset("classes", data=self.classes, compression="gzip")
            f.create_dataset("foreground", data=self.foreground, compression="gzip")

    @staticmethod
    def load_h5(path):
        with h5py.File(path, "r") as f:
            angles = f["angles"][:]
            classes = f["classes"][:]
            foreground = f["foreground"][:]
        return ModelPredictions(angles=angles, classes=classes, foreground=foreground)


class SegmentationResults:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("segmentation", data=self.segmentation, compression="gzip")

    @staticmethod
    def load_h5(path):
        with h5py.File(path, "r") as f:
            segmentation = f["segmentation"][:]
        return SegmentationResults(segmentation=segmentation)
