import logging

import h5py
import numpy as np
import pandas
import torch
from blended_tiling import TilingModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import Optional
from nuc2seg.utils import generate_tiles, get_tile_bounds

logger = logging.getLogger(__name__)


class CelltypingResults:
    def __init__(
        self,
        aic_scores,
        bic_scores,
        expression_profiles,
        prior_probs,
        relative_expression,
        min_n_components,
        max_n_components,
        gene_names,
    ):
        self.aic_scores = aic_scores
        self.bic_scores = bic_scores
        self.expression_profiles = expression_profiles
        self.prior_probs = prior_probs
        self.relative_expression = relative_expression
        self.n_component_values = np.arange(min_n_components, max_n_components + 1)
        self.gene_names = gene_names

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("aic_scores", data=self.aic_scores, compression="gzip")
            f.create_dataset("bic_scores", data=self.bic_scores, compression="gzip")
            f.create_dataset(
                "n_component_values", data=self.n_component_values, compression="gzip"
            )
            f.create_dataset(
                "gene_names",
                data=np.char.encode(self.gene_names),
                compression="gzip",
            )
            for idx, k in enumerate(self.n_component_values):
                f.create_group(str(idx))
                f[str(idx)].create_dataset(
                    "expression_profiles",
                    data=self.expression_profiles[idx],
                    compression="gzip",
                )
                f[str(idx)].create_dataset(
                    "prior_probs",
                    data=self.prior_probs[idx],
                    compression="gzip",
                )
                f[str(idx)].create_dataset(
                    "relative_expression",
                    data=self.relative_expression[idx],
                    compression="gzip",
                )
                f[str(idx)].attrs["n_components"] = k

    @staticmethod
    def load_h5(path):
        with h5py.File(path, "r") as f:
            aic_scores = f["aic_scores"][:]
            bic_scores = f["bic_scores"][:]
            expression_profiles = []
            prior_probs = []
            relative_expression = []
            n_component_values = []
            gene_names = np.char.decode(f["gene_names"][:], "utf-8")
            for idx, k in enumerate(f["n_component_values"][:]):
                expression_profiles.append(f[str(idx)]["expression_profiles"][:])
                prior_probs.append(f[str(idx)]["prior_probs"][:])
                relative_expression.append(f[str(idx)]["relative_expression"][:])
                n_component_values.append(f[str(idx)].attrs["n_components"])
        return CelltypingResults(
            aic_scores=aic_scores,
            bic_scores=bic_scores,
            expression_profiles=expression_profiles,
            prior_probs=prior_probs,
            relative_expression=relative_expression,
            min_n_components=min(n_component_values),
            max_n_components=max(n_component_values),
            gene_names=gene_names,
        )


class RasterizedDataset:
    def __init__(self, labels, angles, transcripts, bbox, n_genes, resolution):
        self.labels = labels
        self.angles = angles
        self.transcripts = transcripts
        self.bbox = bbox
        self.n_genes = n_genes
        self.resolution = resolution

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=self.labels, compression="gzip")
            f.create_dataset("angles", data=self.angles, compression="gzip")
            f.create_dataset("transcripts", data=self.transcripts, compression="gzip")
            f.create_dataset("bbox", data=self.bbox)
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
            transcripts = f["transcripts"][:]
            bbox = f["bbox"][:]
            n_genes = f.attrs["n_genes"]
            resolution = f.attrs["resolution"]
        return RasterizedDataset(
            labels=labels,
            angles=angles,
            transcripts=transcripts,
            bbox=bbox,
            n_genes=n_genes,
            resolution=resolution,
        )


class Nuc2SegDataset:
    def __init__(
        self, labels, angles, classes, transcripts, bbox, n_classes, n_genes, resolution
    ):
        """
        :param labels: array of shape (x, y) of pixel labels,
            0 is background, -1 is border, > 1 is a unique contiguous nucleus
        :param angles: array of shape (x, y) of angles in radians
        :param classes: array of shape (x, y) of the cell type of each pixel, 1-indexed
        :param transcripts: array of shape (n_transcripts, 3), where the columns are x, y, gene_id
        :param bbox: array of shape (4,) of the bounding box of the dataset
        :param n_classes: int of the number of cell types
        :param n_genes: int of the number of genes
        :param resolution: float of the resolution of the dataset (width of a pixel in microns)
        """
        self.labels = labels.astype(int)
        self.angles = angles.astype(float)
        self.classes = classes.astype(int)
        self.transcripts = transcripts.astype(int)
        self.bbox = bbox
        self.n_classes = n_classes
        self.n_genes = n_genes
        self.resolution = resolution

    def save_h5(self, path, compression="gzip"):
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=self.labels, compression=compression)
            f.create_dataset("angles", data=self.angles, compression=compression)
            f.create_dataset("classes", data=self.classes, compression=compression)
            f.create_dataset(
                "transcripts", data=self.transcripts, compression=compression
            )
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

    @property
    def n_nuclei(self):
        return len(np.unique(self.labels[self.labels > 0]))

    @property
    def shape(self):
        return self.labels.shape

    def get_background_frequencies(self):
        """
        Returns the probability of observing a gene in a background pixel

        :return: torch.tensor of shape (n_genes,) of expected expression values
        """
        label_per_transcript = self.labels[
            self.transcripts[:, 0], self.transcripts[:, 1]
        ]

        df = pandas.DataFrame(
            {
                "gene": self.transcripts[:, 2],
                "label": label_per_transcript,
                "x": self.transcripts[:, 0],
                "y": self.transcripts[:, 1],
            }
        )

        selection_vector = df["label"] == 0
        df = df[selection_vector]
        n_background_transcripts = len(df)
        gene_counts = df.groupby(["x", "y", "gene"]).size().reset_index(name="count")

        d = (
            gene_counts.groupby("gene")["count"].sum() / n_background_transcripts
        ).to_dict()

        result = torch.zeros((self.n_genes,), dtype=torch.float)
        result[:] = 1e-10

        for k, v in d.items():
            result[k] = v

        return result

    def get_celltype_frequencies(self):
        """
        Returns the expected count of each gene in each cell type

        :return: dict[int, np.array] of celltype index -> expected expression vector
        """
        class_per_transcript = self.classes[
            self.transcripts[:, 0], self.transcripts[:, 1]
        ]

        # count n pixels per celltype
        celltype_pixel_totals = (
            pandas.Series(self.classes[self.classes > 0] - 1)
            .value_counts()
            .reset_index()
            .rename(columns={"index": "celltype", "count": "n_pixels"})
        )

        df = pandas.DataFrame(
            {
                "gene": self.transcripts[:, 2],
                "celltype": class_per_transcript,
                "x": self.transcripts[:, 0],
                "y": self.transcripts[:, 1],
            }
        )

        df = df[df["celltype"] > 0]
        df["celltype"] = df["celltype"] - 1
        df = df.drop_duplicates(subset=["x", "y", "gene"])

        per_gene_per_celltype_frequencies = (
            df.groupby(["gene", "celltype"]).size().reset_index(name="count")
        )

        per_gene_per_celltype_frequencies = per_gene_per_celltype_frequencies.merge(
            celltype_pixel_totals, left_on="celltype", right_on="celltype", how="left"
        )

        per_gene_per_celltype_frequencies["frequency"] = (
            per_gene_per_celltype_frequencies["count"]
            / per_gene_per_celltype_frequencies["n_pixels"]
        )

        d = per_gene_per_celltype_frequencies.set_index(["celltype", "gene"])[
            "frequency"
        ].to_dict()

        result = torch.zeros((self.n_classes, self.n_genes), dtype=torch.float)
        result[:] = 1e-10

        for (celltype, gene), freq in d.items():
            result[celltype, gene] = freq

        return result

    @staticmethod
    def load_h5(
        path,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        tile_overlap: Optional[float] = None,
        tile_index: Optional[int] = None,
    ):
        with h5py.File(path, "r") as f:
            if (
                tile_width is not None
                and tile_height is not None
                and tile_overlap is not None
            ):
                x1, y1, x2, y2 = get_tile_bounds(
                    tile_width=tile_width,
                    tile_height=tile_height,
                    tile_overlap=tile_overlap,
                    tile_index=tile_index,
                    base_width=f["angles"].shape[0],
                    base_height=f["angles"].shape[1],
                )
                labels = f["labels"][x1:x2, y1:y2]
                angles = f["angles"][x1:x2, y1:y2]
                classes = f["classes"][x1:x2, y1:y2]
                n_classes = f.attrs["n_classes"]
                n_genes = f.attrs["n_genes"]
                resolution = f.attrs["resolution"]
                original_bbox = f["bbox"][:]
                transcripts = f["transcripts"][:]
                transcript_selector = (
                    (transcripts[:, 0] >= x1)
                    & (transcripts[:, 0] < x2)
                    & (transcripts[:, 1] >= y1)
                    & (transcripts[:, 1] < y2)
                )
                new_bbox = np.array(
                    [
                        original_bbox[0] + x1,
                        original_bbox[1] + y1,
                        original_bbox[0] + x2,
                        original_bbox[1] + y2,
                    ]
                )

                new_transcripts = transcripts[transcript_selector].copy()
                new_transcripts[:, 0] = new_transcripts[:, 0] - x1
                new_transcripts[:, 1] = new_transcripts[:, 1] - y1

                current_labels = np.unique(labels)
                current_labels = current_labels[current_labels > 0]

                for current_nuc_index, monotonically_increasing_index in zip(
                    current_labels, range(1, len(current_labels) + 1)
                ):
                    labels[labels == current_nuc_index] = monotonically_increasing_index

                return Nuc2SegDataset(
                    labels=labels,
                    angles=angles,
                    classes=classes,
                    transcripts=new_transcripts,
                    bbox=new_bbox,
                    n_classes=n_classes,
                    n_genes=n_genes,
                    resolution=resolution,
                )
            else:
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

    def clip(self, bbox):
        transcript_selector = (
            (self.transcripts[:, 0] >= bbox[0])
            & (self.transcripts[:, 0] < bbox[2])
            & (self.transcripts[:, 1] >= bbox[1])
            & (self.transcripts[:, 1] < bbox[3])
        )
        new_bbox = np.array(
            [
                bbox[0] + self.bbox[0],
                bbox[1] + self.bbox[1],
                bbox[2] + self.bbox[0],
                bbox[3] + self.bbox[1],
            ]
        )

        new_transcripts = self.transcripts[transcript_selector].copy()
        new_transcripts[:, 0] = new_transcripts[:, 0] - bbox[0]
        new_transcripts[:, 1] = new_transcripts[:, 1] - bbox[1]

        new_labels = self.labels[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()
        current_labels = np.unique(new_labels)
        current_labels = current_labels[current_labels > 0]

        for current_nuc_index, monotonically_increasing_index in zip(
            current_labels, range(1, len(current_labels) + 1)
        ):
            new_labels[new_labels == current_nuc_index] = monotonically_increasing_index

        return Nuc2SegDataset(
            labels=new_labels,
            angles=self.angles[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy(),
            classes=self.classes[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy(),
            transcripts=new_transcripts,
            bbox=new_bbox,
            n_classes=self.n_classes,
            n_genes=self.n_genes,
            resolution=self.resolution,
        )


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
        weights = torch.tensor(
            self.per_tile_class_histograms[:, 2:].mean()
            / self.per_tile_class_histograms[:, 2:].mean(axis=0)
        )

        if not torch.all(torch.isfinite(weights)):
            logger.warning("Non-finite weights found. Replacing with epsilon")
            weights[~torch.isfinite(weights)] = 1e-6

        return weights

    def _get_tile(self, x1, y1, x2, y2):
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
            "X": torch.as_tensor(tile_transcripts[:, 0]).long(),
            "Y": torch.as_tensor(tile_transcripts[:, 1]).long(),
            "gene": torch.as_tensor(tile_transcripts[:, 2]).long(),
            "labels": torch.as_tensor(tile_labels).long(),
            "angles": torch.as_tensor(tile_angles).float(),
            "classes": torch.as_tensor(tile_classes).long(),
            "label_mask": torch.as_tensor(labels_mask).bool(),
            "nucleus_mask": torch.as_tensor(nucleus_mask).bool(),
        }

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
        return self._get_tile(x1, y1, x2, y2)

    def __getitems__(self, indices: list[int]):
        tile_generator = generate_tiles(
            tiler=self.tiler,
            x_extent=self.ds.x_extent_pixels,
            y_extent=self.ds.y_extent_pixels,
            tile_size=(self.tile_height, self.tile_width),
            overlap_fraction=self.tile_overlap,
            tile_ids=indices,
        )

        tiles = []
        for x1, y1, x2, y2 in tile_generator:
            tiles.append(self._get_tile(x1, y1, x2, y2))

        return tiles


class TrainTestSplit:
    def __init__(self, train_indices, test_indices):
        self.train_indices = train_indices
        self.test_indices = test_indices

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "train_indices", data=self.train_indices, compression="gzip"
            )
            f.create_dataset("test_indices", data=self.test_indices, compression="gzip")

    @staticmethod
    def load_h5(path):
        with h5py.File(path, "r") as f:
            train_indices = f["train_indices"][:]
            test_indices = f["test_indices"][:]
        return TrainTestSplit(train_indices=train_indices, test_indices=test_indices)


class ModelPredictions:
    def __init__(self, angles, classes, foreground):
        """
        :param angles: array of shape (x, y) of angles in radians
        :param classes: array of shape (x, y, n_classes) of class predictions
        :param foreground: array of shape (x, y) of foreground probabilities
        """
        self.angles = angles
        self.classes = classes
        self.foreground = foreground

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("angles", data=self.angles, compression="gzip")
            f.create_dataset("classes", data=self.classes, compression="gzip")
            f.create_dataset("foreground", data=self.foreground, compression="gzip")

    @staticmethod
    def load_h5(
        path,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        tile_overlap: Optional[float] = None,
        tile_index: Optional[int] = None,
    ):

        with h5py.File(path, "r") as f:
            if (
                tile_width is not None
                and tile_height is not None
                and tile_overlap is not None
            ):
                x1, y1, x2, y2 = get_tile_bounds(
                    tile_width=tile_width,
                    tile_height=tile_height,
                    tile_overlap=tile_overlap,
                    tile_index=tile_index,
                    base_width=f["angles"].shape[0],
                    base_height=f["angles"].shape[1],
                )
                angles = f["angles"][x1:x2, y1:y2]
                classes = f["classes"][x1:x2, y1:y2, :]
                foreground = f["foreground"][x1:x2, y1:y2]
                return ModelPredictions(
                    angles=angles, classes=classes, foreground=foreground
                )

            else:
                angles = f["angles"][:]
                classes = f["classes"][:]
                foreground = f["foreground"][:]
                return ModelPredictions(
                    angles=angles, classes=classes, foreground=foreground
                )

    def shape(self):
        return self.angles.shape

    def clip(self, bbox):
        return ModelPredictions(
            angles=self.angles[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy(),
            classes=self.classes[bbox[0] : bbox[2], bbox[1] : bbox[3], :].copy(),
            foreground=self.foreground[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy(),
        )


class SegmentationResults:
    def __init__(self, segmentation):
        """
        :param segmentation: array of shape (x, y) of segment ids, 0 is background, -1 is border,
            > 1 is a unique contiguous segment
        """
        self.segmentation = segmentation

    def save_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("segmentation", data=self.segmentation, compression="gzip")

    @staticmethod
    def load_h5(path):
        with h5py.File(path, "r") as f:
            segmentation = f["segmentation"][:]
        return SegmentationResults(segmentation=segmentation)
