import h5py


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
        return self.labels.shape[1]

    @property
    def y_extent_pixels(self):
        return self.labels.shape[0]

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
