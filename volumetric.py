"""
Example usage:
python3.11 -m venv venv
source venv/bin/activate
pip install pyvista
pip install 'git+https://github.com/tansey-lab/nuc2seg.git'
python plot.py --bbox "1500,1500,1750,1750" \
    --ome-tiff /Users/quinnj2/Downloads/xenium_example/morphology.ome.tif \
    --transcripts /Users/quinnj2/Downloads/xenium_example/transcripts.parquet \
    --nuclei /Users/quinnj2/Downloads/xenium_example/nucleus_boundaries.parquet
"""

import argparse

import numpy as np
import pyvista as pv
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", help="Path to pickle file")
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    pixel_size_x = data["pixel_size_x"]
    pixel_size_y = data["pixel_size_y"]
    pixel_size_z = data["pixel_size_z"]
    img_array = data["dapi"]
    p = pv.Plotter()

    def add_nucleus(height):
        for idx, poly in enumerate(data["nuclei"].geometry):
            try:
                points_2d = np.array(list(poly.exterior.coords))
                points_3d = np.pad(points_2d, [(0, 0), (0, 1)])  # shape (N, 3)
                polygon = pv.lines_from_points(points_3d, close=True)
                # extrude along z and plot
                boundary = polygon.extrude((0, 0, height))
                p.add_mesh(boundary, color="white", name=f"nucleus_{idx}")
            except:
                continue

    add_nucleus(3)

    gene_points = data["transcripts"][["x_location", "y_location", "z_location"]].values

    gene_points_actor = p.add_points(gene_points, color="red")

    def toggle_vis(flag):
        gene_points_actor.SetVisibility(flag)

    p.add_checkbox_button_widget(toggle_vis, value=True)

    class CustomSliderUpdateRoutine:
        def __init__(self, p):
            # default parameters
            self.p = p
            self.kwargs = {"threshold": 50, "max_z": 12}

        def __call__(self, param, value):
            self.kwargs[param] = value
            self.update()

        def update(self):
            # This is where you call your simulation
            data = img_array.copy()
            cutoff_high = np.percentile(data, 99.5)
            cutoff_low = np.percentile(data, self.kwargs["threshold"])
            data[data > cutoff_high] = cutoff_high
            data[data < cutoff_low] = 0.0

            selection_vector = data > 0.0
            data[selection_vector] = data[selection_vector] - np.min(
                data[selection_vector]
            )
            data[:, :, int(self.kwargs["max_z"]) :] = 0.0

            self.p.add_volume(
                volume=data,
                opacity="linear",
                name="DAPI",
                cmap="Blues",
                show_scalar_bar=False,
                resolution=[1, 1, pixel_size_z],
            )

    engine = CustomSliderUpdateRoutine(p)
    engine.update()

    p.add_slider_widget(
        callback=lambda value: engine("threshold", int(value)),
        rng=[0, 99.5],
        value=50,
        title="DAPI Threshold",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style="modern",
    )
    p.add_slider_widget(
        callback=lambda value: engine("max_z", int(value)),
        rng=[1, 12],
        value=12,
        title="Max Z",
        pointa=(0.35, 0.1),
        pointb=(0.64, 0.1),
        style="modern",
    )
    p.add_slider_widget(
        callback=lambda value: add_nucleus(value),
        rng=[1, 50],
        value=1,
        title="Extrude Nucleus",
        pointa=(0.67, 0.1),
        pointb=(0.98, 0.1),
        style="modern",
    )

    p.show()


if __name__ == "__main__":
    main()
