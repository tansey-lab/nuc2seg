import matplotlib.pyplot as plt
import numpy as np


def sqexp(x1, x2, bandwidth=2, scale=1, axis=None):
    return scale * np.exp(-np.linalg.norm(x1 - x2, axis=axis) ** 2 / bandwidth**2)


def sample_gp_fast(width, height, step=10, size=1, bandwidth=8, scale=3):
    # Simple GP on a smaller space
    XY_grid_small = np.array(
        np.meshgrid(np.arange(width, step=step), np.arange(height, step=step))
    ).T.reshape(-1, 2)
    Cov = np.array(
        [
            (sqexp(x1[None], XY_grid_small, bandwidth=bandwidth, scale=scale, axis=1))
            for x1 in XY_grid_small
        ]
    )
    Cov += np.diag(np.ones(XY_grid_small.shape[0]) * 1e-3)
    small_means = np.random.multivariate_normal(np.ones(XY_grid_small.shape[0]), Cov)

    # Interpolate to the higher space using splines
    XY_grid = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(
        -1, 2
    )
    ## scipy 1.7.0 version
    # from scipy.interpolate import RBFInterpolator
    # means = RBFInterpolator(XY_grid_small, small_means)(XY_grid)
    ## scipy 1.6.2 version
    from scipy.interpolate import Rbf

    rbfi = Rbf(XY_grid_small[:, 0], XY_grid_small[:, 1], small_means)
    means = rbfi(XY_grid[:, 0], XY_grid[:, 1])

    return XY_grid, means


def sample_poisson_points_fast(n_points, width, height):
    from scipy.special import softmax

    XY_grid, log_means = sample_gp_fast(width, height)
    probs = softmax(log_means)
    return XY_grid[
        np.random.choice(XY_grid.shape[0], size=n_points, p=probs, replace=False)
    ]


def estimate_flow(
    Ligands,
    Receptors,
    ligands_per_receptor=1,
    lam=1,
    lam_min=1e-1,
    lam_max=500,
    n_lam=30,
    penalty_type="l1",
):
    import torch
    from autograd_minimize import minimize

    # Calculate Ligand-Receptor distances
    Distances = np.linalg.norm((Ligands[:, None] - Receptors[None]) ** 2, axis=2)
    t_Distances = torch.Tensor(Distances)

    # Initialize
    cur_Rates = np.ones_like(Distances)

    def loss(logits):
        # Likelihood
        weights = torch.nn.Softmax(dim=1)(logits)
        log_like = (weights * t_Distances).mean()

        # Penalty constraining ligands to balance across receptors
        if penalty_type == "l1":
            penalty = torch.abs(weights.sum(axis=0) - ligands_per_receptor).mean()
        elif penalty_type == "l2":
            penalty = ((weights.sum(axis=0) - ligands_per_receptor) ** 2).mean()

        return log_like + lam * penalty

    # Optimize using a 2nd order method with autograd for gradient calculation.
    res = minimize(
        loss,
        cur_Rates,
        method="L-BFGS-B",
        backend="torch",
        bounds=(1e-4, None),
        tol=1e-6,
    )
    cur_Rates = res.x

    #### If you want to do this over a grid of lams...
    # lams = np.exp(np.linspace(np.log(lam_min), np.log(lam_max), n_lam))
    # for lam_idx, lam in tqdm(enumerate(lams)):
    #     def loss(logits):
    #         # Likelihood
    #         weights = torch.nn.Softmax(dim=1)(logits)
    #         log_like = (weights * t_Distances).mean()

    #         # Penalty constraining ligands to balance across receptors
    #         if penalty_type == 'l1':
    #             penalty = torch.abs(weights.sum(axis=0) - ligands_per_receptor).mean()
    #         elif penalty_type == 'l2':
    #             penalty = ((weights.sum(axis=0) - ligands_per_receptor)**2).mean()

    #         return log_like + lam*penalty

    #     # Optimize using a 2nd order method with autograd for gradient calculation.
    #     res = minimize(loss, cur_Rates, method='L-BFGS-B', backend='torch', bounds=(1e-4,None), tol=1e-6)
    #     cur_Rates = res.x
    ####

    from scipy.special import softmax

    return softmax(cur_Rates, axis=1)


def visualize_flow(Ligands, Receptors, Grid, Flows):
    # Calculate the distance from each grid point to each ligand
    # grid_distances = np.linalg.norm(Grid[:,None] - Ligands[None], axis=2)
    vecs = Grid[:, None] - Ligands[None]
    scale, bandwidth = 2, 10
    grid_distances = scale * np.exp(-np.linalg.norm(vecs, axis=2) ** 2 / bandwidth**2)
    grid_distances /= grid_distances.sum(axis=1, keepdims=True)

    # Calculate the average flow vector of each ligand
    ligand_vectors = ((Receptors[None] - Ligands[:, None]) * Flows[..., None]).sum(
        axis=1
    )

    # Calculate the weighted average flow vector of each grid point
    grid_vectors = (grid_distances[..., None] * ligand_vectors[None]).sum(axis=1)

    # Normalize everything
    grid_vectors = grid_vectors / (np.sqrt(2) * np.abs(grid_vectors).max(axis=0))

    # Plot the flow vectors
    for i, ((y, x), (dy, dx)) in enumerate(zip(Grid, grid_vectors)):
        print(i)
        plt.arrow(
            x,
            y,
            dx * 10,
            dy * 10,
            width=0.2 * np.sqrt(dx**2 + dy**2),
            head_width=0.3 * np.sqrt(dx**2 + dy**2),
            color=(dx / 2 + 0.5, dy / 2 + 0.5, np.sqrt(dx**2 + dy**2)),
        )
    plt.savefig(f"plots/flows.pdf", bbox_inches="tight")
    plt.close()


def generate_fake_data():
    width, height = 300, 300
    n_ligands = 500
    n_receptors = 250

    Ligands = sample_poisson_points_fast(n_ligands, width, height)
    Receptors = sample_poisson_points_fast(n_receptors, width, height)

    return width, height, Ligands, Receptors


def load_real_data(ligand_name="CXCL12", receptor_name="CXCR4", gene_name="VEGFA"):
    import pandas as pd

    df = pd.read_csv("data/transcripts.csv", header=0)

    # Load the ligands and receptors
    ligand_idxs = np.arange(df.shape[0])[
        (df["feature_name"] == ligand_name) & (df["qv"] >= 20)
    ]
    receptor_idxs = np.arange(df.shape[0])[
        (df["feature_name"] == receptor_name) & (df["qv"] >= 20)
    ]
    Ligands = df.iloc[ligand_idxs][["x_location", "y_location"]].values
    Receptors = df.iloc[receptor_idxs][["x_location", "y_location"]].values

    Ligand_cells = df.iloc[receptor_idxs]

    # Get the
    Genes = None  # df[(df['feature_name'] == gene_name) & (df['qv'] >= 20)][['x_location', 'y_location']].values

    # Subsample to 2% of data for a quick look at things
    ligand_cells = np.random.choice(
        Ligands.shape[0], replace=False, size=Ligands.shape[0] // 50
    )
    receptor_cells = np.random.choice(
        Receptors.shape[0], replace=False, size=Receptors.shape[0] // 50
    )
    Ligands = Ligands[ligand_cells]
    Receptors = Receptors[receptor_cells]
    Genes = Genes[receptor_cells]

    return (
        int(max(Ligands[:, 1].max(), Receptors[:, 1].max()) + 10),
        int(max(Ligands[:, 0].max(), Receptors[:, 0].max()) + 10),
        Ligands,
        Receptors,
        Genes,
    )


if __name__ == "__main__":
    # width, height, Ligands, Receptors = generate_fake_data()
    width, height, Ligands, Receptors, Genes = load_real_data()
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    axarr[0].scatter(Ligands[:, 0], Ligands[:, 1], s=2)
    axarr[1].scatter(Receptors[:, 0], Receptors[:, 1], s=2)
    plt.savefig("plots/raw.pdf", bbox_inches="tight")
    plt.close()

    # Solve the max flow problem
    Flows = estimate_flow(Ligands, Receptors, lam=0.01, penalty_type="l1")

    # Visualize the solution as L-R flows
    min_prob = 0.1
    ligand_vectors = Receptors[None] - Ligands[:, None]
    max_vec = np.linalg.norm(ligand_vectors, axis=1).max()
    for i, ((y, x), sinks) in enumerate(zip(Ligands, ligand_vectors)):
        print(i)
        for j, (dy, dx) in enumerate(sinks):
            if Flows[i, j] < min_prob:
                continue
            plt.arrow(
                x,
                y,
                dx * 0.8,
                dy * 0.8,
                width=2 * Flows[i, j],
                head_width=2 * Flows[i, j],
                color="black",
                alpha=Flows[i, j],
            )
    plt.scatter(
        Ligands[:, 1], Ligands[:, 0], s=3, color="red", zorder=100, label="Ligands"
    )
    plt.scatter(
        Receptors[:, 1],
        Receptors[:, 0],
        s=3,
        color="blue",
        zorder=100,
        label="Receptors",
    )
    # Put a legend to the right of the plot
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"plots/flows-cells.pdf", bbox_inches="tight")
    plt.close()

    # Visualize the solution on a grid
    Grid = np.array(
        np.meshgrid(
            np.arange(width, step=max(10, width // 100)),
            np.arange(height, step=max(10, height // 100)),
        )
    ).T.reshape(-1, 2)
    visualize_flow(Ligands, Receptors, Grid, Flows)
