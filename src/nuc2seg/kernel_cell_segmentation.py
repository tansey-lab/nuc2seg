import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import poisson
import torch
from torch.distributions import Poisson
from autograd_minimize import minimize
from tqdm import tqdm
from nuc2seg.utils import grid_graph_edges, calc_plateaus
from sklearn.cluster import KMeans


def calculate_neighbor_weights(
    transcript_xy, cell_xy, max_distance, nucleus_size, bandwidth
):
    distances = np.linalg.norm(transcript_xy[:, None] - cell_xy[None], axis=2)
    mask = distances <= max_distance
    transcript_neighbors = []
    neighbor_weights = []
    known_cells = np.zeros(transcript_xy.shape[0], dtype=int) - 1
    for i in range(transcript_xy.shape[0]):
        neighbors = np.where(mask[i])[0]
        dists = distances[i, neighbors]

        # Assign transcripts that are too far from any nucleus to noise
        if len(neighbors) == 0:
            neighbors = np.array([-1])
            weights = np.array([1])
            known_cells[i] = 0
        # If the transcript is on the nucleus, hard-code it to be from this cell (or noise)
        elif dists.min() < nucleus_size:
            nuc = neighbors[np.argmin(dists)]
            neighbors = np.array([nuc])
            weights = np.array([1])
            known_cells[i] = nuc + 1
        else:
            # Otherwise, use a kernel smoothed distance
            weights = np.exp(-(dists**2) / bandwidth**2)
        neighbor_weights.append(weights)
        transcript_neighbors.append(neighbors + 1)
    return transcript_neighbors, neighbor_weights, known_cells


def calculate_transcript_cell_probs(neighbor_types, w, gene, model_gene_probs):
    # Get the current local spatial weights based on cell typing
    w = w[np.arange(w.shape[0]), neighbor_types - 1]

    # Add the background weight
    w = np.hstack([w, [0.1]])

    # Get the likelihood of the gene given the cell types
    likelihoods = model_gene_probs[neighbor_types - 1, gene]

    # Add the background uniform likelihood
    likelihoods = np.hstack([likelihoods, [1 / n_genes]])

    # Calculate the posterior probabilities
    probs = w * likelihoods
    probs = probs / probs.sum()
    return probs


# E-step: Run a Gibbs sampler to estimate the joint posterior probs of cell types and transcript cells
def e_step(
    transcript_genes,
    transcript_neighbors,
    known_transcript_cells,
    local_weights,
    model_amplitudes,
    model_gene_probs,
    n_cells,
    init_cell_types=None,
    n_burn=100,
    n_thin=5,
    n_samples=100,
):
    # Initialize the cell assignments
    n_transcripts = transcript_genes.shape[0]
    cur_transcript_cells = np.array(
        [
            z if z != -1 else np.random.choice(neighbors)
            for z, neighbors in zip(known_transcript_cells, transcript_neighbors)
        ]
    )

    # Assign cells to closest matching type if not initially assigned
    n_cell_types = model_gene_probs.shape[0]
    n_genes = model_gene_probs.shape[1]
    if init_cell_types is None:
        cell_transcript_counts = np.zeros((n_cells, n_genes))
        np.add.at(
            cell_transcript_counts,
            (
                cur_transcript_cells[cur_transcript_cells != 0] - 1,
                transcript_genes[cur_transcript_cells != 0],
            ),
            1,
        )
        cell_transcript_counts /= cell_transcript_counts.sum(axis=1, keepdims=True)
        cur_cell_types = (
            np.argmin(
                np.linalg.norm(
                    cell_transcript_counts[:, None] - model_gene_probs[None], axis=2
                ),
                axis=1,
            )
            + 1
        )
    else:
        cur_cell_types = np.array(init_cell_types)

    # Track which transcripts do not need to be updated
    known_mask = known_transcript_cells != -1

    transcript_cell_samples = np.zeros(
        (n_samples, cur_transcript_cells.shape[0]), dtype=int
    )
    cell_type_samples = np.zeros((n_samples, cur_cell_types.shape[0]), dtype=int)
    for step in range(n_burn + n_thin * n_samples):
        if (step == 0) or (((step + 1) % 100) == 0):
            print(f"\t\tGibbs step {step+1}/{n_burn + n_thin*n_samples}")
        #### Sample transcript cell assignments given cell types
        for i in range(n_transcripts):
            # Skip transcripts that have been deterministically assigned (e.g. noise and nucleus transcripts)
            if known_mask[i]:
                continue

            # Calculate the probability of assigning the transcript to each neighboring cell
            probs = calculate_transcript_cell_probs(
                cur_cell_types[transcript_neighbors[i] - 1],
                local_weights[i],
                transcript_genes[i],
                model_gene_probs,
            )

            # Sample the cell assignment
            sidx = np.random.choice(probs.shape[0], p=probs)
            if sidx == (probs.shape[0] - 1):
                cur_transcript_cells[i] = 0
            else:
                cur_transcript_cells[i] = transcript_neighbors[i][sidx]

        #### Sample cell type assignments given transcript cell assignments
        cell_transcript_counts = np.zeros((n_cells, n_genes))
        np.add.at(
            cell_transcript_counts,
            (
                cur_transcript_cells[cur_transcript_cells != 0] - 1,
                transcript_genes[cur_transcript_cells != 0],
            ),
            1,
        )
        cell_type_rates = model_gene_probs * model_amplitudes[:, None]

        # Total transcript count is Poisson(tau * theta) because the w's have to integrate to 1
        logits = poisson.logpmf(cell_transcript_counts[:, None], cell_type_rates[None])
        logits[np.isnan(logits)] = -20
        logits = np.clip(logits, -20, 20)
        logits = logits.sum(axis=2)
        probs = softmax(logits, axis=1)

        # Sample cell type assignments
        cur_cell_types = np.array([np.random.choice(n_cell_types, p=p) for p in probs])

        if ((step - n_burn) % n_thin) == 0:
            sample_idx = (step - n_burn) // n_thin
            transcript_cell_samples[sample_idx] = cur_transcript_cells
            cell_type_samples[sample_idx] = cur_cell_types

    return transcript_cell_samples, cell_type_samples


def fit_fused_lasso_count_process(
    counts, lam_min=1e-2, lam_max=50, n_lam=30, segmentation_threshold=0.1
):

    # Estimates of the latent rates
    Rates = np.zeros((n_lam,) + counts.shape)

    # Build the torch data tensors
    t_Data = torch.Tensor(counts)

    cur_Rates = np.ones(counts.shape) * counts.mean()
    lams = np.exp(np.linspace(np.log(lam_min), np.log(lam_max), n_lam))[::-1]
    for lam_idx, lam in tqdm(enumerate(lams)):
        # Estimate the rates as Poisson counts with a fused lasso penalty
        def loss(t_rates):
            # Log-probability of the observations
            pois = Poisson(t_rates)
            l = -pois.log_prob(t_Data).mean()
            if torch.isnan(l):
                print(f"loss went to nan for lam={lam:.2f}")

            # Add the fused lasso penalty
            if lam > 0:
                rows = torch.abs(t_rates[1:] - t_rates[:-1]).reshape(-1) ** 2
                cols = torch.abs(t_rates[:, 1:] - t_rates[:, :-1]).reshape(-1) ** 2
                diag1 = torch.abs(t_rates[1:, 1:] - t_rates[:-1, :-1]).reshape(-1) ** 2
                diag2 = torch.abs(t_rates[1:, :-1] - t_rates[:-1, 1:]).reshape(-1) ** 2
                l += (
                    lam
                    * (rows.sum() + cols.sum() + diag1.sum() + diag2.sum())
                    / (rows.shape[0] + cols.shape[0] + diag1.shape[0] + diag2.shape[0])
                )

            return l

        # Optimize using a 2nd order method with autograd for gradient calculation.
        res = minimize(
            loss,
            cur_Rates,
            method="L-BFGS-B",
            backend="torch",
            bounds=(1e-5, None),
            tol=1e-6,
        )
        cur_Rates = res.x

        # Save the results
        Rates[lam_idx] = cur_Rates

    # Calculate the BIC scores
    from scipy.stats import poisson

    edges = grid_graph_edges(counts.shape[0], counts.shape[1])
    dof = np.array(
        [
            len(
                calc_plateaus(
                    (rates.reshape(-1) > rates.max() * segmentation_threshold).astype(
                        int
                    ),
                    edges,
                    rel_tol=1e-2,
                )
            )
            for rates in Rates
        ]
    )
    loglike = poisson(Rates).logpmf(counts[None]).sum(axis=(1, 2))
    bic_scores = dof * np.log(np.prod(counts.shape)) - 2 * loglike

    # Estimate a homogeneous poisson process as the final mask
    best_idx = np.argmin(np.abs(dof - 2))
    best_segmentation = Rates[best_idx] > Rates[best_idx].max() * segmentation_threshold
    Rates_final = np.zeros_like(Rates[best_idx])
    Rates_final[best_segmentation] = counts.sum() / best_segmentation.sum()

    return (
        Rates_final,
        Rates[::-1],
        lams[::-1],
        dof[::-1],
        loglike[::-1],
        bic_scores[::-1],
    )


def test_fused_lasso_count_process():
    from scipy.stats import multivariate_normal

    width = 10
    height = 10
    grid = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(-1, 2)

    total = 15
    mean = np.array([5, 5])
    cov = np.array([[3, 1.5], [1.5, 3]])
    rv = multivariate_normal(mean, cov)
    Rates = np.zeros((width, height))
    Rates[grid[:, 0], grid[:, 1]] = rv.pdf(grid)
    Rates = (Rates >= np.quantile(Rates, 0.65)).astype(int)
    Rates = Rates / Rates.sum()
    Rates *= total

    counts = np.random.poisson(Rates)
    Rates_final, Rates_hat, lams, dofs, loglikes, bic_scores = (
        fit_fused_lasso_count_process(counts)
    )

    fig, axarr = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    axarr[0].imshow(Rates, vmin=Rates_hat.min(), vmax=Rates_hat.max())
    axarr[0].set_title("Truth")
    axarr[1].imshow(Rates_final, vmin=Rates_hat.min(), vmax=Rates_hat.max())
    axarr[1].set_title("Best estimate")
    plt.show()

    fig, axarr = plt.subplots(1, 6, figsize=(25, 5), sharex=True, sharey=True)
    axarr[0].imshow(Rates, vmin=Rates_hat.min(), vmax=Rates_hat.max())
    axarr[0].set_title("Truth")
    for i, j in enumerate([0, 10, 22, 26, np.argmin(np.abs(dofs - 2))]):
        axarr[i + 1].imshow(
            Rates_hat[j] * (Rates_hat[j] > Rates_hat[j].max() * 0.1),
            vmin=Rates_hat.min(),
            vmax=Rates_hat.max(),
        )
        axarr[i + 1].set_title(
            f"lam={lams[j]:.2f} dof={dofs[j]}\nBIC={bic_scores[j]:.2f}"
        )
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # Model hyperparameters
    bandwidth = 4  # Kernel smoothing bandwidth
    max_distance = 10  # Maximum distance that a transcript can be from a nucleus
    max_steps = 10  # Maximum number of steps to take in the EM algorithm
    tol = 1e-4  # Convergence tolerance
    gibbs_burn = 50
    gibbs_thin = 2
    gibbs_samples = 100

    # Precompute the neighborhoods, distance weights, and hard-coded cell assignments
    transcript_neighbors, neighbor_weights, known_transcript_cells = (
        calculate_neighbor_weights(
            transcript_xy, cell_xy, max_distance, nucleus_size, bandwidth
        )
    )
    known_mask = known_transcript_cells != -1

    # Initialize transcript cell assignments just on distance
    nearest_cell = np.array(
        [t[np.argmax(w)] for t, w in zip(transcript_neighbors, neighbor_weights)],
        dtype=int,
    )
    cell_transcript_counts = np.zeros((n_cells, n_genes))
    np.add.at(
        cell_transcript_counts,
        (
            nearest_cell[known_transcript_cells != 0] - 1,
            transcript_genes[known_transcript_cells != 0],
        ),
        1,
    )

    # Initialize cell type profiles and cell type IDs via K-means

    kmeans = KMeans(n_clusters=n_cell_types).fit(cell_transcript_counts)
    model_cell_types = kmeans.labels_ + 1
    model_amplitudes = (kmeans.cluster_centers_ + 1).sum(axis=1)
    model_gene_probs = (kmeans.cluster_centers_ + 1) / model_amplitudes[:, None]
    # model_cell_type_prior = (kmeans.labels_[:,None] == np.arange(n_cell_types)[None]).sum(axis=0)
    # model_cell_type_prior = (model_cell_type_prior+1) / (model_cell_type_prior+1).sum()

    # Re-initialize to probabilistic assignment for use in the E step.
    # cell_type_dists = 1/(np.linalg.norm(cell_transcript_counts[:,None] - kmeans.cluster_centers_[None], axis=2) + 1e-2)
    # model_cell_types = cell_type_dists / cell_type_dists.sum(axis=1, keepdims=True)

    # Run the EM algorithm to convergence
    prev_gene_probs = np.zeros_like(model_gene_probs)
    prev_amplitudes = np.zeros_like(model_amplitudes)
    for step in range(max_steps):
        print(f"EM step {step+1}/{max_steps}")
        # Compute the weighted prior probs
        local_weights = [w[:, None] * model_amplitudes[None] for w in neighbor_weights]

        # E-step: Run a Gibbs sampler to estimate the joint posterior of cell types and transcript cells
        transcript_cell_samples, cell_type_samples = e_step(
            transcript_genes,
            transcript_neighbors,
            known_transcript_cells,
            local_weights,
            model_amplitudes,
            model_gene_probs,
            n_cells,
            init_cell_types=model_cell_types,
            n_burn=gibbs_burn,
            n_thin=gibbs_thin,
            n_samples=gibbs_samples,
        )

        # Collapse down to sufficient statistics: for each posterior sample,
        # how many counts of each gene were assigned to each cell?
        cell_gene_counts = np.zeros((gibbs_samples, n_cells, n_genes), dtype=int)
        foreground_mask = transcript_cell_samples != 0
        n_row = transcript_cell_samples.shape[0]
        n_col = transcript_cell_samples.shape[1]
        row_idx = np.repeat(np.arange(n_row), n_col).reshape(n_row, n_col)[
            foreground_mask
        ]
        col_idx = (
            np.repeat(np.arange(n_col), n_row).reshape(n_col, n_row).T[foreground_mask]
        )
        np.add.at(
            cell_gene_counts,
            (
                row_idx,
                transcript_cell_samples[foreground_mask] - 1,
                transcript_genes[col_idx],
            ),
            1,
        )

        # Collapse further: group count vectors by the cell type
        count_vectors = [
            cell_gene_counts[cell_type_samples == k] for k in range(n_cell_types)
        ]

        # Simple MAP: average the rates but use a pseudo-count
        for k in range(n_cell_types):
            rates = (count_vectors[k].sum(axis=0) + 1) / (count_vectors[k].shape[0] + 1)
            model_gene_probs[k] = rates / rates.sum()
            model_amplitudes[k] = rates.sum()

        # Check convergence for early stopping
        converge1 = np.linalg.norm(prev_gene_probs - model_gene_probs)
        converge2 = np.linalg.norm(prev_amplitudes - model_amplitudes)
        converge = max(converge1, converge2)
        print(f"Convergence: {converge:.4f}")
        if converge <= tol:
            print("Stopping early.")
            break

        prev_gene_probs = np.array(model_gene_probs)
        prev_amplitudes = np.array(model_amplitudes)

    # Run the E step one last time to get robust classification
    final_burn = 1000
    final_thin = 10
    final_samples = 100
    # Compute the weighted prior probs
    local_weights = [w[:, None] * model_amplitudes[None] for w in neighbor_weights]

    # E-step: Run a Gibbs sampler to estimate the joint posterior of cell types and transcript cells
    transcript_cell_samples, cell_type_samples = e_step(
        transcript_genes,
        transcript_neighbors,
        known_transcript_cells,
        local_weights,
        model_amplitudes,
        model_gene_probs,
        n_cells,
        init_cell_types=model_cell_types,
        n_burn=final_burn,
        n_thin=final_thin,
        n_samples=final_samples,
    )

    ##### Take the last gibbs sample as the final answer
    model_transcript_cells = transcript_cell_samples[-1]
    model_cell_types = cell_type_samples[-1]

    ##### Do a hardcore assignment
    # # Assign the MAP cell type first
    # from scipy.stats import mode
    # model_cell_types = mode(cell_type_samples, axis=1)

    # # With fixed cell types, pick the most likely cell to assign each transcript
    # model_transcript_cells = np.zeros(n_transcripts, dtype=int)
    # for i in range(n_transcripts):
    #     probs = calculate_transcript_cell_probs(model_cell_types[transcript_neighbors[i]-1], local_weights[i], transcript_genes[i])
    #     sidx = np.argmax(probs)
    #     if sidx == (probs.shape[0]-1):
    #         model_transcript_cells[i] = 0
    #     else:
    #         model_transcript_cells[i] = transcript_neighbors[i][sidx]

    # Plot the initial estimates based just on k-means and nearest neighbors
    plt.scatter(
        cell_xy[:, 0], cell_xy[:, 1], c=cell_types, marker="*", s=200, zorder=100
    )
    plt.scatter(
        transcript_xy[:, 0], transcript_xy[:, 1], c=transcript_genes, marker="o", s=20
    )
    mapped_cells, method_name = model_transcript_cells, f"EM (bandwidth={bandwidth})"
    # mapped_cells, method_name = nearest_cell, 'kNN'
    for i in range(n_transcripts):
        mapped_cell = mapped_cells[i]
        true_cell = transcript_cells[i]
        x, y = transcript_xy[i]
        if mapped_cell == 0:
            if true_cell != 0:
                true_dx, true_dy = cell_xy[true_cell - 1] - transcript_xy[i]
                plt.arrow(
                    x, y, true_dx, true_dy, width=0.1, head_width=0.05, color="blue"
                )
            continue
        dx, dy = cell_xy[mapped_cell - 1] - transcript_xy[i]
        if known_mask[i]:
            plt.arrow(x, y, dx, dy, width=0.1, head_width=0.05, color="gray")
        elif mapped_cell == transcript_cells[i]:
            plt.arrow(x, y, dx, dy, width=0.1, head_width=0.05, color="black")
        else:
            plt.arrow(x, y, dx, dy, width=0.1, head_width=0.05, color="red")
            if true_cell != 0:
                true_dx, true_dy = cell_xy[true_cell - 1] - transcript_xy[i]
                plt.arrow(
                    x, y, true_dx, true_dy, width=0.1, head_width=0.05, color="blue"
                )

    plt.title(
        f"{method_name} transcript assignment\nNucleus={nucleus_size} Max cell size={max_distance}"
    )
    plt.show()
