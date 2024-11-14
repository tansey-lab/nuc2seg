import numpy as np
import tqdm
from scipy.spatial import KDTree


def gene_uniformity_test(genes, distances, cell_locations, n_trials=100, exact=False):
    gene_names, gene_idxs = np.unique(genes, return_inverse=True)
    n_genes = len(gene_names)
    n_transcripts = len(genes)

    # Boundaries
    xy_min, xy_max = cell_locations.min(axis=0), cell_locations.max(axis=0)
    xy_range = xy_max - xy_min

    # Get the true test statistics
    test_stats = np.zeros(n_genes)
    np.add.at(test_stats, gene_idxs, distances)

    # Fast lookup for nearest cell
    kdtree = KDTree(cell_locations)

    null_stats = np.zeros((n_trials, n_genes))
    for trial in tqdm.trange(n_trials):
        print(f"Trial {trial+1}/{n_trials}")

        # Sample uniformly over the region
        null_locations = (
            np.random.random(size=(n_transcripts, 2)) * xy_range[None] + xy_min[None]
        )

        # Compute nearest cells
        null_distances = kdtree.query(null_locations, k=1)[0]

        # Calculate the test statistics
        trial_idxs = np.array([trial] * len(genes))
        np.add.at(null_stats, (trial_idxs, gene_idxs), null_distances)

    if exact:
        # How often was each null gene closer than the true gene
        p_values = (test_stats[None] >= null_stats).mean(axis=0)
    else:
        # Normal approximation (asymptotically correct based on CLT)
        from scipy.stats import norm

        p_values = norm.cdf(test_stats, null_stats.mean(axis=0), null_stats.std(axis=0))

    return gene_names, p_values, test_stats, null_stats
