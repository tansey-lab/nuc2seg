import numpy as np
from scipy.special import softmax


def sqexp(x1, x2, bandwidth=2, scale=1, axis=None):
    return scale * np.exp(-np.linalg.norm(x1 - x2, axis=axis) ** 2 / bandwidth**2)


def sample_gp_fast(width, height, step=10, size=1, bandwidth=10, scale=1):
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


def grid_graph_edges(rows, cols):
    from collections import defaultdict

    edges = defaultdict(list)
    for x in range(cols):
        for y in range(rows):
            if x < cols - 1:
                i = int(y * cols + x)
                j = int(y * cols + x + 1)
                edges[i].append(j)
                edges[j].append(i)
            if y < rows - 1:
                i = int(y * cols + x)
                j = int((y + 1) * cols + x)
                edges[i].append(j)
                edges[j].append(i)
    return edges


def calc_plateaus(beta, edges, rel_tol=1e-4, verbose=0):
    """Calculate the plateaus (degrees of freedom) of a graph of beta values in linear time."""
    from collections import deque

    if not isinstance(edges, dict):
        raise Exception("Edges must be a map from each node to a list of neighbors.")
    to_check = deque(range(len(beta)))
    check_map = np.zeros(beta.shape, dtype=bool)
    check_map[np.isnan(beta)] = True
    plateaus = []

    if verbose:
        print("\tCalculating plateaus...")

    if verbose > 1:
        print("\tIndices to check {0} {1}".format(len(to_check), check_map.shape))

    # Loop until every beta index has been checked
    while to_check:
        if verbose > 1:
            print("\t\tPlateau #{0}".format(len(plateaus) + 1))

        # Get the next unchecked point on the grid
        idx = to_check.popleft()

        # If we already have checked this one, just pop it off
        while to_check and check_map[idx]:
            try:
                idx = to_check.popleft()
            except:
                break

        # Edge case -- If we went through all the indices without reaching an unchecked one.
        if check_map[idx]:
            break

        # Create the plateau and calculate the inclusion conditions
        cur_plateau = set([idx])
        cur_unchecked = deque([idx])
        val = beta[idx]
        min_member = val - rel_tol
        max_member = val + rel_tol

        # Check every possible boundary of the plateau
        while cur_unchecked:
            idx = cur_unchecked.popleft()

            # neighbors to check
            local_check = []

            # Generic graph case, get all neighbors of this node
            local_check.extend(edges[idx])

            # Check the index's unchecked neighbors
            for local_idx in local_check:
                if (
                    not check_map[local_idx]
                    and min_member <= beta[local_idx] <= max_member
                ):
                    # Label this index as being checked so it's not re-checked unnecessarily
                    check_map[local_idx] = True

                    # Add it to the plateau and the list of local unchecked locations
                    cur_unchecked.append(local_idx)
                    cur_plateau.add(local_idx)

        # Track each plateau's indices
        plateaus.append((val, cur_plateau))

    # Returns the list of plateaus and their values
    return plateaus


def simulate_gaussian_cells(
    img_width=100,
    img_height=100,
    n_cell_types=6,
    n_genes=20,
    n_transcripts=1000,
    nucleus_size=2,
    n_cells_approx=200,
    background_rate=0.05,
):
    # Relative mRNA per cell type
    truth_amplitudes = np.random.beta(3, 3, size=n_cell_types)

    # Simulate some gene profiles in a way that doesn't create much overlap
    # between marker genes for each cell type.
    # To do this, we normalize across cell types first for each gene
    gene_probs = np.random.normal(size=(n_cell_types, n_genes))
    gene_probs = softmax(gene_probs, axis=0)
    gene_probs = gene_probs / gene_probs.sum(axis=1, keepdims=True)
    gene_probs = np.random.dirichlet(n_genes * gene_probs)

    # Generate the nuclei locations
    cell_xy = []
    cell_types = []
    cell_covs = []
    for k in range(n_cell_types):
        # Sample cell locations independently from a poisson process
        locs = sample_poisson_points_fast(
            np.random.binomial(n_cells_approx, 0.5), img_width, img_height
        )
        cell_xy.extend(locs)
        cell_types.extend([k + 1] * len(locs))

        # Generate the cell shape -- just use Gaussian cell shapes
        cor = (np.random.random(size=len(locs)) - 0.5) * 0.7
        sigma = np.random.random(size=(len(locs), 2)) * 5 + 1
        cell_covs.extend([np.array([[s[0], c], [c, s[1]]]) for s, c in zip(sigma, cor)])
    cell_xy = np.array(cell_xy)
    cell_types = np.array(cell_types, dtype=int)
    cell_covs = np.array(cell_covs)
    n_cells = cell_types.shape[0]

    # Sample transcripts and assign ones within radius nucleus_size of the center to the cell
    z_probs = truth_amplitudes[cell_types]
    z_probs /= z_probs.sum()
    z_probs = np.hstack([background_rate, [(1 - background_rate) * z_probs]])
    n_transcripts_per_cell = np.random.multinomial(n_transcripts, z_probs)
    transcript_genes = np.zeros(n_transcripts, dtype=int)
    transcript_cells = np.zeros(n_transcripts, dtype=int)
    transcript_xy = np.zeros((n_transcripts, 2))
    offsets = np.cumsum(n_transcripts_per_cell)
    for j in range(n_cells + 1):
        if j == 0:
            # Background noise transcripts are uniform random
            transcript_xy[: offsets[j]] = (
                np.random.random(size=(n_transcripts_per_cell[0], 2))
                * np.array([img_width, img_height])[None]
            )
            transcript_genes[: offsets[j]] = np.random.choice(
                n_genes, size=n_transcripts_per_cell[0]
            )
        else:
            transcript_xy[offsets[j - 1] : offsets[j]] = np.random.multivariate_normal(
                cell_xy[j - 1], cell_covs[j - 1], size=n_transcripts_per_cell[j]
            )
            transcript_genes[offsets[j - 1] : offsets[j]] = np.random.choice(
                n_genes,
                p=gene_probs[cell_types[j - 1] - 1],
                size=n_transcripts_per_cell[j],
            )
            transcript_cells[offsets[j - 1] : offsets[j]] = j

            # Remove transcripts within the definite-yes range of the nucleus
            for i in range(offsets[j - 1], offsets[j]):
                distances = np.sqrt(
                    ((transcript_xy[i : i + 1] - cell_xy) ** 2).sum(axis=1)
                )
                nearest = np.argmin(distances)
                while nearest != (j - 1) and distances[nearest] < nucleus_size:
                    # Resample the x y location of this transcript
                    transcript_xy[i] = np.random.multivariate_normal(
                        cell_xy[j - 1], cell_covs[j - 1]
                    )
                    distances = np.sqrt(
                        ((transcript_xy[i : i + 1] - cell_xy) ** 2).sum(axis=1)
                    )
                    nearest = np.argmin(distances)
