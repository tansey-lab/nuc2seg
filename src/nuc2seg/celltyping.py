import tqdm
import numpy as np
import geopandas
from kneed import KneeLocator

from scipy.special import softmax
from nuc2seg.xenium import logger
from nuc2seg.data import CelltypingResults, RasterizedDataset
from scipy.special import logsumexp
from collections import defaultdict


def aic_bic(gene_counts, expression_profiles, prior_probs):
    n_components = expression_profiles.shape[0]
    n_genes = expression_profiles.shape[1]
    n_samples = gene_counts.shape[0]

    dof = n_components * (n_genes - 1) + n_components - 1
    log_likelihood = logsumexp(
        np.log(prior_probs[None])
        + (gene_counts[:, None] * np.log(expression_profiles[None])).sum(axis=2),
        axis=1,
    ).sum()
    aic = -2 * log_likelihood + 2 * dof
    bic = -2 * log_likelihood + dof * np.log(n_samples)

    return aic, bic


def estimate_cell_types(
    gene_counts,
    gene_names,
    min_components=2,
    max_components=25,
    max_em_steps=100,
    tol=1e-4,
    warm_start=False,
    rng: np.random.Generator = None,
):
    if rng is None:
        rng = np.random.default_rng()

    n_nuclei, n_genes = gene_counts.shape

    # Randomly initialize cell type profiles
    cur_expression_profiles = rng.dirichlet(np.ones(n_genes), size=min_components)

    # Initialize probabilities to be uniform
    cur_prior_probs = np.ones(min_components) / min_components

    # No need to initialize cell type assignments
    cur_cell_types = None

    # Track BIC and AIC scores
    aic_scores = np.zeros(max_components - min_components + 1)
    bic_scores = np.zeros(max_components - min_components + 1)

    # Iterate through every possible number of components
    (
        final_expression_profiles,
        final_prior_probs,
        final_cell_types,
        final_expression_rates,
    ) = ([], [], [], [])
    prev_expression_profiles = np.zeros_like(cur_expression_profiles)
    for idx, n_components in enumerate(
        tqdm.trange(
            min_components, max_components + 1, desc="estimate_cell_types", position=0
        )
    ):

        # Warm start from the previous iteration
        if warm_start and n_components > min_components:
            # Expand and copy over the current parameters
            next_prior_probs = np.zeros(n_components)
            next_expression_profiles = np.zeros((n_components, n_genes))
            next_prior_probs[: n_components - 1] = cur_prior_probs
            next_expression_profiles[: n_components - 1] = cur_expression_profiles

            # Split the dominant cluster
            dominant = np.argmax(cur_prior_probs)
            split_prob = rng.random()
            next_prior_probs[-1] = next_prior_probs[dominant] * split_prob
            next_prior_probs[dominant] = next_prior_probs[dominant] * (1 - split_prob)
            next_expression_profiles[-1] = cur_expression_profiles[
                dominant
            ] * split_prob + (1 - split_prob) * rng.dirichlet(np.ones(n_genes))

            cur_prior_probs = next_prior_probs
            cur_expression_profiles = next_expression_profiles

            logger.debug("priors:", cur_prior_probs)
            logger.debug("expression:", cur_expression_profiles)
        else:
            # Cold start from a random location
            cur_expression_profiles = rng.dirichlet(np.ones(n_genes), size=n_components)
            cur_prior_probs = np.ones(n_components) / n_components

        converge = tol + 1
        for step in tqdm.trange(
            max_em_steps,
            desc=f"EM for n_components {n_components}",
            unit="step",
            position=1,
        ):

            # E-step: estimate cell type assignments (posterior probabilities)
            logits = np.log(cur_prior_probs[None]) + (
                gene_counts[:, None] * np.log(cur_expression_profiles[None])
            ).sum(axis=2)
            cur_cell_types = softmax(logits, axis=1)

            # M-step (part 1): estimate cell type profiles
            prev_expression_profiles = np.array(cur_expression_profiles)
            cur_expression_profiles = (
                cur_cell_types[..., None] * gene_counts[:, None]
            ).sum(axis=0) + 1
            cur_expression_profiles = (
                cur_expression_profiles / (cur_cell_types.sum(axis=0) + 1)[:, None]
            )

            # M-step (part 2): estimate cell type probabilities
            cur_prior_probs = (cur_cell_types.sum(axis=0) + 1) / (
                cur_cell_types.shape[0] + n_components
            )
            logger.debug(f"cur_prior_probs: {cur_prior_probs}")

            # Track convergence of the cell type profiles
            converge = np.linalg.norm(
                prev_expression_profiles - cur_expression_profiles
            ) / np.linalg.norm(prev_expression_profiles)

            logger.debug(f"Convergence: {converge:.4f}")
            if converge <= tol:
                logger.debug(f"Stopping early.")
                break

        # Save the results
        final_expression_profiles.append(cur_expression_profiles)
        final_cell_types.append(cur_cell_types)
        final_prior_probs.append(cur_prior_probs)

        # Calculate BIC and AIC
        aic_scores[idx], bic_scores[idx] = aic_bic(
            gene_counts, cur_expression_profiles, cur_prior_probs
        )

        logger.debug(f"K={n_components}")
        logger.debug(f"AIC: {aic_scores[idx]:.4f}")
        logger.debug(f"BIC: {bic_scores[idx]:.4f}")

    relative_expression = calculate_celltype_relative_expression(
        gene_counts, final_cell_types
    )

    return CelltypingResults(
        aic_scores=aic_scores,
        bic_scores=bic_scores,
        final_expression_profiles=final_expression_profiles,
        final_prior_probs=final_prior_probs,
        final_cell_types=final_cell_types,
        relative_expression=relative_expression,
        min_n_components=min_components,
        max_n_components=max_components,
        gene_names=gene_names,
    )


def calculate_celltype_relative_expression(gene_counts, final_cell_types):
    results = []
    for idx, cell_types in enumerate(final_cell_types):
        assignments = np.argmax(cell_types, axis=1)

        relative_expressions = []
        for cell_type_idx in range(cell_types.shape[1]):

            if np.count_nonzero(assignments == cell_type_idx) == 0:
                relative_expressions.append(np.zeros(gene_counts.shape[1]))
                continue

            in_group_mean = gene_counts[assignments == cell_type_idx].mean(axis=0)
            out_of_group_mean = gene_counts[assignments != cell_type_idx].mean(axis=0)

            out_of_group_mean = out_of_group_mean.clip(min=1e-2)

            relative_expression = in_group_mean / out_of_group_mean

            relative_expressions.append(relative_expression)

        relative_expressions = np.array(relative_expressions)
        results.append(relative_expressions)
    return results


def run_cell_type_estimation(
    nuclei_geo_df: geopandas.GeoDataFrame,
    tx_geo_df: geopandas.GeoDataFrame,
    foreground_nucleus_distance: float = 1,
    min_components: int = 2,
    max_components: int = 25,
    rng: np.random.Generator = None,
):

    # Create a nuclei x gene count matrix
    tx_nuclei_geo_df = geopandas.sjoin_nearest(
        tx_geo_df, nuclei_geo_df, distance_col="nucleus_distance"
    )

    n_genes = tx_nuclei_geo_df["gene_id"].max() + 1

    nuclei_count_geo_df = tx_nuclei_geo_df[
        tx_nuclei_geo_df["nucleus_distance"] <= foreground_nucleus_distance
    ]

    # I think we have enough memory to just store this as a dense array
    nuclei_count_matrix = np.zeros((nuclei_geo_df.shape[0] + 1, n_genes), dtype=int)
    np.add.at(
        nuclei_count_matrix,
        (
            nuclei_count_geo_df["nucleus_label"].values.astype(int),
            nuclei_count_geo_df["gene_id"].values.astype(int),
        ),
        1,
    )

    gene_name_map = dict(zip(tx_geo_df["gene_id"], tx_geo_df["feature_name"]))

    gene_names = np.array([gene_name_map[i] for i in range(n_genes)])

    return estimate_cell_types(
        nuclei_count_matrix,
        gene_names=gene_names,
        min_components=min_components,
        max_components=max_components,
        rng=rng,
    )


def select_best_celltyping_chain(results: list[CelltypingResults]):
    aic_scores = []
    bic_scores = []
    gene_names = None

    for result in results:
        if gene_names is None:
            gene_names = result.gene_names
        else:
            if not np.all(gene_names == result.gene_names):
                raise ValueError("Gene names do not match between results.")
        aic_scores.append(result.aic_scores)
        bic_scores.append(result.bic_scores)

    aic_scores = np.stack(aic_scores)
    bic_scores = np.stack(bic_scores)

    best_chain, best_k = np.where(bic_scores == bic_scores.min())
    best_chain = best_chain.item()
    best_k = best_k.item()

    logger.info(f"Best chain: {best_chain}, best k: {best_k}")

    best_result = results[best_chain]

    return (best_result, np.stack(aic_scores), np.stack(bic_scores), best_k)


def get_best_k(aic_scores, bic_scores):
    best_k_aic = np.argmin(aic_scores)
    best_k_bic = np.argmin(bic_scores)

    if best_k_bic == best_k_aic:
        return best_k_aic
    else:
        logger.warning(
            f"The best k according to AIC and BIC do not match ({best_k_aic} vs {best_k_bic}). Using BIC eblow to determine k"
        )
        kneedle = KneeLocator(
            x=np.arange(len(bic_scores)),
            y=bic_scores,
            S=2,
            curve="convex",
            direction="decreasing",
        )
        best_k = kneedle.elbow

        logger.info(f"BIC elbow to chose k: {best_k}")

        return best_k
