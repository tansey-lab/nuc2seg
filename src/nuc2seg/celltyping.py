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
    prior_probs,
    expression_profiles,
    gene_counts,
):
    """
    Estimate the cell types probabilities for each cell.

    :param prior_probs: The prior probabilities of each cell type, shape (n_cell_types,)
    :param expression_profiles: The expression profiles of each cell type, shape (n_cell_types, n_genes)
    :param gene_counts: The gene counts for each cell, shape (n_cells, n_genes)
    :return: Array of probabilities for each cell type, shape(n_cell, n_cell_types)
    """
    logits = np.log(prior_probs[None]) + (
        gene_counts[:, None] * np.log(expression_profiles[None])
    ).sum(axis=2)
    return softmax(logits, axis=1)


def fit_celltype_em_model(
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
        cur_cell_types = None
        for step in tqdm.trange(
            max_em_steps,
            desc=f"EM for n_components {n_components}",
            unit="step",
            position=1,
        ):

            # E-step: estimate cell type assignments (posterior probabilities)
            cur_cell_types = estimate_cell_types(
                prior_probs=cur_prior_probs,
                expression_profiles=cur_expression_profiles,
                gene_counts=gene_counts,
            )

            # M-step (part 1): estimate cell type profiles
            prev_expression_profiles = np.array(cur_expression_profiles)
            cur_expression_profiles = (
                cur_cell_types[..., None] * gene_counts[:, None]
            ).sum(axis=0) + 1
            cur_expression_profiles = (
                cur_expression_profiles
                / cur_expression_profiles.sum(axis=1, keepdims=True)
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
        final_prior_probs.append(cur_prior_probs)
        if cur_cell_types is not None:
            final_cell_types.append(cur_cell_types)

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
        expression_profiles=final_expression_profiles,
        prior_probs=final_prior_probs,
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


def create_dense_gene_counts_matrix(
    segmentation_geo_df: geopandas.GeoDataFrame,
    transcript_geo_df: geopandas.GeoDataFrame,
    max_distance: float = 0,
    gene_id_col: str = "gene_id",
):
    """
    Create a dense gene counts matrix from segment polygons and transcript points

    :param segmentation_geo_df: GeoDataFrame where the geometry column in polygons
    :param transcript_geo_df: GeoDataFrame where the geometry column is points, should have column
    gene_id_col, which is integer gene ids from 0 to n_genes
    :param max_distance: Maximum distance to consider a transcript to be associated with a segment,
    default 0 means will only include transcripts that are inside the segment boundary
    :returns: A dense gene counts matrix, (n_segments, n_genes)
    """
    segmentation_geo_df = segmentation_geo_df.reset_index(names="segment_id")

    # Create a nuclei x gene count matrix
    joined_df = geopandas.sjoin_nearest(
        transcript_geo_df, segmentation_geo_df, distance_col="_sjoin_distance"
    )

    if "transcript_id" in joined_df.columns:
        del joined_df["transcript_id"]

    joined_df = joined_df.reset_index(drop=False, names="transcript_id")

    # dedupe ties where transcript is equidistant to multiple nuclei
    joined_df = joined_df.drop_duplicates(subset=["transcript_id"]).reset_index(
        drop=True
    )

    n_genes = transcript_geo_df[gene_id_col].nunique()

    nuclei_count_geo_df = joined_df[
        joined_df["_sjoin_distance"] <= max_distance
    ].reset_index(drop=True)

    del joined_df["_sjoin_distance"]

    nuclei_count_matrix = np.zeros((len(segmentation_geo_df), n_genes), dtype=int)
    np.add.at(
        nuclei_count_matrix,
        (
            nuclei_count_geo_df["segment_id"].values.astype(int),
            nuclei_count_geo_df[gene_id_col].values.astype(int),
        ),
        1,
    )

    return nuclei_count_matrix


def fit_celltyping_on_segments_and_transcripts(
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

    n_genes = tx_geo_df["gene_id"].nunique()

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

    return fit_celltype_em_model(
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


def predict_celltypes_for_segments_and_transcripts(
    expression_profiles,
    prior_probs,
    segment_geo_df: geopandas.GeoDataFrame,
    transcript_geo_df: geopandas.GeoDataFrame,
    chunk_size: int = 10_000,
    gene_name_column: str = "feature_name",
    max_distinace: float = 0,
    gene_names: list[str] = None,
):
    if gene_names is not None:
        # gene_name to id map
        gene_name_to_id = dict(
            zip(
                gene_names,
                range(len(gene_names)),
            )
        )

        selection_vector = transcript_geo_df[gene_name_column].isin(gene_names)
        transcript_geo_df = transcript_geo_df[selection_vector]
    else:
        gene_name_to_id = dict(
            zip(
                sorted(transcript_geo_df[gene_name_column].unique()),
                range(transcript_geo_df[gene_name_column].nunique()),
            )
        )

    transcript_geo_df["gene_id"] = transcript_geo_df[gene_name_column].map(
        gene_name_to_id
    )

    results = []

    # iterate segment_geo_df in chunks of chunk_size
    current_index = 0
    while current_index < len(segment_geo_df):
        segment_chunk = segment_geo_df.iloc[
            current_index : current_index + chunk_size
        ].reset_index(drop=True)
        current_index += chunk_size

        gene_counts = create_dense_gene_counts_matrix(
            segment_chunk,
            transcript_geo_df,
            max_distance=max_distinace,
            gene_id_col="gene_id",
        )

        results.append(
            estimate_cell_types(
                expression_profiles=expression_profiles,
                prior_probs=prior_probs,
                gene_counts=gene_counts,
            )
        )

    return np.concatenate(results)
