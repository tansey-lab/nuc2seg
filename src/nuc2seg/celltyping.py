import anndata
import geopandas
import numpy as np
import tqdm
from kneed import KneeLocator
from scipy.special import logsumexp
from scipy.special import softmax
from scipy.sparse import issparse
from nuc2seg.data import CelltypingResults
from nuc2seg.xenium import logger


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
    max_components=20,
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
    n_genes: int,
    gene_id_col: str = "gene_id",
):
    """
    Create a dense gene counts matrix from segment polygons and transcript points

    :param segmentation_geo_df: GeoDataFrame where the geometry column in polygons
    :param transcript_geo_df: GeoDataFrame where the geometry column is points, should have column
    gene_id_col, which is integer gene ids from 0 to n_genes
    :param n_genes: Number of genes in the dataset total
    :param max_distance: Maximum distance to consider a transcript to be associated with a segment,
    default 0 means will only include transcripts that are inside the segment boundary
    :returns: A dense gene counts matrix, (n_segments, n_genes)
    """
    segmentation_geo_df = segmentation_geo_df.reset_index(names="segment_id")

    # Create a nuclei x gene count matrix
    nuclei_count_geo_df = geopandas.sjoin(transcript_geo_df, segmentation_geo_df)

    if "transcript_id" in nuclei_count_geo_df.columns:
        del nuclei_count_geo_df["transcript_id"]

    nuclei_count_geo_df = nuclei_count_geo_df.reset_index(
        drop=False, names="transcript_id"
    )

    # dedupe ties where transcript is equidistant to multiple nuclei
    nuclei_count_geo_df = nuclei_count_geo_df.drop_duplicates(
        subset=["transcript_id"]
    ).reset_index(drop=True)

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


def fit_celltyping_on_adata(
    adata: anndata.AnnData,
    min_components: int = 2,
    max_components: int = 20,
    rng: np.random.Generator = None,
):
    adata = adata[:, sorted(adata.var_names)]

    if issparse(adata.X):
        counts_matrix = np.array(adata.X.todense())
    else:
        counts_matrix = np.array(adata.X)

    return fit_celltype_em_model(
        counts_matrix,
        gene_names=adata.var_names.tolist(),
        min_components=min_components,
        max_components=max_components,
        rng=rng,
    )


def select_best_celltyping_chain(results: list[CelltypingResults], best_k=None):
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

    if best_k is None:
        best_chain, best_k = np.where(bic_scores == bic_scores.min())
        best_chain = best_chain.item()
        best_k = best_k.item()
    else:
        best_chain = bic_scores[:, best_k].argmin()

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
    """
    :returns: Array of probabilities for each cell type, shape(n_cell, n_cell_types)
    """
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

    pbar = tqdm.tqdm(total=len(segment_geo_df), desc="predict_celltypes")

    segment_geo_df["_x_centroid"] = segment_geo_df.centroid.x
    segment_geo_df["_y_centroid"] = segment_geo_df.centroid.y
    segment_geo_df.sort_values(by=["_x_centroid", "_y_centroid"], inplace=True)
    n_genes = transcript_geo_df["gene_id"].max() + 1
    # iterate segment_geo_df in chunks of chunk_size
    current_index = 0
    while current_index < len(segment_geo_df):
        segment_chunk = segment_geo_df.iloc[
            current_index : current_index + chunk_size
        ].reset_index(drop=True)
        xmin, ymin, xmax, ymax = segment_chunk.total_bounds
        transcript_chunk = transcript_geo_df.cx[xmin:xmax, ymin:ymax].reset_index(
            drop=True
        )

        pbar.update(len(segment_chunk))
        current_index += chunk_size

        gene_counts = create_dense_gene_counts_matrix(
            segment_chunk,
            transcript_chunk,
            n_genes,
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


def predict_celltypes_for_anndata(
    expression_profiles,
    prior_probs,
    ad: anndata.AnnData,
    gene_names: list[str] = None,
    chunk_size: int = 10_000,
):
    if len(ad) == 0:
        return None

    results = []
    ad = ad[:, ad.var_names.isin(gene_names)]
    current_index = 0
    pbar = tqdm.tqdm(total=len(ad), desc="predict_celltypes")
    while current_index < len(ad):
        ad_chunk = ad[current_index : current_index + chunk_size]
        gene_counts = ad_chunk.to_df()

        for g in gene_names:
            if g not in gene_counts.columns:
                gene_counts[g] = 0
        gene_counts = gene_counts[gene_names].values

        results.append(
            estimate_cell_types(
                expression_profiles=expression_profiles,
                prior_probs=prior_probs,
                gene_counts=gene_counts,
            )
        )
        current_index += chunk_size
        pbar.update(len(ad_chunk))

    return np.concatenate(results)


def predict_celltypes_for_anndata_with_noise_type(
    expression_profiles,
    prior_probs,
    ad: anndata.AnnData,
    chunk_size: int = 10_000,
    min_transcripts: int = 10,
):
    if len(ad) == 0:
        return None

    proportion_noise = (np.array(ad.X.sum(axis=1)).squeeze() < min_transcripts).mean()

    adjusted_prior_probs = np.concatenate(
        [
            np.array([proportion_noise / (prior_probs.sum() + proportion_noise)]),
            prior_probs / (prior_probs.sum() + proportion_noise),
        ]
    )
    expression_profiles_with_noise_profile = np.concatenate(
        [
            np.ones((1, expression_profiles.shape[1])) / len(ad.var_names),
            expression_profiles,
        ]
    )

    results = []
    current_index = 0
    pbar = tqdm.tqdm(total=len(ad), desc="predict_celltypes")
    while current_index < len(ad):
        ad_chunk = ad[current_index : current_index + chunk_size]

        if issparse(ad_chunk.X):
            gene_counts = np.array(ad_chunk.X.todense())
        else:
            gene_counts = np.array(ad_chunk.X)

        results.append(
            estimate_cell_types(
                expression_profiles=expression_profiles_with_noise_profile,
                prior_probs=adjusted_prior_probs,
                gene_counts=gene_counts,
            )
        )
        current_index += chunk_size
        pbar.update(len(ad_chunk))

    return np.concatenate(results)


def predict_celltype_probabilities_for_all_segments(
    labels,
    transcripts,
    expression_profiles,
    prior_probs,
):
    n_unique_cells = len(np.unique(labels[labels > 0]))

    counts = np.zeros((n_unique_cells, expression_profiles.shape[1]))

    label_per_transcript = labels[transcripts[:, 0], transcripts[:, 1]]

    selection_vector = label_per_transcript > 0
    np.add.at(
        counts,
        (
            label_per_transcript[selection_vector] - 1,
            transcripts[:, 2][selection_vector],
        ),
        1,
    )

    return estimate_cell_types(prior_probs, expression_profiles, counts)
