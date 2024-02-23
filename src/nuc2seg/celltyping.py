import numpy as np
import tqdm
from scipy.special import softmax

from nuc2seg.xenium import logger
import numpy as np
from scipy.special import logsumexp


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


## TODO: Maybe replace this with a library that does the same thing
def estimate_cell_types(
    gene_counts,
    min_components=2,
    max_components=25,
    max_em_steps=100,
    tol=1e-4,
    warm_start=False,
):
    n_nuclei, n_genes = gene_counts.shape

    # Randomly initialize cell type profiles
    cur_expression_profiles = np.random.dirichlet(np.ones(n_genes), size=min_components)

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
            split_prob = np.random.random()
            next_prior_probs[-1] = next_prior_probs[dominant] * split_prob
            next_prior_probs[dominant] = next_prior_probs[dominant] * (1 - split_prob)
            next_expression_profiles[-1] = cur_expression_profiles[
                dominant
            ] * split_prob + (1 - split_prob) * np.random.dirichlet(np.ones(n_genes))

            cur_prior_probs = next_prior_probs
            cur_expression_profiles = next_expression_profiles

            logger.debug("priors:", cur_prior_probs)
            logger.debug("expression:", cur_expression_profiles)
        else:
            # Cold start from a random location
            cur_expression_profiles = np.random.dirichlet(
                np.ones(n_genes), size=n_components
            )
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

    return (
        aic_scores,
        bic_scores,
        final_expression_profiles,
        final_prior_probs,
        final_cell_types,
        relative_expression,
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
