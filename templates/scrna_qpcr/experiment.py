from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc

from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_predict


# ================================================================
# CONFIGURATION
# ================================================================

RNG_SEED = 42

DISCOVERY_FRACTION = 0.70

MARKER_CANDIDATES = 30
MAX_MARKERS_PER_CLUSTER = 5

REFERENCE_POOL_SIZE = 200
N_REFERENCE_GENES = 3

N_TRAIN_PSEUDOBULK = 400
N_TEST_PSEUDOBULK = 300

CELLS_PER_PSEUDOBULK = 100

DETECTION_THRESHOLD = 0.35

LOW_FRACTION_THRESHOLD = 0.10
HIGH_FRACTION_THRESHOLD = 0.50

MIN_MAE_GAIN = 0.002

N_CV_SPLITS = 5


# ================================================================
# UTILITIES
# ================================================================


def _to_dense_vector(x):
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel()

    return np.asarray(x).ravel()


def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)

    if np.unique(y_true).size < 2:
        return 0.5

    return float(
        roc_auc_score(
            y_true,
            y_score,
        )
    )


def _safe_spearman(y_true, y_pred):
    result = spearmanr(
        y_true,
        y_pred,
    )

    value = result.statistic

    if np.isnan(value):
        return 0.0

    return float(value)


def _safe_pearson(y_true, y_pred):
    if np.std(y_true) == 0:
        return 0.0

    if np.std(y_pred) == 0:
        return 0.0

    result = pearsonr(
        y_true,
        y_pred,
    )

    value = result.statistic

    if np.isnan(value):
        return 0.0

    return float(value)


# ================================================================
# SPLIT CELLS
# ================================================================


def _split_cells_by_cluster(
    cluster_series,
    rng,
):
    discovery_indices = []
    holdout_indices = []

    cluster_values = (
        cluster_series.values
    )

    cluster_names = sorted(
        cluster_series.unique().tolist()
    )

    for cluster in cluster_names:

        idx = np.where(
            cluster_values == cluster
        )[0]

        idx = idx.copy()

        rng.shuffle(idx)

        if len(idx) < 4:
            raise ValueError(
                f"Cluster {cluster} has only {len(idx)} cells. "
                "At least 4 cells are required."
            )

        n_discovery = int(
            round(
                len(idx)
                * DISCOVERY_FRACTION
            )
        )

        n_discovery = max(
            2,
            min(
                n_discovery,
                len(idx) - 2,
            ),
        )

        discovery_indices.extend(
            idx[:n_discovery]
        )

        holdout_indices.extend(
            idx[n_discovery:]
        )

    return (
        np.asarray(
            discovery_indices,
            dtype=int,
        ),
        np.asarray(
            holdout_indices,
            dtype=int,
        ),
    )


# ================================================================
# REFERENCE GENE SELECTION
# ================================================================


def _select_stable_reference_genes(
    adata_discovery,
    cluster_names,
    candidate_marker_genes,
):
    """
    Select genes with relatively stable expression across clusters.

    These genes are used for normalization only.

    NOTE:
    This is a qPCR-style normalization proxy because the H5AD matrix
    contains expression values, not physical Ct measurements.
    """

    x = adata_discovery.X

    overall_mean = _to_dense_vector(
        x.mean(axis=0)
    )

    if sparse.issparse(x):

        expressed_fraction = (
            np.asarray(
                (x > 0).mean(axis=0)
            ).ravel()
        )

    else:

        expressed_fraction = (
            np.asarray(x) > 0
        ).mean(axis=0)

    cluster_means = []

    labels = (
        adata_discovery.obs[
            "leiden"
        ]
        .astype(str)
        .values
    )

    for cluster in cluster_names:

        idx = np.where(
            labels == cluster
        )[0]

        cluster_mean = _to_dense_vector(
            x[idx].mean(axis=0)
        )

        cluster_means.append(
            cluster_mean
        )

    cluster_means = np.asarray(
        cluster_means
    )

    between_cluster_sd = np.std(
        cluster_means,
        axis=0,
    )

    stability_score = (
        between_cluster_sd
        / (
            np.abs(overall_mean)
            + 1e-6
        )
    )

    order = np.argsort(
        stability_score
    )

    candidate_marker_set = set(
        candidate_marker_genes
    )

    stable_pool = []

    for idx in order:

        gene = str(
            adata_discovery.var_names[
                idx
            ]
        )

        if (
            expressed_fraction[idx]
            < 0.20
        ):
            continue

        if gene in candidate_marker_set:
            continue

        stable_pool.append(
            gene
        )

        if (
            len(stable_pool)
            >= REFERENCE_POOL_SIZE
        ):
            break

    if (
        len(stable_pool)
        < N_REFERENCE_GENES
    ):
        raise ValueError(
            "Unable to identify enough stable reference genes."
        )

    return stable_pool


# ================================================================
# TARGETED PSEUDOBULK SIMULATION
# ================================================================


def _sample_target_fraction(
    rng,
):
    """
    Explicitly cover low, medium and high abundance regimes.
    """

    regime = rng.choice(
        [
            "low",
            "medium",
            "high",
        ],
        p=[
            0.30,
            0.45,
            0.25,
        ],
    )

    if regime == "low":
        return float(
            rng.uniform(
                0.01,
                0.10,
            )
        )

    if regime == "medium":
        return float(
            rng.uniform(
                0.10,
                0.50,
            )
        )

    return float(
        rng.uniform(
            0.50,
            0.90,
        )
    )


def _simulate_targeted_pseudobulk(
    x_sub,
    cluster_names,
    cluster_to_indices,
    target_cluster,
    rng,
    n_samples,
    cells_per_sample,
):
    """
    Generate pseudobulk RNA mixtures for one target cluster.

    The target-cell proportion is explicitly sampled over a broad
    range. Remaining cells are distributed among all other clusters.
    """

    n_genes = x_sub.shape[1]

    expression = np.zeros(
        (
            n_samples,
            n_genes,
        ),
        dtype=float,
    )

    true_fraction = np.zeros(
        n_samples,
        dtype=float,
    )

    other_clusters = [
        c
        for c in cluster_names
        if c != target_cluster
    ]

    for sample_idx in range(
        n_samples
    ):

        target_fraction = (
            _sample_target_fraction(
                rng
            )
        )

        target_cells = int(
            round(
                target_fraction
                * cells_per_sample
            )
        )

        target_cells = max(
            1,
            min(
                target_cells,
                cells_per_sample - 1,
            ),
        )

        remaining_cells = (
            cells_per_sample
            - target_cells
        )

        remaining_mix = (
            rng.dirichlet(
                np.ones(
                    len(
                        other_clusters
                    )
                )
            )
        )

        other_counts = (
            rng.multinomial(
                remaining_cells,
                remaining_mix,
            )
        )

        chosen_rows = []

        target_source = (
            cluster_to_indices[
                target_cluster
            ]
        )

        target_picks = rng.choice(
            target_source,
            size=target_cells,
            replace=True,
        )

        chosen_rows.append(
            target_picks
        )

        for cluster, count in zip(
            other_clusters,
            other_counts,
        ):

            if count == 0:
                continue

            source = (
                cluster_to_indices[
                    cluster
                ]
            )

            picks = rng.choice(
                source,
                size=count,
                replace=True,
            )

            chosen_rows.append(
                picks
            )

        rows = np.concatenate(
            chosen_rows
        )

        bulk = x_sub[
            rows
        ].mean(
            axis=0
        )

        expression[
            sample_idx
        ] = _to_dense_vector(
            bulk
        )

        true_fraction[
            sample_idx
        ] = (
            target_cells
            / float(
                cells_per_sample
            )
        )

    return (
        expression,
        true_fraction,
    )


# ================================================================
# PANEL SCORING
# ================================================================


def _normalized_panel_score(
    expression,
    marker_genes,
    reference_genes,
    gene_to_idx,
):
    """
    qPCR-style normalized expression score.

    Because the input is expression rather than Ct values, we use:

        mean(marker expression) - mean(reference expression)

    as a Delta-expression proxy.
    """

    marker_idx = [
        gene_to_idx[g]
        for g in marker_genes
        if g in gene_to_idx
    ]

    reference_idx = [
        gene_to_idx[g]
        for g in reference_genes
        if g in gene_to_idx
    ]

    if not marker_idx:
        raise ValueError(
            "Marker panel is empty."
        )

    if not reference_idx:
        raise ValueError(
            "Reference panel is empty."
        )

    marker_signal = (
        expression[
            :,
            marker_idx
        ]
        .mean(axis=1)
    )

    reference_signal = (
        expression[
            :,
            reference_idx
        ]
        .mean(axis=1)
    )

    return (
        marker_signal
        - reference_signal
    )


# ================================================================
# CROSS-VALIDATED PANEL SELECTION
# ================================================================


def _cross_validated_mae(
    expression,
    fractions,
    marker_genes,
    reference_genes,
    gene_to_idx,
):
    score = (
        _normalized_panel_score(
            expression,
            marker_genes,
            reference_genes,
            gene_to_idx,
        )
    )

    x_feature = score.reshape(
        -1,
        1,
    )

    cv = KFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=RNG_SEED,
    )

    predictions = (
        cross_val_predict(
            LinearRegression(),
            x_feature,
            fractions,
            cv=cv,
        )
    )

    predictions = np.clip(
        predictions,
        0.0,
        1.0,
    )

    return float(
        mean_absolute_error(
            fractions,
            predictions,
        )
    )


def _select_marker_panel(
    train_expression,
    train_fraction,
    candidates,
    reference_genes,
    gene_to_idx,
):
    """
    Greedy panel construction using only training pseudobulk samples.

    Criterion:
        cross-validated MAE of cell-fraction prediction
    """

    valid_candidates = [
        gene
        for gene in candidates
        if gene in gene_to_idx
    ]

    if not valid_candidates:
        raise ValueError(
            "No valid marker candidates."
        )

    # ------------------------------------------------------------
    # Find best single marker first
    # ------------------------------------------------------------

    best_single_gene = None
    best_single_mae = np.inf

    for gene in valid_candidates:

        mae = (
            _cross_validated_mae(
                train_expression,
                train_fraction,
                [gene],
                reference_genes,
                gene_to_idx,
            )
        )

        if mae < best_single_mae:

            best_single_mae = mae
            best_single_gene = gene

    chosen = [
        best_single_gene
    ]

    current_mae = (
        best_single_mae
    )

    remaining = [
        gene
        for gene in valid_candidates
        if gene != best_single_gene
    ]

    # ------------------------------------------------------------
    # Greedily add markers only if CV MAE improves
    # ------------------------------------------------------------

    while (
        len(chosen)
        < MAX_MARKERS_PER_CLUSTER
        and remaining
    ):

        best_gene = None
        best_mae = current_mae

        for gene in remaining:

            trial_panel = (
                chosen
                + [gene]
            )

            trial_mae = (
                _cross_validated_mae(
                    train_expression,
                    train_fraction,
                    trial_panel,
                    reference_genes,
                    gene_to_idx,
                )
            )

            if trial_mae < best_mae:

                best_mae = (
                    trial_mae
                )

                best_gene = gene

        if best_gene is None:
            break

        improvement = (
            current_mae
            - best_mae
        )

        if (
            improvement
            < MIN_MAE_GAIN
        ):
            break

        chosen.append(
            best_gene
        )

        remaining.remove(
            best_gene
        )

        current_mae = (
            best_mae
        )

    return (
        chosen,
        best_single_gene,
        best_single_mae,
        current_mae,
    )


# ================================================================
# FIT + EVALUATE
# ================================================================


def _fit_fraction_model(
    train_expression,
    train_fraction,
    test_expression,
    marker_genes,
    reference_genes,
    gene_to_idx,
):
    train_score = (
        _normalized_panel_score(
            train_expression,
            marker_genes,
            reference_genes,
            gene_to_idx,
        )
    )

    test_score = (
        _normalized_panel_score(
            test_expression,
            marker_genes,
            reference_genes,
            gene_to_idx,
        )
    )

    model = LinearRegression()

    model.fit(
        train_score.reshape(
            -1,
            1,
        ),
        train_fraction,
    )

    predictions = model.predict(
        test_score.reshape(
            -1,
            1,
        )
    )

    predictions = np.clip(
        predictions,
        0.0,
        1.0,
    )

    return (
        predictions,
        test_score,
    )


def _calculate_metrics(
    true_fraction,
    predicted_fraction,
):
    true_fraction = np.asarray(
        true_fraction
    )

    predicted_fraction = np.asarray(
        predicted_fraction
    )

    labels = (
        true_fraction
        >= DETECTION_THRESHOLD
    ).astype(int)

    auc = _safe_auc(
        labels,
        predicted_fraction,
    )

    mae = float(
        mean_absolute_error(
            true_fraction,
            predicted_fraction,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                true_fraction,
                predicted_fraction,
            )
        )
    )

    r2 = float(
        r2_score(
            true_fraction,
            predicted_fraction,
        )
    )

    spearman = (
        _safe_spearman(
            true_fraction,
            predicted_fraction,
        )
    )

    pearson = (
        _safe_pearson(
            true_fraction,
            predicted_fraction,
        )
    )

    low_mask = (
        true_fraction
        <= LOW_FRACTION_THRESHOLD
    )

    high_mask = (
        true_fraction
        >= HIGH_FRACTION_THRESHOLD
    )

    if np.any(
        low_mask
    ):

        low_mae = float(
            mean_absolute_error(
                true_fraction[
                    low_mask
                ],
                predicted_fraction[
                    low_mask
                ],
            )
        )

    else:
        low_mae = np.nan

    if np.any(
        high_mask
    ):

        high_mae = float(
            mean_absolute_error(
                true_fraction[
                    high_mask
                ],
                predicted_fraction[
                    high_mask
                ],
            )
        )

    else:
        high_mae = np.nan

    return {
        "auc": auc,
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "low_fraction_mae": low_mae,
        "high_fraction_mae": high_mae,
    }


# ================================================================
# MAIN
# ================================================================


def main(
    out_dir,
):
    rng = np.random.default_rng(
        RNG_SEED
    )

    base_dir = Path(
        __file__
    ).resolve().parent

    data_path = (
        base_dir
        / "input_single_cell_pbmc.h5ad"
    )

    run_dir = (
        base_dir
        / out_dir
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # LOAD H5AD
    # ------------------------------------------------------------

    print(
        f"Loading dataset: {data_path}"
    )

    adata = sc.read_h5ad(
        data_path
    )

    print(
        f"Cells: {adata.n_obs}"
    )

    print(
        f"Genes: {adata.n_vars}"
    )

    if "leiden" not in adata.obs:

        raise ValueError(
            "Expected adata.obs['leiden'] for cluster labels."
        )

    adata.obs[
        "leiden"
    ] = (
        adata.obs[
            "leiden"
        ]
        .astype(str)
        .values
    )

    cluster_series = (
        adata.obs[
            "leiden"
        ]
        .astype(str)
    )

    cluster_names = sorted(
        cluster_series
        .unique()
        .tolist()
    )

    print(
        f"Clusters: {len(cluster_names)}"
    )

    # ------------------------------------------------------------
    # DISCOVERY / HOLDOUT CELL SPLIT
    # ------------------------------------------------------------

    (
        discovery_idx,
        holdout_idx,
    ) = _split_cells_by_cluster(
        cluster_series,
        rng,
    )

    adata_discovery = (
        adata[
            discovery_idx
        ]
        .copy()
    )

    adata_holdout = (
        adata[
            holdout_idx
        ]
        .copy()
    )

    print(
        f"Discovery cells: {adata_discovery.n_obs}"
    )

    print(
        f"Held-out cells: {adata_holdout.n_obs}"
    )

    # ------------------------------------------------------------
    # DISCOVERY-ONLY MARKER RANKING
    # ------------------------------------------------------------

    print(
        "Ranking marker candidates using discovery cells..."
    )

    sc.tl.rank_genes_groups(
        adata_discovery,
        groupby="leiden",
        method="wilcoxon",
        n_genes=min(
            MARKER_CANDIDATES,
            adata_discovery.n_vars,
        ),
    )

    ranked_candidates = {}

    all_candidate_genes = []

    for cluster in cluster_names:

        result_df = (
            sc.get.rank_genes_groups_df(
                adata_discovery,
                group=cluster,
            )
        )

        genes = []

        for gene in (
            result_df[
                "names"
            ]
            .astype(str)
            .tolist()
        ):

            if gene not in genes:
                genes.append(
                    gene
                )

            if (
                len(genes)
                >= MARKER_CANDIDATES
            ):
                break

        ranked_candidates[
            cluster
        ] = genes

        all_candidate_genes.extend(
            genes
        )

    # ------------------------------------------------------------
    # STABLE REFERENCE GENES
    # ------------------------------------------------------------

    stable_pool = (
        _select_stable_reference_genes(
            adata_discovery,
            cluster_names,
            all_candidate_genes,
        )
    )

    reference_genes = (
        stable_pool[
            :N_REFERENCE_GENES
        ]
    )

    print(
        "Reference genes:"
    )

    for gene in reference_genes:
        print(
            f"  {gene}"
        )

    # ------------------------------------------------------------
    # REDUCED GENE MATRIX
    # ------------------------------------------------------------

    required_gene_set = set(
        reference_genes
    )

    for genes in (
        ranked_candidates.values()
    ):
        required_gene_set.update(
            genes
        )

    union_genes = [
        str(gene)
        for gene in adata.var_names
        if str(gene)
        in required_gene_set
    ]

    gene_to_idx = {
        gene: idx
        for idx, gene
        in enumerate(
            union_genes
        )
    }

    x_discovery = (
        adata_discovery[
            :,
            union_genes,
        ].X
    )

    x_holdout = (
        adata_holdout[
            :,
            union_genes,
        ].X
    )

    discovery_labels = (
        adata_discovery.obs[
            "leiden"
        ]
        .astype(str)
        .values
    )

    holdout_labels = (
        adata_holdout.obs[
            "leiden"
        ]
        .astype(str)
        .values
    )

    discovery_cluster_indices = {
        cluster: np.where(
            discovery_labels
            == cluster
        )[0]
        for cluster in cluster_names
    }

    holdout_cluster_indices = {
        cluster: np.where(
            holdout_labels
            == cluster
        )[0]
        for cluster in cluster_names
    }

    # ------------------------------------------------------------
    # RESULTS CONTAINERS
    # ------------------------------------------------------------

    selected_markers = {}

    best_single_markers = {}

    per_cluster_metrics = {}

    evaluation_data = {}

    # ------------------------------------------------------------
    # ONE TARGET CELL TYPE AT A TIME
    # ------------------------------------------------------------

    for cluster in cluster_names:

        print()
        print(
            "=" * 65
        )

        print(
            f"Processing cluster {cluster}"
        )

        print(
            "=" * 65
        )

        # --------------------------------------------------------
        # TRAINING PSEUDOBULK
        # --------------------------------------------------------

        (
            train_expression,
            train_fraction,
        ) = _simulate_targeted_pseudobulk(
            x_sub=x_discovery,
            cluster_names=cluster_names,
            cluster_to_indices=discovery_cluster_indices,
            target_cluster=cluster,
            rng=rng,
            n_samples=N_TRAIN_PSEUDOBULK,
            cells_per_sample=CELLS_PER_PSEUDOBULK,
        )

        # --------------------------------------------------------
        # HELD-OUT TEST PSEUDOBULK
        # --------------------------------------------------------

        (
            test_expression,
            test_fraction,
        ) = _simulate_targeted_pseudobulk(
            x_sub=x_holdout,
            cluster_names=cluster_names,
            cluster_to_indices=holdout_cluster_indices,
            target_cluster=cluster,
            rng=rng,
            n_samples=N_TEST_PSEUDOBULK,
            cells_per_sample=CELLS_PER_PSEUDOBULK,
        )

        # --------------------------------------------------------
        # TRAIN-ONLY PANEL SELECTION
        # --------------------------------------------------------

        (
            panel,
            best_single_gene,
            single_cv_mae,
            panel_cv_mae,
        ) = _select_marker_panel(
            train_expression=train_expression,
            train_fraction=train_fraction,
            candidates=ranked_candidates[
                cluster
            ],
            reference_genes=reference_genes,
            gene_to_idx=gene_to_idx,
        )

        selected_markers[
            cluster
        ] = panel

        best_single_markers[
            cluster
        ] = (
            best_single_gene
        )

        print(
            f"Selected panel: {panel}"
        )

        print(
            f"Single-marker baseline: {best_single_gene}"
        )

        # --------------------------------------------------------
        # FINAL PANEL MODEL
        # --------------------------------------------------------

        (
            panel_prediction,
            panel_test_score,
        ) = _fit_fraction_model(
            train_expression=train_expression,
            train_fraction=train_fraction,
            test_expression=test_expression,
            marker_genes=panel,
            reference_genes=reference_genes,
            gene_to_idx=gene_to_idx,
        )

        # --------------------------------------------------------
        # SINGLE-MARKER BASELINE
        # --------------------------------------------------------

        (
            single_prediction,
            single_test_score,
        ) = _fit_fraction_model(
            train_expression=train_expression,
            train_fraction=train_fraction,
            test_expression=test_expression,
            marker_genes=[
                best_single_gene
            ],
            reference_genes=reference_genes,
            gene_to_idx=gene_to_idx,
        )

        panel_metrics = (
            _calculate_metrics(
                test_fraction,
                panel_prediction,
            )
        )

        single_metrics = (
            _calculate_metrics(
                test_fraction,
                single_prediction,
            )
        )

        per_cluster_metrics[
            cluster
        ] = {
            "panel": panel_metrics,
            "single_marker": single_metrics,
            "panel_cv_mae": float(
                panel_cv_mae
            ),
            "single_marker_cv_mae": float(
                single_cv_mae
            ),
            "panel_size": int(
                len(panel)
            ),
        }

        labels = (
            test_fraction
            >= DETECTION_THRESHOLD
        ).astype(int)

        fraction_bin = []

        for fraction in test_fraction:

            if (
                fraction
                <= LOW_FRACTION_THRESHOLD
            ):
                fraction_bin.append(
                    "low"
                )

            elif (
                fraction
                >= HIGH_FRACTION_THRESHOLD
            ):
                fraction_bin.append(
                    "high"
                )

            else:
                fraction_bin.append(
                    "medium"
                )

        evaluation_data[
            cluster
        ] = {
            "true_fraction": (
                test_fraction.tolist()
            ),
            "panel_prediction": (
                panel_prediction.tolist()
            ),
            "single_marker_prediction": (
                single_prediction.tolist()
            ),
            "panel_score": (
                panel_test_score.tolist()
            ),
            "single_marker_score": (
                single_test_score.tolist()
            ),
            "binary_label": (
                labels.tolist()
            ),
            "fraction_bin": (
                fraction_bin
            ),
        }

        print(
            f"Held-out panel AUC: "
            f"{panel_metrics['auc']:.4f}"
        )

        print(
            f"Held-out panel Spearman: "
            f"{panel_metrics['spearman']:.4f}"
        )

        print(
            f"Held-out panel MAE: "
            f"{panel_metrics['mae']:.4f}"
        )

        print(
            f"Single marker AUC: "
            f"{single_metrics['auc']:.4f}"
        )

    # ============================================================
    # AGGREGATE METRICS
    # ================================================================

    panel_auc_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "auc"
            ]
            for c in cluster_names
        ]
    )

    single_auc_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "auc"
            ]
            for c in cluster_names
        ]
    )

    single_spearman_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "spearman"
            ]
            for c in cluster_names
        ]
    )

    single_pearson_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "pearson"
            ]
            for c in cluster_names
        ]
    )

    single_mae_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "mae"
            ]
            for c in cluster_names
        ]
    )

    single_rmse_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "rmse"
            ]
            for c in cluster_names
        ]
    )

    single_r2_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "r2"
            ]
            for c in cluster_names
        ]
    )

    single_low_mae_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "low_fraction_mae"
            ]
            for c in cluster_names
        ],
        dtype=float,
    )

    single_high_mae_values = np.array(
        [
            per_cluster_metrics[c][
                "single_marker"
            ][
                "high_fraction_mae"
            ]
            for c in cluster_names
        ],
        dtype=float,
    )

    spearman_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "spearman"
            ]
            for c in cluster_names
        ]
    )

    pearson_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "pearson"
            ]
            for c in cluster_names
        ]
    )

    mae_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "mae"
            ]
            for c in cluster_names
        ]
    )

    rmse_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "rmse"
            ]
            for c in cluster_names
        ]
    )

    r2_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "r2"
            ]
            for c in cluster_names
        ]
    )

    low_mae_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "low_fraction_mae"
            ]
            for c in cluster_names
        ],
        dtype=float,
    )

    high_mae_values = np.array(
        [
            per_cluster_metrics[c][
                "panel"
            ][
                "high_fraction_mae"
            ]
            for c in cluster_names
        ],
        dtype=float,
    )

    panel_sizes = np.array(
        [
            len(
                selected_markers[c]
            )
            for c in cluster_names
        ],
        dtype=float,
    )

    metrics = {
        "macro_panel_auc": float(
            np.mean(
                panel_auc_values
            )
        ),
        "macro_single_marker_auc": float(
            np.mean(
                single_auc_values
            )
        ),
        "macro_auc_delta": float(
            np.mean(
                panel_auc_values
                - single_auc_values
            )
        ),
        "macro_spearman": float(
            np.mean(
                spearman_values
            )
        ),
        "macro_pearson": float(
            np.mean(
                pearson_values
            )
        ),
        "macro_mae": float(
            np.mean(
                mae_values
            )
        ),
        "macro_rmse": float(
            np.mean(
                rmse_values
            )
        ),
        "macro_r2": float(
            np.mean(
                r2_values
            )
        ),
        "macro_single_marker_spearman": float(
            np.mean(
                single_spearman_values
            )
        ),
        "macro_single_marker_pearson": float(
            np.mean(
                single_pearson_values
            )
        ),
        "macro_single_marker_mae": float(
            np.mean(
                single_mae_values
            )
        ),
        "macro_single_marker_rmse": float(
            np.mean(
                single_rmse_values
            )
        ),
        "macro_single_marker_r2": float(
            np.mean(
                single_r2_values
            )
        ),
        "single_marker_low_fraction_mae": float(
            np.nanmean(
                single_low_mae_values
            )
        ),
        "single_marker_high_fraction_mae": float(
            np.nanmean(
                single_high_mae_values
            )
        ),
        "low_fraction_mae": float(
            np.nanmean(
                low_mae_values
            )
        ),
        "high_fraction_mae": float(
            np.nanmean(
                high_mae_values
            )
        ),
        "mean_marker_geneset_size": float(
            np.mean(
                panel_sizes
            )
        ),
        "n_clusters": int(
            len(
                cluster_names
            )
        ),
        "n_cells": int(
            adata.n_obs
        ),
        "n_genes": int(
            adata.n_vars
        ),
        "discovery_cells": int(
            adata_discovery.n_obs
        ),
        "holdout_cells": int(
            adata_holdout.n_obs
        ),
        "n_train_pseudobulk": int(
            N_TRAIN_PSEUDOBULK
        ),
        "n_test_pseudobulk": int(
            N_TEST_PSEUDOBULK
        ),
        "cells_per_pseudobulk": int(
            CELLS_PER_PSEUDOBULK
        ),
        "detection_threshold": float(
            DETECTION_THRESHOLD
        ),
    }

    detailed_metrics = {
        **metrics,
        "reference_genes": reference_genes,
        "selected_markers": selected_markers,
        "best_single_markers": (
            best_single_markers
        ),
        "per_cluster": (
            per_cluster_metrics
        ),
    }

    # ============================================================
    # SAVE DETAILED RESULTS
    # ================================================================

    with (
        run_dir
        / "model_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            detailed_metrics,
            f,
            indent=2,
        )

    with (
        run_dir
        / "selected_markers.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selected_markers,
            f,
            indent=2,
        )

    with (
        run_dir
        / "selected_references.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            reference_genes,
            f,
            indent=2,
        )

    with (
        run_dir
        / "best_single_markers.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_single_markers,
            f,
            indent=2,
        )

    evaluation_output = {
        "cluster_names": (
            cluster_names
        ),
        "detection_threshold": (
            DETECTION_THRESHOLD
        ),
        "low_fraction_threshold": (
            LOW_FRACTION_THRESHOLD
        ),
        "high_fraction_threshold": (
            HIGH_FRACTION_THRESHOLD
        ),
        "clusters": (
            evaluation_data
        ),
    }

    with (
        run_dir
        / "evaluation_data.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation_output,
            f,
        )

    # ============================================================
    # AI-SCIENTIST FINAL INFO
    # ================================================================

    final_info = {
        # --------------------------------------------------------
        # MULTI-GENE PANEL
        # --------------------------------------------------------
        "macro_panel_auc": {
            "means": float(np.mean(panel_auc_values)),
            "stds": float(np.std(panel_auc_values)),
        },
        "macro_spearman": {
            "means": float(np.mean(spearman_values)),
            "stds": float(np.std(spearman_values)),
        },
        "macro_pearson": {
            "means": float(np.mean(pearson_values)),
            "stds": float(np.std(pearson_values)),
        },
        "macro_mae": {
            "means": float(np.mean(mae_values)),
            "stds": float(np.std(mae_values)),
        },
        "macro_rmse": {
            "means": float(np.mean(rmse_values)),
            "stds": float(np.std(rmse_values)),
        },
        "macro_r2": {
            "means": float(np.mean(r2_values)),
            "stds": float(np.std(r2_values)),
        },
        "low_fraction_mae": {
            "means": float(np.nanmean(low_mae_values)),
            "stds": float(np.nanstd(low_mae_values)),
        },
        "high_fraction_mae": {
            "means": float(np.nanmean(high_mae_values)),
            "stds": float(np.nanstd(high_mae_values)),
        },

        # --------------------------------------------------------
        # BEST SINGLE-MARKER BASELINE
        # --------------------------------------------------------
        "macro_single_marker_auc": {
            "means": float(np.mean(single_auc_values)),
            "stds": float(np.std(single_auc_values)),
        },
        "macro_single_marker_spearman": {
            "means": float(np.mean(single_spearman_values)),
            "stds": float(np.std(single_spearman_values)),
        },
        "macro_single_marker_pearson": {
            "means": float(np.mean(single_pearson_values)),
            "stds": float(np.std(single_pearson_values)),
        },
        "macro_single_marker_mae": {
            "means": float(np.mean(single_mae_values)),
            "stds": float(np.std(single_mae_values)),
        },
        "macro_single_marker_rmse": {
            "means": float(np.mean(single_rmse_values)),
            "stds": float(np.std(single_rmse_values)),
        },
        "macro_single_marker_r2": {
            "means": float(np.mean(single_r2_values)),
            "stds": float(np.std(single_r2_values)),
        },
        "single_marker_low_fraction_mae": {
            "means": float(np.nanmean(single_low_mae_values)),
            "stds": float(np.nanstd(single_low_mae_values)),
        },
        "single_marker_high_fraction_mae": {
            "means": float(np.nanmean(single_high_mae_values)),
            "stds": float(np.nanstd(single_high_mae_values)),
        },

        # --------------------------------------------------------
        # DIRECT PANEL-VS-SINGLE COMPARISON
        # --------------------------------------------------------
        "macro_auc_delta": {
            "means": float(
                np.mean(
                    panel_auc_values
                    - single_auc_values
                )
            ),
            "stds": float(
                np.std(
                    panel_auc_values
                    - single_auc_values
                )
            ),
        },

        # --------------------------------------------------------
        # COMPLEXITY
        # --------------------------------------------------------
        "mean_marker_geneset_size": {
            "means": float(np.mean(panel_sizes)),
            "stds": float(np.std(panel_sizes)),
        },
    }

    with (
        run_dir
        / "final_info.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_info,
            f,
            indent=2,
        )

    # ============================================================
    # SUMMARY
    # ================================================================

    print()
    print(
        "=" * 65
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        f"Macro panel AUC:        "
        f"{metrics['macro_panel_auc']:.4f}"
    )

    print(
        f"Single-marker AUC:      "
        f"{metrics['macro_single_marker_auc']:.4f}"
    )

    print(
        f"Macro AUC delta:        "
        f"{metrics['macro_auc_delta']:.4f}"
    )

    print(
        f"Macro Spearman:         "
        f"{metrics['macro_spearman']:.4f}"
    )

    print(
        f"Macro Pearson:          "
        f"{metrics['macro_pearson']:.4f}"
    )

    print(
        f"Macro R2:               "
        f"{metrics['macro_r2']:.4f}"
    )

    print(
        f"Macro MAE:              "
        f"{metrics['macro_mae']:.4f}"
    )

    print(
        f"Macro RMSE:             "
        f"{metrics['macro_rmse']:.4f}"
    )

    print(
        f"Low-fraction MAE:       "
        f"{metrics['low_fraction_mae']:.4f}"
    )

    print(
        f"High-fraction MAE:      "
        f"{metrics['high_fraction_mae']:.4f}"
    )

    print(
        f"Mean panel size:        "
        f"{metrics['mean_marker_geneset_size']:.2f}"
    )

    print(
        f"\nSaved to: {run_dir}"
    )


# ================================================================
# COMMAND LINE
# ================================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Develop and evaluate qPCR marker panels "
            "from scRNA-seq data."
        )
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="run_0",
        help="Output directory",
    )

    args = parser.parse_args()

    main(
        args.out_dir
    )