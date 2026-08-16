from __future__ import annotations

import json
import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from scipy import sparse
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)


# ================================================================
# CONFIGURATION
# ================================================================

RNG_SEED = 42

DATA_FILE = "input_single_cell_pbmc.h5ad"

rng = np.random.default_rng(
    RNG_SEED
)


# ================================================================
# HELPERS
# ================================================================


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def run_sort_key(name):
    """
    Sort run_0, run_1, run_2, ... numerically.
    """

    match = re.match(
        r"run_(\d+)$",
        name,
    )

    if match:
        return int(
            match.group(1)
        )

    return 10**9


def find_runs():
    folders = sorted(
        os.listdir("."),
        key=run_sort_key,
    )

    return [
        folder
        for folder in folders
        if (
            folder.startswith("run_")
            and osp.isdir(folder)
            and osp.exists(
                osp.join(
                    folder,
                    "evaluation_data.json",
                )
            )
            and osp.exists(
                osp.join(
                    folder,
                    "model_metrics.json",
                )
            )
            and osp.exists(
                osp.join(
                    folder,
                    "selected_markers.json",
                )
            )
        )
    ]


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()

    return np.asarray(x)


# ================================================================
# FIND ALL VALID RUNS
# ================================================================


run_folders = find_runs()

if not run_folders:
    raise FileNotFoundError(
        "No completed experiment runs were found."
    )

print(
    "Found runs for plotting: "
    + ", ".join(run_folders)
)


# ================================================================
# LOAD DATASET
# ================================================================


adata = None

if osp.exists(DATA_FILE):

    print(
        f"Loading dataset for biological plots: {DATA_FILE}"
    )

    adata = sc.read_h5ad(
        DATA_FILE
    )

else:

    print(
        f"Dataset {DATA_FILE} not found. "
        "Dataset-level figures will be skipped."
    )


# ================================================================
# FIGURE 1
# DATASET-LEVEL UMAP
# ================================================================


if (
    adata is not None
    and "X_umap" in adata.obsm
    and "leiden" in adata.obs
):

    print()
    print(
        "Generating dataset-level UMAP"
    )

    umap = np.asarray(
        adata.obsm[
            "X_umap"
        ]
    )

    cell_types = (
        adata.obs[
            "leiden"
        ]
        .astype(str)
        .values
    )

    unique_types = sorted(
        np.unique(
            cell_types
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            7,
        )
    )

    for cell_type in unique_types:

        mask = (
            cell_types
            == cell_type
        )

        ax.scatter(
            umap[
                mask,
                0,
            ],
            umap[
                mask,
                1,
            ],
            s=18,
            alpha=0.70,
            label=str(
                cell_type
            ),
        )

    ax.set_title(
        "PBMC cell-type landscape"
    )

    ax.set_xlabel(
        "UMAP 1"
    )

    ax.set_ylabel(
        "UMAP 2"
    )

    ax.legend(
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
        fontsize=8,
        frameon=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    fig.savefig(
        "umap_cell_types.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved: umap_cell_types.png"
    )


# ================================================================
# PLOT EACH COMPLETED RUN
# ================================================================


for run in run_folders:

    print()
    print(
        "=" * 65
    )

    print(
        f"Generating figures for {run}"
    )

    print(
        "=" * 65
    )

    evaluation = load_json(
        osp.join(
            run,
            "evaluation_data.json",
        )
    )

    metrics = load_json(
        osp.join(
            run,
            "model_metrics.json",
        )
    )

    selected_markers = load_json(
        osp.join(
            run,
            "selected_markers.json",
        )
    )

    cluster_names = (
        evaluation[
            "cluster_names"
        ]
    )

    cluster_data = (
        evaluation[
            "clusters"
        ]
    )

    n_clusters = len(
        cluster_names
    )


    # ============================================================
    # FIGURE 2
    # SELECTED-MARKER EXPRESSION
    # ============================================================

    if (
        adata is not None
        and "leiden" in adata.obs
    ):

        print(
            "Generating selected-marker expression plot"
        )

        marker_entries = []

        for target_cell_type in cluster_names:

            for gene in selected_markers.get(
                target_cell_type,
                [],
            ):

                marker_entries.append(
                    (
                        target_cell_type,
                        gene,
                    )
                )

        marker_entries = list(
            dict.fromkeys(
                marker_entries
            )
        )

        valid_entries = [
            (
                target,
                gene,
            )
            for target, gene
            in marker_entries
            if gene
            in adata.var_names
        ]

        if valid_entries:

            genes = [
                gene
                for _, gene
                in valid_entries
            ]

            gene_indices = [
                adata.var_names.get_loc(
                    gene
                )
                for gene
                in genes
            ]

            expression = to_dense(
                adata[
                    :,
                    gene_indices,
                ].X
            )

            dataset_labels = (
                adata.obs[
                    "leiden"
                ]
                .astype(str)
                .values
            )

            mean_expression = np.zeros(
                (
                    len(
                        valid_entries
                    ),
                    len(
                        cluster_names
                    ),
                ),
                dtype=float,
            )

            fraction_expressed = np.zeros_like(
                mean_expression
            )

            for cell_idx, cell_type in enumerate(
                cluster_names
            ):

                mask = (
                    dataset_labels
                    == cell_type
                )

                if not np.any(
                    mask
                ):
                    continue

                cell_expression = (
                    expression[
                        mask,
                        :
                    ]
                )

                mean_expression[
                    :,
                    cell_idx,
                ] = np.mean(
                    cell_expression,
                    axis=0,
                )

                fraction_expressed[
                    :,
                    cell_idx,
                ] = np.mean(
                    cell_expression
                    > 0,
                    axis=0,
                )

            row_mean = np.mean(
                mean_expression,
                axis=1,
                keepdims=True,
            )

            row_std = np.std(
                mean_expression,
                axis=1,
                keepdims=True,
            )

            normalized_expression = (
                mean_expression
                - row_mean
            ) / (
                row_std
                + 1e-8
            )

            fig_height = max(
                5.5,
                0.48
                * len(
                    valid_entries
                ),
            )

            fig_width = max(
                10,
                1.25
                * len(
                    cluster_names
                ),
            )

            fig, ax = plt.subplots(
                figsize=(
                    fig_width,
                    fig_height,
                )
            )

            for gene_idx, (
                target_cell_type,
                gene,
            ) in enumerate(
                valid_entries
            ):

                for cell_idx, cell_type in enumerate(
                    cluster_names
                ):

                    dot_size = (
                        30
                        + 260
                        * fraction_expressed[
                            gene_idx,
                            cell_idx,
                        ]
                    )

                    ax.scatter(
                        cell_idx,
                        gene_idx,
                        s=dot_size,
                        c=[
                            normalized_expression[
                                gene_idx,
                                cell_idx,
                            ]
                        ],
                        cmap="viridis",
                        vmin=-2,
                        vmax=2,
                        alpha=0.85,
                        edgecolors="none",
                    )

            row_labels = [
                (
                    f"{gene} "
                    f"({target})"
                )
                for target, gene
                in valid_entries
            ]

            ax.set_xticks(
                np.arange(
                    len(
                        cluster_names
                    )
                )
            )

            ax.set_xticklabels(
                cluster_names,
                rotation=35,
                ha="right",
            )

            ax.set_yticks(
                np.arange(
                    len(
                        row_labels
                    )
                )
            )

            ax.set_yticklabels(
                row_labels
            )

            ax.invert_yaxis()

            ax.set_xlabel(
                "Cell type"
            )

            ax.set_ylabel(
                "Selected marker gene "
                "(target cell type)"
            )

            ax.set_title(
                "Expression specificity of selected marker genes "
                f"— {run}"
            )

            ax.grid(
                alpha=0.10,
            )

            norm = matplotlib.colors.Normalize(
                vmin=-2,
                vmax=2,
            )

            mapper = matplotlib.cm.ScalarMappable(
                norm=norm,
                cmap="viridis",
            )

            mapper.set_array([])

            colorbar = fig.colorbar(
                mapper,
                ax=ax,
                pad=0.02,
            )

            colorbar.set_label(
                "Relative mean expression"
            )

            fig.tight_layout()

            filename = (
                f"selected_marker_expression_{run}.png"
            )

            fig.savefig(
                filename,
                dpi=240,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

            print(
                f"Saved: {filename}"
            )


    # ============================================================
    # FIGURE 3
    # ROC CURVES
    # ============================================================

    print(
        "Generating ROC curves"
    )

    n_cols = min(
        3,
        n_clusters,
    )

    n_rows = int(
        np.ceil(
            n_clusters
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            5
            * n_cols,
            4
            * n_rows,
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for idx, cluster in enumerate(
        cluster_names
    ):

        ax = axes[idx]

        data = (
            cluster_data[
                cluster
            ]
        )

        y = np.asarray(
            data[
                "binary_label"
            ],
            dtype=int,
        )

        panel_pred = np.asarray(
            data[
                "panel_prediction"
            ],
            dtype=float,
        )

        single_pred = np.asarray(
            data[
                "single_marker_prediction"
            ],
            dtype=float,
        )

        panel_fpr, panel_tpr, _ = (
            roc_curve(
                y,
                panel_pred,
            )
        )

        single_fpr, single_tpr, _ = (
            roc_curve(
                y,
                single_pred,
            )
        )

        panel_cluster_auc = (
            roc_auc_score(
                y,
                panel_pred,
            )
        )

        single_cluster_auc = (
            roc_auc_score(
                y,
                single_pred,
            )
        )

        ax.plot(
            panel_fpr,
            panel_tpr,
            linewidth=2.2,
            label=(
                "Marker panel "
                f"AUC={panel_cluster_auc:.3f}"
            ),
        )

        ax.plot(
            single_fpr,
            single_tpr,
            linestyle="--",
            linewidth=2,
            label=(
                "Single marker "
                f"AUC={single_cluster_auc:.3f}"
            ),
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle=":",
        )

        ax.set_title(
            str(
                cluster
            )
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_xlim(
            0,
            1,
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.grid(
            alpha=0.2,
        )

        ax.legend(
            fontsize=8,
            loc="lower right",
        )

    for ax in axes[
        n_clusters:
    ]:
        ax.axis(
            "off"
        )

    fig.suptitle(
        (
            "Held-out cell-type detection "
            f"performance — {run}"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    filename = (
        f"roc_curves_{run}.png"
    )

    fig.savefig(
        filename,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {filename}"
    )


    # ============================================================
    # FIGURE 4
    # TRUE VS PREDICTED
    # ============================================================

    print(
        "Generating true-vs-predicted fraction plot"
    )

    n_cols = 2

    n_rows = int(
        np.ceil(
            n_clusters
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            13,
            4.5
            * n_rows,
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for idx, cluster in enumerate(
        cluster_names
    ):

        ax = axes[idx]

        data = (
            cluster_data[
                cluster
            ]
        )

        true_fraction = np.asarray(
            data[
                "true_fraction"
            ],
            dtype=float,
        )

        prediction = np.asarray(
            data[
                "panel_prediction"
            ],
            dtype=float,
        )

        bins = np.asarray(
            data[
                "fraction_bin"
            ]
        )

        for regime, marker in [
            (
                "low",
                "o",
            ),
            (
                "medium",
                "s",
            ),
            (
                "high",
                "^",
            ),
        ]:

            mask = (
                bins
                == regime
            )

            ax.scatter(
                true_fraction[
                    mask
                ],
                prediction[
                    mask
                ],
                s=24,
                alpha=0.55,
                marker=marker,
                label=(
                    regime.capitalize()
                ),
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.4,
            label="Ideal",
        )

        cluster_metrics = (
            metrics[
                "per_cluster"
            ][cluster][
                "panel"
            ]
        )

        ax.text(
            0.03,
            0.97,
            (
                f"ρ="
                f"{cluster_metrics['spearman']:.3f}\n"
                f"MAE="
                f"{cluster_metrics['mae']:.3f}\n"
                f"RMSE="
                f"{cluster_metrics['rmse']:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
        )

        ax.set_xlim(
            0,
            1,
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.set_xlabel(
            "True cell fraction"
        )

        ax.set_ylabel(
            "Predicted cell fraction"
        )

        ax.set_title(
            str(
                cluster
            )
        )

        ax.grid(
            alpha=0.15,
        )

        ax.legend(
            fontsize=8,
        )

    for ax in axes[
        n_clusters:
    ]:
        ax.axis(
            "off"
        )

    fig.suptitle(
        (
            "Quantitative cell-fraction prediction "
            f"— {run}"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    filename = (
        "fraction_prediction_scatter_"
        f"{run}.png"
    )

    fig.savefig(
        filename,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {filename}"
    )


    # ============================================================
    # FIGURE 5
    # RESIDUAL PLOT
    # ============================================================

    print(
        "Generating residual plot"
    )

    n_cols = 2

    n_rows = int(
        np.ceil(
            n_clusters
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            13,
            4.2
            * n_rows,
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for idx, cluster in enumerate(
        cluster_names
    ):

        ax = axes[idx]

        data = (
            cluster_data[
                cluster
            ]
        )

        true_fraction = np.asarray(
            data[
                "true_fraction"
            ],
            dtype=float,
        )

        prediction = np.asarray(
            data[
                "panel_prediction"
            ],
            dtype=float,
        )

        residual = (
            prediction
            - true_fraction
        )

        bins = np.asarray(
            data[
                "fraction_bin"
            ]
        )

        for regime, marker in [
            (
                "low",
                "o",
            ),
            (
                "medium",
                "s",
            ),
            (
                "high",
                "^",
            ),
        ]:

            mask = (
                bins
                == regime
            )

            ax.scatter(
                true_fraction[
                    mask
                ],
                residual[
                    mask
                ],
                s=22,
                alpha=0.50,
                marker=marker,
                label=(
                    regime.capitalize()
                ),
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1.5,
        )

        ax.axvline(
            evaluation[
                "low_fraction_threshold"
            ],
            linestyle=":",
            linewidth=1,
        )

        ax.axvline(
            evaluation[
                "high_fraction_threshold"
            ],
            linestyle=":",
            linewidth=1,
        )

        ax.set_xlabel(
            "True cell fraction"
        )

        ax.set_ylabel(
            "Prediction error"
        )

        ax.set_title(
            str(
                cluster
            )
        )

        ax.grid(
            alpha=0.15,
        )

        ax.legend(
            fontsize=8,
        )

    for ax in axes[
        n_clusters:
    ]:
        ax.axis(
            "off"
        )

    fig.suptitle(
        (
            "Prediction residuals across "
            f"abundance regimes — {run}"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    filename = (
        f"residual_scatter_{run}.png"
    )

    fig.savefig(
        filename,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {filename}"
    )


    # ============================================================
    # FIGURE 6
    # PANEL VS SINGLE-MARKER
    # ============================================================

    print(
        "Generating panel-vs-single-marker comparison"
    )

    panel_auc = np.asarray(
        [
            metrics[
                "per_cluster"
            ][cluster][
                "panel"
            ][
                "auc"
            ]
            for cluster
            in cluster_names
        ],
        dtype=float,
    )

    single_auc = np.asarray(
        [
            metrics[
                "per_cluster"
            ][cluster][
                "single_marker"
            ][
                "auc"
            ]
            for cluster
            in cluster_names
        ],
        dtype=float,
    )

    panel_mae = np.asarray(
        [
            metrics[
                "per_cluster"
            ][cluster][
                "panel"
            ][
                "mae"
            ]
            for cluster
            in cluster_names
        ],
        dtype=float,
    )

    single_mae = np.asarray(
        [
            metrics[
                "per_cluster"
            ][cluster][
                "single_marker"
            ][
                "mae"
            ]
            for cluster
            in cluster_names
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            14,
            6,
        ),
    )

    x = np.arange(
        n_clusters
    )


    ax = axes[0]

    for idx in range(
        n_clusters
    ):

        ax.plot(
            [
                idx - 0.12,
                idx + 0.12,
            ],
            [
                single_auc[
                    idx
                ],
                panel_auc[
                    idx
                ],
            ],
            alpha=0.40,
            linewidth=1.5,
        )

    ax.scatter(
        x - 0.12,
        single_auc,
        s=65,
        marker="s",
        label="Best single marker",
    )

    ax.scatter(
        x + 0.12,
        panel_auc,
        s=65,
        marker="o",
        label="Multi-gene panel",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        cluster_names,
        rotation=40,
        ha="right",
    )

    ax.set_ylabel(
        "Held-out ROC AUC"
    )

    ax.set_ylim(
        0,
        1.03,
    )

    ax.set_title(
        "A. Cell-type detection"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.20,
    )


    ax = axes[1]

    for idx in range(
        n_clusters
    ):

        ax.plot(
            [
                idx - 0.12,
                idx + 0.12,
            ],
            [
                single_mae[
                    idx
                ],
                panel_mae[
                    idx
                ],
            ],
            alpha=0.40,
            linewidth=1.5,
        )

    ax.scatter(
        x - 0.12,
        single_mae,
        s=65,
        marker="s",
        label="Best single marker",
    )

    ax.scatter(
        x + 0.12,
        panel_mae,
        s=65,
        marker="o",
        label="Multi-gene panel",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        cluster_names,
        rotation=40,
        ha="right",
    )

    ax.set_ylabel(
        "Mean absolute error"
    )

    ax.set_title(
        "B. Cell-fraction quantification"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.suptitle(
        (
            "Multi-gene panel versus best single-marker baseline "
            f"— {run}"
        ),
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )

    filename = (
        f"panel_vs_single_marker_{run}.png"
    )

    fig.savefig(
        filename,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {filename}"
    )


# ================================================================
# FIGURE 7
# CROSS-RUN COMPARISON
# ================================================================


if len(
    run_folders
) > 1:

    print()
    print(
        "Generating cross-run comparison"
    )

    macro_auc = []
    macro_spearman = []
    macro_mae = []
    macro_auc_delta = []

    for run in run_folders:

        run_metrics = load_json(
            osp.join(
                run,
                "model_metrics.json",
            )
        )

        macro_auc.append(
            run_metrics[
                "macro_panel_auc"
            ]
        )

        macro_spearman.append(
            run_metrics[
                "macro_spearman"
            ]
        )

        macro_mae.append(
            run_metrics[
                "macro_mae"
            ]
        )

        macro_auc_delta.append(
            run_metrics[
                "macro_auc_delta"
            ]
        )

    x = np.arange(
        len(
            run_folders
        )
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            10,
        ),
    )

    axes = axes.ravel()

    axes[0].plot(
        x,
        macro_auc,
        marker="o",
        linewidth=1.8,
    )

    axes[0].set_title(
        "A. Macro panel ROC AUC"
    )

    axes[0].set_ylabel(
        "ROC AUC"
    )

    axes[1].plot(
        x,
        macro_spearman,
        marker="o",
        linewidth=1.8,
    )

    axes[1].set_title(
        "B. Macro Spearman correlation"
    )

    axes[1].set_ylabel(
        "Spearman correlation"
    )

    axes[2].plot(
        x,
        macro_mae,
        marker="o",
        linewidth=1.8,
    )

    axes[2].set_title(
        "C. Macro MAE"
    )

    axes[2].set_ylabel(
        "Mean absolute error"
    )

    axes[3].plot(
        x,
        macro_auc_delta,
        marker="o",
        linewidth=1.8,
    )

    axes[3].axhline(
        0,
        linestyle="--",
        linewidth=1.2,
    )

    axes[3].set_title(
        "D. Panel minus single-marker AUC"
    )

    axes[3].set_ylabel(
        "ROC AUC difference"
    )

    for ax in axes:

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            run_folders,
            rotation=25,
        )

        ax.set_xlabel(
            "Experiment"
        )

        ax.grid(
            alpha=0.20,
        )

    fig.suptitle(
        "Cross-run experiment comparison",
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        "cross_run_comparison.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved: cross_run_comparison.png"
    )


# ================================================================
# FIGURE METADATA
# ================================================================

# Keep lightweight metadata about the figures that were actually generated.
# Do not hardcode scientific conclusions or choose a best run here.
# The writeup stage will use notes.txt to decide where figures belong.

figure_metadata = {}

for filename in sorted(
    f
    for f in os.listdir(".")
    if f.endswith(".png")
):
    figure_metadata[filename] = {
        "filename": filename,
    }

with open(
    "figure_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        figure_metadata,
        f,
        indent=2,
    )

print()
print(
    "Saved: figure_metadata.json"
)


# ================================================================
# COMPLETE
# ================================================================


print()
print(
    "=" * 65
)

print(
    "PLOT GENERATION COMPLETE"
)

print(
    "=" * 65
)