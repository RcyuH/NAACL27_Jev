"""Rebuild follow-up figures/tables from saved aggregate results, CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#3366CC", "#D97706", "#159A8C", "#AD4B85"]


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append, don't replace_suffix: a rate such as r0.05 is part of the name.
    # Replacing the suffix would silently overwrite every rate with "r0.png".
    fig.savefig(Path(str(path) + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(Path(str(path) + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _edge(value):
    return (
        value.replace("identity_attack", "identity")
        .replace("toxicity", "toxicity")
        .replace("->", " → ")
    )


def _local_figure(table, path):
    common = table[table.support_scope == "common_grid"]
    edges = list(common.edge.unique())
    fig, axes = plt.subplots(
        len(edges), 2, figsize=(10, 3.0 * len(edges)), squeeze=False, constrained_layout=True
    )
    for row, edge in enumerate(edges):
        for column, metric in enumerate(("signed_error", "absolute_local_score_gap")):
            ax = axes[row, column]
            for color, (rate, part) in zip(COLORS, common[common.edge == edge].groupby("rate")):
                part = part.sort_values("bandwidth")
                ax.plot(part.bandwidth, part[metric], "o-", color=color, label=f"{rate:.0%} routed")
                ax.vlines(
                    part.bandwidth, part[f"{metric}_ci_low"], part[f"{metric}_ci_high"], color=color
                )
            ax.axhline(0, color="#64748B", linewidth=0.8, linestyle="--")
            ax.set_xscale("log")
            ax.set_xticks(
                sorted(common.bandwidth.unique()),
                [f"{v:g}" for v in sorted(common.bandwidth.unique())],
            )
            ax.set_xlabel("Gaussian bandwidth (probability units)")
            ax.set_title(_edge(edge))
            ax.set_ylabel("Signed-error gap" if column == 0 else "Mean absolute local score gap")
            ax.grid(alpha=0.2)
            if row == 0 and column == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        "Local score balancing — shared support across bandwidths\n"
        "Pointwise cluster-bootstrap intervals; finite-bandwidth diagnostic",
        fontsize=12,
    )
    _save(fig, path)


def _null_figures(table, directory):
    for rate, selected in table.groupby("rate"):
        edges = list(selected.edge.unique())
        fig, axes = plt.subplots(
            1, len(edges), figsize=(4.3 * len(edges), 3.8), squeeze=False, constrained_layout=True
        )
        for ax, edge in zip(axes.flat, edges):
            part = selected[selected.edge == edge]
            observed = part.drop_duplicates("n_bins").sort_values("n_bins")
            ax.plot(
                observed.n_bins,
                observed.observed_signed_error,
                "ko-",
                label="Observed held-out gap",
            )
            for color, (kind, values) in zip(COLORS, part.groupby("null_model")):
                values = values.sort_values("n_bins")
                ax.plot(
                    values.n_bins,
                    values.fitted_null_expected_gap,
                    "--",
                    color=color,
                    label=f"Null mean ({kind})",
                )
                ax.fill_between(
                    values.n_bins,
                    values.predictive_low,
                    values.predictive_high,
                    color=color,
                    alpha=0.16,
                )
            ax.axhline(0, color="#64748B", linewidth=0.8)
            ax.set_title(_edge(edge))
            ax.set_xlabel("Pooled quantile bins")
            ax.set_ylabel("Signed-error gap")
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"{rate:.0%} routing — observed vs fitted invariant-score null\n"
            "Shading: model-based predictive bands, NOT confidence intervals for an effect",
            fontsize=11,
        )
        _save(fig, directory / f"binning_null_r{rate:g}")


def _reliability_figures(table, directory):
    for (edge, rate), part in table.groupby(["edge", "rate"]):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
        for ax, kind in zip(axes, ("raw", "platt", "spline", "spline_route")):
            points = part[part.calibrator == kind]
            maximum = max(
                0.05, float(points[["mean_probability", "outcome_rate"]].max().max()) * 1.1
            )
            for color, group in zip(
                ("#64748B", "#DB5461", "#159A8C"), ("all", "routed", "non_routed")
            ):
                values = points[points.group == group].sort_values("mean_raw_score")
                ax.plot(
                    values.mean_probability,
                    values.outcome_rate,
                    "o-",
                    color=color,
                    markersize=3,
                    linewidth=1,
                    label=group.replace("_", " "),
                )
            ax.plot([0, maximum], [0, maximum], "k--", alpha=0.5, linewidth=0.8)
            ax.set(
                xlim=(0, maximum),
                ylim=(0, maximum),
                xlabel="Mean predicted probability",
                ylabel="Observed frequency",
                title=kind.replace("_", " + "),
            )
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(
            f"Held-out calibration: {_edge(edge)}, target route {rate:.0%}\n"
            "Shared raw-score strata; descriptive points, no curve confidence bands",
            fontsize=11,
        )
        name = edge.replace("->", "_to_")
        _save(fig, directory / f"heldout_{name}_r{rate:g}")


def _matrix_figure(table, path):
    edges, rates = list(table.edge.unique()), sorted(table.rate.unique(), reverse=True)
    fig, ax = plt.subplots(figsize=(6, max(5, len(edges) * 0.39)), constrained_layout=True)
    values = table.pivot(index="edge", columns="rate", values="signed_error").reindex(
        index=edges, columns=rates
    )
    array = values.to_numpy()
    limit = max(0.01, np.nanmax(abs(array))) if np.isfinite(array).any() else 0.01
    artist = ax.imshow(
        np.ma.masked_invalid(array), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
    )
    for i, edge in enumerate(edges):
        for j, rate in enumerate(rates):
            row = table[(table.edge == edge) & (table.rate == rate)]
            if row.empty or not np.isfinite(array[i, j]):
                label = "unsupported"
            else:
                q = row.iloc[0].get("q_signed_error", np.nan)
                label = f"{array[i, j]:+.3f}" + ("*" if q < 0.05 else "")
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(array[i, j]) > limit * 0.6 else "#172033",
            )
    ax.set_xticks(range(len(rates)), [f"{rate:.0%} routed" for rate in rates])
    ax.set_yticks(range(len(edges)), [_edge(edge) for edge in edges])
    ax.set_title("Complete directed routing matrix\n* BH q < .05 within model/prompt family")
    fig.colorbar(artist, ax=ax, label="Within-bin signed-error gap")
    _save(fig, path)


def _escape(value):
    return (
        str(value)
        .replace("\\", "\\textbackslash{} ")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _latex(table, columns, caption, path):
    # No pandas.to_latex/Jinja dependency. booktabs + xcolor[table] + longtable.
    lines = [
        "% Requires \\usepackage{booktabs,longtable} and \\usepackage[table]{xcolor}",
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{" + "l" * len(columns) + "}",
        "\\caption{" + caption + "}\\\\",
        "\\toprule",
        "\\rowcolor{blue!12} " + " & ".join(_escape(title) for _, title in columns) + " \\\\",
        "\\midrule\\endfirsthead",
        "\\toprule",
        " & ".join(_escape(title) for _, title in columns) + " \\\\",
        "\\midrule\\endhead",
    ]
    previous = None
    for index, (_, row) in enumerate(table.iterrows()):
        entries = []
        for key, _ in columns:
            value = row.get(key, np.nan)
            if key == "edge" and value == previous:
                text = ""
            elif isinstance(value, (float, np.floating)):
                text = (
                    "--"
                    if not np.isfinite(value)
                    else (f"{value:.0%}" if key == "rate" else f"{value:.4f}")
                )
            else:
                text = str(value)
            entries.append(_escape(text))
        previous = row.get("edge")
        lines.append(
            ("\\rowcolor{blue!4} " if index % 2 == 0 else "") + " & ".join(entries) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}", "\\end{small}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_strengthening_report(root: str | Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    publication = root / "publication_tables"
    publication.mkdir(exist_ok=True)
    summaries = [
        "# Additional score-resolution / held-out audit",
        "",
        f"Status: **{manifest['status']}**. Quick mode: **{manifest['quick_mode']}**.",
        "",
        "These are exploratory follow-ups. Intervals are pointwise. Statistical significance",
        "of finite-bin/finite-bandwidth gaps does not prove exact-score conditional dependence.",
        "",
        "Null bands are model-based predictive intervals under fitted invariant-score models.",
        "Their tail fractions are NOT calibrated p-values and are not BH-adjusted.",
        "",
        "Held-out calibration intervals refit training and resample evaluation text clusters.",
        "They condition on the saved train/evaluation split, not on a new random split.",
        "",
        "All plots are exported as PNG and vector PDF. Longtables target an appendix;",
        "include them in a one-column appendix (or adapt selected rows to a main-paper table).",
        "",
    ]
    definitions = {
        "local_balance": [
            ("edge", "Edge"),
            ("rate", "Route"),
            ("bandwidth", "h"),
            ("signed_error", "Gap"),
            ("signed_error_ci_low", "CI low"),
            ("signed_error_ci_high", "CI high"),
            ("absolute_local_score_gap", "Imbalance"),
            ("common_support", "Support"),
        ],
        "binning_null": [
            ("edge", "Edge"),
            ("rate", "Route"),
            ("null_model", "Null"),
            ("n_bins", "Bins"),
            ("observed_signed_error", "Observed"),
            ("fitted_null_expected_gap", "Null mean"),
            ("predictive_low", "Band low"),
            ("predictive_high", "Band high"),
        ],
        "heldout_calibration": [
            ("edge", "Edge"),
            ("rate", "Route"),
            ("calibrator", "Method"),
            ("group", "Group"),
            ("signed_error", "SE"),
            ("brier", "Brier"),
            ("ece", "ECE"),
            ("positive_n", "Positives"),
        ],
        "heldout_paired_comparisons": [
            ("edge", "Edge"),
            ("rate", "Route"),
            ("candidate", "Candidate"),
            ("reference", "Reference"),
            ("group", "Group"),
            ("paired_delta", "Delta Brier"),
            ("ci_low", "CI low"),
            ("ci_high", "CI high"),
        ],
        "complete_matrix": [
            ("edge", "Edge"),
            ("rate", "Route"),
            ("signed_error", "Gap"),
            ("signed_error_ci_low", "CI low"),
            ("signed_error_ci_high", "CI high"),
            ("q_signed_error", "BH q"),
            ("common_support", "Support"),
        ],
    }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        for name, columns in definitions.items():
            path = root / "tables" / f"{name}.csv"
            if not path.exists():
                summaries.extend([f"- Missing stage output: `{name}`", ""])
                continue
            full = pd.read_csv(path)
            for prediction_set, table in full.groupby("prediction_set", sort=False):
                directory = root / "figures" / prediction_set
                directory.mkdir(parents=True, exist_ok=True)
                if name == "local_balance":
                    _local_figure(table, directory / "local_balance")
                    table = table[table.support_scope == "common_grid"]
                elif name == "binning_null":
                    _null_figures(table, directory)
                elif name == "complete_matrix":
                    _matrix_figure(table, directory / "complete_matrix")
                elif name == "heldout_paired_comparisons":
                    table = table[table.metric == "brier"]
                caption = name.replace("_", " ").capitalize() + ": " + _escape(prediction_set)
                if name == "local_balance":
                    caption += (
                        ". Common-grid support across bandwidths; pointwise bootstrap intervals."
                    )
                if name == "binning_null":
                    caption += ". Model-based predictive bands, not effect confidence intervals."
                _latex(table, columns, caption, publication / f"{prediction_set}_{name}.tex")
                summaries.append(f"- `{prediction_set}` / `{name}`: {len(table)} reported rows.")
        reliability = root / "tables" / "heldout_reliability.csv"
        if reliability.exists():
            for prediction_set, table in pd.read_csv(reliability).groupby("prediction_set"):
                _reliability_figures(table, root / "figures" / prediction_set)
    summaries.extend(
        [
            "",
            "## Quality checks before interpretation",
            "",
            "- Inspect support, valid bootstrap counts, and calibration_fit_diagnostics.csv.",
            "- Compare local imbalance as bandwidth shrinks; do not select bandwidth by significance.",
            "- Check null-model sensitivity AND held-out fit quality before interpreting null excess.",
            "- A negative paired Brier/log-loss delta favors the candidate; SE sign is not quality.",
            "- ECE is a finite-bin descriptive metric; reliability curves have no confidence bands.",
            "- Null simulations assume independent Bernoulli outcomes conditional on scores;",
            "  text-cluster nuisance resampling does not remove that model assumption.",
            "- qualitative_sample_ids.csv contains randomly sampled IDs, not cherry-picked text.",
            "- Use identical IDs to retrieve text locally; do not publish sensitive examples unreviewed.",
            "- The all-pairs matrix has a new multiplicity family; q-values may differ from the draft.",
            "- Do not silently replace a primary estimator with a more favorable follow-up.",
        ]
    )
    summaries.extend(["", "## Automatically checked diagnostics", ""])
    for prediction_set, metadata in manifest["inputs"].items():
        summaries.append(
            f"- `{prediction_set}`: {metadata['rows']:,} rows / "
            f"{metadata['independent_units']:,} bootstrap units; grouping `{metadata['grouping']}`."
        )
        if "fallback" in metadata["grouping"]:
            summaries.append(
                "  **WARNING:** no text/hash supplied; duplicate-text leakage cannot be checked."
            )
    for filename in ("null_fit_diagnostics", "calibration_fit_diagnostics"):
        source = root / "tables" / f"{filename}.csv"
        if source.exists():
            frame = pd.read_csv(source)
            failures = int(frame.bootstrap_fit_failures.sum())
            summaries.append(
                f"- `{filename}`: {failures} failed bootstrap fits across {len(frame)} fits/configurations."
            )
    for filename in ("local_balance", "heldout_calibration", "complete_matrix"):
        source = root / "tables" / f"{filename}.csv"
        if source.exists():
            frame = pd.read_csv(source)
            unsupported = int((~frame.signed_error_ci_supported.astype(bool)).sum())
            summaries.append(
                f"- `{filename}`: {unsupported}/{len(frame)} rows lack a supported signed-error CI."
            )
    source = root / "tables" / "heldout_calibration.csv"
    if source.exists():
        frame = pd.read_csv(source)
        sparse = frame[
            (frame.calibrator == "raw") & (frame.group == "routed") & (frame.positive_n < 20)
        ]
        summaries.append(
            f"- {len(sparse)} held-out routed comparisons have fewer than 20 positive labels; "
            "interpret these cautiously regardless of interval width."
        )
    (root / "REPORT.md").write_text("\n".join(summaries) + "\n", encoding="utf-8")
    return {
        "report": str(root / "REPORT.md"),
        "figures": str(root / "figures"),
        "publication_tables": str(publication),
    }
