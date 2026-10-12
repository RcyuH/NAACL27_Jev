"""Resumable CPU-only follow-up analyses for the score-stratified paper."""

from __future__ import annotations

import importlib.metadata
import itertools
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .config import config_digest
from .io import atomic_write_json, atomic_write_parquet, sha256_file, text_sha256
from .statistics import benjamini_hochberg, bootstrap_sign_pvalue
from .strengthen_stats import (
    apply_policy,
    bin_coefficients,
    binned_metrics,
    calibration_metrics,
    derived_seed,
    fit_calibrator,
    fit_policy,
    interval,
    local_balance,
    routing_weights,
    split_by_unit,
    tie_uniform,
)

STAGES = ("local", "null", "recalibration", "matrix")
DEFAULTS = {
    "labels": ["toxicity", "identity_attack", "obscene", "insult"],
    "audit_edges": [
        ["toxicity", "identity_attack"],
        ["toxicity", "obscene"],
        ["toxicity", "insult"],
    ],
    "rates": [0.20, 0.10, 0.05, 0.02],
    "matrix_rates": [0.10, 0.05],
    "bootstrap_resamples": 2000,
    "null_resamples": 2000,
    "confidence_level": 0.95,
    "minimum_valid_fraction": 0.95,
    "bandwidths": [0.005, 0.01, 0.02],
    "local_grid_size": 1001,
    "minimum_kernel_ess": 50,
    "bin_counts": [5, 10, 20, 40, 80],
    "matrix_bins": 10,
    "min_per_group": 50,
    "null_models": ["platt", "spline"],
    "calibration_ridge": 1.0,
    "calibration_grid_size": 2049,
    "ece_bins": 15,
    "examples_per_bin_group": 3,
}


def load_settings(path: str | Path | None) -> dict:
    settings = dict(DEFAULTS)
    if path:
        with Path(path).open(encoding="utf-8") as stream:
            supplied = yaml.safe_load(stream)
        if not isinstance(supplied, dict):
            raise ValueError("Strengthening settings must be a YAML mapping")
        unknown = set(supplied) - set(settings)
        if unknown:
            raise ValueError(f"Unknown strengthening settings: {sorted(unknown)}")
        settings.update(supplied)
    for key in (
        "bootstrap_resamples",
        "null_resamples",
        "local_grid_size",
        "minimum_kernel_ess",
        "matrix_bins",
        "min_per_group",
        "calibration_grid_size",
        "ece_bins",
        "examples_per_bin_group",
    ):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if min(settings["bootstrap_resamples"], settings["null_resamples"]) < 2:
        raise ValueError("At least two resamples are required")
    if min(settings["local_grid_size"], settings["calibration_grid_size"]) < 101:
        raise ValueError("Score grids must have at least 101 points")
    for key in ("rates", "matrix_rates", "bandwidths"):
        values = settings[key]
        if not values or len(set(values)) != len(values) or not all(0 < x < 1 for x in values):
            raise ValueError(f"{key} must contain distinct numbers in (0,1)")
    if not settings["bin_counts"] or any(int(x) != x or x < 2 for x in settings["bin_counts"]):
        raise ValueError("bin_counts must contain integers >= 2")
    if not 0 < settings["confidence_level"] < 1:
        raise ValueError("confidence_level must be in (0,1)")
    if not 0 < settings["minimum_valid_fraction"] <= 1 or settings["calibration_ridge"] <= 0:
        raise ValueError("Invalid valid-draw fraction or ridge")
    if not settings["null_models"] or set(settings["null_models"]) - {"platt", "spline"}:
        raise ValueError("null_models must be platt and/or spline")
    labels = settings["labels"]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("At least two distinct labels are required")
    for edge in settings["audit_edges"]:
        if len(edge) != 2 or edge[0] == edge[1] or not set(edge) <= set(labels):
            raise ValueError(f"Invalid audit edge: {edge}")
    if not settings["audit_edges"]:
        raise ValueError("audit_edges cannot be empty")
    return settings


def read_predictions(path: Path, labels: list[str]) -> tuple[pd.DataFrame, str]:
    frame = pd.read_parquet(path)
    required = ["example_id", *[f"{prefix}_{label}" for label in labels for prefix in ("p", "y")]]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame["example_id"].isna().any() or frame["example_id"].duplicated().any():
        raise ValueError(f"{path}: example_id must be non-null and unique")
    frame["example_id"] = frame["example_id"].astype(str)
    if frame["example_id"].duplicated().any():
        raise ValueError("example_id values collide after string conversion")
    frame = frame.sort_values("example_id", kind="stable").reset_index(drop=True)
    for label in labels:
        p, y = frame[f"p_{label}"].to_numpy(float), frame[f"y_{label}"].to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid probability column p_{label}: {path}")
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"y_{label} must be binary; soft targets are not used in this suite")
    if "text" in frame:
        if frame["text"].isna().any():
            raise ValueError("Null text values would corrupt grouped splitting")
        frame["audit_unit"] = frame["text"].astype(str).map(text_sha256)
        grouping = "sha256_of_exact_text"
    elif "text_sha256" in frame:
        if frame["text_sha256"].isna().any():
            raise ValueError("Null text_sha256")
        frame["audit_unit"] = frame["text_sha256"].astype(str)
        grouping = "provided_text_sha256"
    else:
        frame["audit_unit"] = frame["example_id"]
        grouping = "example_id_fallback_no_text_duplicate_protection"
    if len(frame) < 20:
        raise ValueError("Too few prediction rows for a held-out audit")
    return frame, grouping


def _sampler(units: np.ndarray):
    unique, inverse = np.unique(units, return_inverse=True)
    probabilities = np.full(len(unique), 1 / len(unique))
    return lambda rng: rng.multinomial(len(unique), probabilities)[inverse].astype(float)


def _add_intervals(rows: list[dict], samples: dict[str, np.ndarray], settings: dict) -> None:
    for metric, draws in samples.items():
        for j, row in enumerate(rows):
            summary = interval(
                draws[:, j], settings["confidence_level"], settings["minimum_valid_fraction"]
            )
            row.update({f"{metric}_{key}": value for key, value in summary.items()})


_CONTEXT: dict | None = None


def _init_worker(context: dict):
    global _CONTEXT
    _CONTEXT = context


def _run_job(job: tuple) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    if _CONTEXT is None:
        raise RuntimeError("Worker not initialized")
    stage, up_label, down_label, rate = job
    context = _CONTEXT
    up = context[f"p_{up_label}"]
    p, y = context[f"p_{down_label}"], context[f"y_{down_label}"]
    seed = derived_seed(context["seed"], context["name"], *job)
    rng = np.random.default_rng(seed)
    function = {
        "local": _local_job,
        "null": _null_job,
        "recalibration": _calibration_job,
        "matrix": _matrix_job,
    }[stage]
    tables, draws = function(up, p, y, rate, context, rng)
    for table in tables.values():
        table.insert(0, "prediction_set", context["name"])
        table.insert(1, "edge", f"{up_label}->{down_label}")
        table.insert(2, "rate", rate)
        table["model"] = context["model"]
        table["prompt_set"] = context["prompt_set"]
    return tables, draws


def _local_job(up, p, y, rate, context, rng):
    s = context["settings"]
    ones = np.ones(len(p))

    def compute(weights):
        route = routing_weights(up, weights, rate)
        return local_balance(
            p,
            y,
            route,
            weights - route,
            s["bandwidths"],
            s["local_grid_size"],
            s["minimum_kernel_ess"],
        )

    rows = compute(ones)
    metrics = ("signed_error", "score_gap", "absolute_local_score_gap", "common_support")
    draws = {key: np.full((s["bootstrap_resamples"], len(rows)), np.nan) for key in metrics}
    sample = _sampler(context["units"])
    for i in range(s["bootstrap_resamples"]):
        estimates = compute(sample(rng))
        for key in metrics:
            draws[key][i] = [row[key] for row in estimates]
    _add_intervals(rows, draws, s)
    for row in rows:
        row["uncertainty"] = "text_cluster_bootstrap_refit_routing_support_local_means"
    return {"local_balance": pd.DataFrame(rows)}, draws


def _heldout(up, context, rate):
    train = context["train"]
    policy = fit_policy(up[train], context["ties"][train], np.ones(train.sum()), rate)
    return train, policy, apply_policy(up, context["ties"], policy)


def _null_job(up, p, y, rate, context, rng):
    s = context["settings"]
    train, policy, route = _heldout(up, context, rate)
    pt, yt = p[train], y[train]
    pe, ye, re = p[~train], y[~train], route[~train]
    sample = _sampler(context["units"][train])
    coefficients, info = [], []
    for bins in s["bin_counts"]:
        c, metadata, _ = bin_coefficients(
            pe, re.astype(float), (~re).astype(float), bins, s["min_per_group"]
        )
        coefficients.append(c)
        info.append(metadata)
    coefficients = np.array(coefficients)
    rows, draw_outputs, fit_rows = [], {}, []
    alpha = (1 - s["confidence_level"]) / 2
    for kind in s["null_models"]:
        fit = fit_calibrator(
            pt,
            yt,
            np.ones(len(pt)),
            kind,
            ridge=s["calibration_ridge"],
            grid_size=s["calibration_grid_size"],
        )
        if not fit.success:
            raise RuntimeError(f"Observed null {kind} fit failed: gradient {fit.gradient_max}")
        q = fit.predict(pe)
        expected = coefficients @ (pe - q)
        draws = np.full((s["null_resamples"], len(coefficients)), np.nan)
        failed_fits = 0
        for b in range(s["null_resamples"]):
            bootfit = fit_calibrator(
                pt,
                yt,
                sample(rng),
                kind,
                ridge=s["calibration_ridge"],
                grid_size=s["calibration_grid_size"],
            )
            failed_fits += not bootfit.success
            if bootfit.success:
                y_star = rng.binomial(1, bootfit.predict(pe))
                draws[b] = coefficients @ (pe - y_star)
        draw_outputs[f"{kind}_predictive_signed_error"] = draws
        fit_rows.append(
            {
                "null_model": kind,
                "fit_success": fit.success,
                "gradient_max": fit.gradient_max,
                "bootstrap_fit_failures": failed_fits,
                "train_n": len(pt),
                "train_positive_n": int(yt.sum()),
                "eval_n": len(pe),
                "eval_positive_n": int(ye.sum()),
                "heldout_brier": float(np.mean((q - ye) ** 2)),
            }
        )
        for j, count in enumerate(s["bin_counts"]):
            finite = draws[np.isfinite(draws[:, j]), j]
            sufficient = len(finite) >= max(2, s["minimum_valid_fraction"] * len(draws))
            observed = float(coefficients[j] @ (pe - ye))
            centered = abs(finite - expected[j])
            tail = (
                (np.sum(centered >= abs(observed - expected[j])) + 1) / (len(finite) + 1)
                if sufficient and np.isfinite(observed)
                else np.nan
            )
            rows.append(
                {
                    "null_model": kind,
                    "n_bins": count,
                    **info[j],
                    "observed_signed_error": observed,
                    "fitted_null_expected_gap": float(expected[j]),
                    "observed_minus_fitted_null": observed - expected[j],
                    "predictive_low": float(np.quantile(finite, alpha)) if sufficient else np.nan,
                    "predictive_high": float(np.quantile(finite, 1 - alpha))
                    if sufficient
                    else np.nan,
                    "predictive_tail_fraction_NOT_pvalue": tail,
                    "valid_simulations": len(finite),
                    "prediction_band_supported": sufficient,
                    "eval_route_fraction": float(re.mean()),
                    "routing_threshold": policy[0],
                    "interpretation": "model_based_predictive_check_not_conditional_independence_test",
                }
            )
    return {
        "binning_null": pd.DataFrame(rows),
        "null_fit_diagnostics": pd.DataFrame(fit_rows),
    }, draw_outputs


def _calibration_job(up, p, y, rate, context, rng):
    s = context["settings"]
    train = context["train"]
    pt, yt, pe, ye = p[train], y[train], p[~train], y[~train]
    methods = ("raw", "platt", "spline", "spline_route")
    metric_keys = ("signed_error", "brier", "log_loss", "ece")
    sample_train, sample_eval = (
        _sampler(context["units"][train]),
        _sampler(context["units"][~train]),
    )

    def compute(wt, we, save_points=False):
        fitted_policy = fit_policy(up[train], context["ties"][train], wt, rate)
        rt = apply_policy(up[train], context["ties"][train], fitted_policy)
        re = apply_policy(up[~train], context["ties"][~train], fitted_policy)
        rows, points, diagnostics = [], [], []
        for kind in methods:
            success = True
            if kind == "raw":
                q = pe
            else:
                fit = fit_calibrator(
                    pt,
                    yt,
                    wt,
                    kind,
                    rt,
                    ridge=s["calibration_ridge"],
                    grid_size=s["calibration_grid_size"],
                )
                success = fit.success
                # Keep counts/policy diagnostics even for failed fits. Never
                # pass nonfinite fitted scores into histogram-based metrics.
                q = fit.predict(pe, re) if success else pe
                diagnostics.append(
                    {
                        "calibrator": kind,
                        "fit_success": success,
                        "gradient_max": fit.gradient_max,
                        "train_routed_n": float(wt @ rt),
                        "train_routed_positive_n": float(wt @ (rt * yt)),
                    }
                )
            metrics = calibration_metrics(q, ye, we, re, s["ece_bins"])
            for row in metrics:
                row["calibrator"] = kind
                row["eval_route_fraction"] = float(we @ re / we.sum())
                row["routing_threshold"] = fitted_policy[0]
                if not success:
                    for key in metric_keys:
                        row[key] = np.nan
                rows.append(row)
            if save_points and success:
                # Use shared RAW-score strata to display the SAME examples for
                # each calibrator. Empty groups are omitted, never filled with 0.
                boundaries = np.unique(np.quantile(pt, np.linspace(0, 1, 16)[1:-1]))
                bins = np.searchsorted(boundaries, pe, side="right")
                for group, mask in (
                    ("all", np.ones(len(pe), bool)),
                    ("routed", re),
                    ("non_routed", ~re),
                ):
                    for k in np.unique(bins):
                        chosen = mask & (bins == k)
                        n = int(chosen.sum())
                        if n:
                            points.append(
                                {
                                    "calibrator": kind,
                                    "group": group,
                                    "bin": int(k),
                                    "n": n,
                                    "positive_n": int(ye[chosen].sum()),
                                    "mean_probability": float(q[chosen].mean()),
                                    "outcome_rate": float(ye[chosen].mean()),
                                    "mean_raw_score": float(pe[chosen].mean()),
                                }
                            )
        return rows, points, diagnostics

    rows, points, diagnostics = compute(np.ones(len(pt)), np.ones(len(pe)), True)
    if any(not item["fit_success"] for item in diagnostics):
        raise RuntimeError("Observed held-out calibrator failed; inspect fitting configuration")
    draws = {key: np.full((s["bootstrap_resamples"], len(rows)), np.nan) for key in metric_keys}
    failure_counts = {kind: 0 for kind in methods[1:]}
    for b in range(s["bootstrap_resamples"]):
        bootrows, _, fits = compute(sample_train(rng), sample_eval(rng))
        for fit in fits:
            failure_counts[fit["calibrator"]] += not fit["fit_success"]
        for key in metric_keys:
            draws[key][b] = [row[key] for row in bootrows]
    _add_intervals(rows, draws, s)
    for row in rows:
        row["uncertainty"] = (
            "fixed_split_two_sample_text_cluster_bootstrap_refit_policy_and_calibrators"
        )
    comparisons = []
    for candidate, reference in (("platt", "raw"), ("spline", "raw"), ("spline_route", "spline")):
        for group in ("all", "routed", "non_routed"):
            i = next(
                i
                for i, row in enumerate(rows)
                if (row["calibrator"], row["group"]) == (candidate, group)
            )
            j = next(
                i
                for i, row in enumerate(rows)
                if (row["calibrator"], row["group"]) == (reference, group)
            )
            for key in metric_keys:
                differences = draws[key][:, i] - draws[key][:, j]
                comparisons.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "group": group,
                        "metric": key,
                        "paired_delta": rows[i][key] - rows[j][key],
                        **interval(differences, s["confidence_level"], s["minimum_valid_fraction"]),
                    }
                )
    for diagnostic in diagnostics:
        diagnostic["bootstrap_fit_failures"] = failure_counts[diagnostic["calibrator"]]
        diagnostic["train_n"] = len(pt)
        diagnostic["eval_n"] = len(pe)
    return {
        "heldout_calibration": pd.DataFrame(rows),
        "heldout_paired_comparisons": pd.DataFrame(comparisons),
        "heldout_reliability": pd.DataFrame(points),
        "calibration_fit_diagnostics": pd.DataFrame(diagnostics),
    }, draws


def _matrix_job(up, p, y, rate, context, rng):
    s = context["settings"]
    ones = np.ones(len(p))
    route = routing_weights(up, ones, rate)
    point = binned_metrics(p, y, route, ones - route, s["matrix_bins"], s["min_per_group"])
    sample = _sampler(context["units"])
    draws = {
        key: np.full((s["bootstrap_resamples"], 1), np.nan)
        for key in ("signed_error", "brier", "score_gap", "outcome_gap")
    }
    for b in range(s["bootstrap_resamples"]):
        w = sample(rng)
        r = routing_weights(up, w, rate)
        metrics = binned_metrics(p, y, r, w - r, s["matrix_bins"], s["min_per_group"])
        for key, values in draws.items():
            values[b, 0] = metrics[key]
    rows = [point]
    _add_intervals(rows, draws, s)
    for metric in ("signed_error", "brier"):
        point[f"p_{metric}"] = (
            bootstrap_sign_pvalue(draws[metric][:, 0])
            if point[f"{metric}_ci_supported"] and np.isfinite(point[metric])
            else np.nan
        )
    _, _, bins = bin_coefficients(p, route, ones - route, s["matrix_bins"], s["min_per_group"])
    bin_rows, examples = [], []
    for k in np.unique(bins):
        selected = bins == k
        nr, nc = int(route[selected].sum()), int((1 - route[selected]).sum())
        supported = min(nr, nc) >= s["min_per_group"]
        for group, mask in (("routed", route > 0), ("non_routed", route == 0)):
            indices = np.flatnonzero(selected & mask)
            if not len(indices):
                continue
            bin_rows.append(
                {
                    "bin": int(k),
                    "group": group,
                    "supported": supported,
                    "n": len(indices),
                    "positive_n": float(y[indices].sum()),
                    "mean_probability": float(p[indices].mean()),
                    "outcome_rate": float(y[indices].mean()),
                    "signed_error": float((p[indices] - y[indices]).mean()),
                    "score_min": float(p[indices].min()),
                    "score_max": float(p[indices].max()),
                }
            )
            if supported:
                chosen = rng.choice(
                    indices, min(len(indices), s["examples_per_bin_group"]), replace=False
                )
                for i in chosen:
                    examples.append(
                        {
                            "bin": int(k),
                            "group": group,
                            "example_id": context["ids"][i],
                            "p_upstream": up[i],
                            "p_downstream": p[i],
                            "label": y[i],
                        }
                    )
    if not examples:
        example_frame = pd.DataFrame(
            columns=["bin", "group", "example_id", "p_upstream", "p_downstream", "label"]
        )
    else:
        example_frame = pd.DataFrame(examples)
    return {
        "complete_matrix": pd.DataFrame(rows),
        "edge_bin_diagnostics": pd.DataFrame(bin_rows),
        "qualitative_sample_ids": example_frame,
    }, draws


def _job_key(job):
    stage, up, down, rate = job
    return f"{stage}__{up}__{down}__{rate:.6f}"


def _jobs(settings):
    jobs = []
    for stage in STAGES:
        edges = (
            itertools.permutations(settings["labels"], 2)
            if stage == "matrix"
            else settings["audit_edges"]
        )
        rates = settings["matrix_rates"] if stage == "matrix" else settings["rates"]
        jobs.extend((stage, up, down, float(rate)) for up, down in edges for rate in rates)
    return jobs


def _checkpoint_valid(directory: Path, fingerprint: str) -> bool:
    record = directory / "done.json"
    if not record.exists():
        return False
    try:
        metadata = json.loads(record.read_text())
        return metadata["fingerprint"] == fingerprint and all(
            (directory / name).exists() and sha256_file(directory / name) == digest
            for name, digest in metadata["files"].items()
        )
    except (ValueError, KeyError, OSError):
        return False


def _save_job(directory, fingerprint, tables, draws):
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, table in tables.items():
        path = directory / f"{name}.parquet"
        atomic_write_parquet(table, path)
        files[path.name] = sha256_file(path)
    draw_path = directory / "draws.npz"
    np.savez_compressed(draw_path, **draws)
    files[draw_path.name] = sha256_file(draw_path)
    atomic_write_json({"fingerprint": fingerprint, "files": files}, directory / "done.json")


def run_strengthening(
    config: dict[str, Any],
    *,
    settings_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    prediction_sets: list[str] | None = None,
    stages: list[str] | None = None,
    workers: int = 1,
    quick: bool = False,
) -> dict:
    settings = load_settings(settings_path)
    if quick:
        settings["bootstrap_resamples"] = min(8, settings["bootstrap_resamples"])
        settings["null_resamples"] = min(8, settings["null_resamples"])
    selected_stages = set(stages or STAGES)
    if selected_stages - set(STAGES):
        raise ValueError(f"Unknown stages: {selected_stages - set(STAGES)}")
    if workers < 1:
        raise ValueError("workers must be positive")
    definitions = config.get("robustness", {}).get("prediction_sets", []) or [
        {
            "name": "openjev_criteria_rich",
            "model": "OpenJev-27B",
            "prompt_set": "criteria_rich",
            "path": str(Path(config["paths"]["predictions_dir"]) / "civil_comments.parquet"),
        }
    ]
    names = [item["name"] for item in definitions]
    if len(names) != len(set(names)):
        raise ValueError("Prediction-set names must be unique")
    if prediction_sets:
        if set(prediction_sets) - set(names):
            raise ValueError(f"Unknown prediction sets; configured: {names}")
        definitions = [item for item in definitions if item["name"] in prediction_sets]
    for item in definitions:
        if not item["name"] or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in item["name"]
        ):
            raise ValueError("Prediction-set names must be safe simple directory names")
        if not Path(item["path"]).is_file():
            raise FileNotFoundError(
                f"Missing prediction set {item['name']}: {item['path']}. "
                "Update robustness.prediction_sets paths or explicitly select --prediction-set."
            )
    frames, inputs, reference = {}, {}, None
    seed = int(config["seed"])
    for item in definitions:
        path = Path(item["path"])
        frame, grouping = read_predictions(path, settings["labels"])
        comparison = frame[["example_id", "audit_unit", *[f"y_{x}" for x in settings["labels"]]]]
        if reference is not None and not np.array_equal(
            comparison.to_numpy(), reference.to_numpy()
        ):
            raise ValueError(
                "Prediction sets must have identical IDs, text units, and labels; no silent intersection"
            )
        reference = comparison
        frames[item["name"]] = frame
        inputs[item["name"]] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(frame),
            "grouping": grouping,
            "independent_units": int(frame["audit_unit"].nunique()),
            "model": item.get("model", item["name"]),
            "prompt_set": item.get("prompt_set", "unspecified"),
        }
        sidecar = path.with_suffix(".manifest.json")
        if sidecar.exists():
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            recorded = provenance.get("output_sha256")
            if recorded and recorded != inputs[item["name"]]["sha256"]:
                raise ValueError(f"Prediction/manifest checksum mismatch: {path}")
            inputs[item["name"]]["inference_manifest_sha256"] = sha256_file(sidecar)
            inputs[item["name"]]["inference_provenance"] = {
                key: provenance[key]
                for key in (
                    "model_revision",
                    "model_profile",
                    "request_model",
                    "prompt_set",
                    "prepared_sha256",
                    "request_signature",
                )
                if key in provenance
            }
    source_hashes = {path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")}
    identity = {
        "settings": settings,
        "inputs": inputs,
        "seed": seed,
        "quick_mode": quick,
        "code_sha256": source_hashes,
    }
    fingerprint = config_digest(identity)
    root = (
        Path(output_dir) if output_dir else Path(config["paths"]["results_dir"]) / "strengthening"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() and any(root.iterdir()):
        raise ValueError(
            f"{root} is nonempty but has no strengthening manifest. Use a new directory."
        )
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise ValueError(
                f"{root} contains another protocol/input/code version. Use a NEW --output-dir; "
                "existing results have not been overwritten."
            )
    # Exclusive lock prevents two processes from racing over checkpoints/reports.
    lock = root / "RUNNING.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"{lock} exists. Check the recorded PID; only remove a stale lock after "
            "confirming that no strengthening process is running."
        ) from exc
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"pid": os.getpid(), "host": platform.node()}, stream)
    manifest = {
        **identity,
        "fingerprint": fingerprint,
        "status": "running",
        "base_config_sha256": config_digest(config),
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pandas", "pyarrow", "matplotlib")
            },
        },
        "protocol_notes": [
            "Exploratory follow-up, not retroactively preregistered confirmatory evidence.",
            "50/50 exact-text-group split shared across models and edges; no evaluation-label fitting.",
            "Local and matrix bootstraps resample text clusters and reconstruct routing/support.",
            "Local common_grid is the intersection across bandwidths, recomputed in each resample.",
            "Held-out two-sample bootstrap refits policy/calibrators with fixed split membership.",
            "Null keeps evaluation scores, routing and bins fixed; refits pooled m on bootstrap training clusters.",
            "Null generates independent Bernoulli evaluation labels: model-based predictive bands, NOT exact p-values.",
            "Calibrator likelihood uses fixed logit-grid aggregation; prediction uses actual scores.",
            "Pointwise intervals, not simultaneous intervals; no BH applied to predictive tail fractions.",
            "No legacy continuous-spline covariance intervals are used.",
        ],
    }
    jobs = _jobs(settings)
    try:
        atomic_write_json(manifest, manifest_path)
        for definition in definitions:
            name = definition["name"]
            frame = frames[name]
            units = frame["audit_unit"].to_numpy(str)
            context = {
                "settings": settings,
                "seed": seed,
                "name": name,
                "model": definition.get("model", name),
                "prompt_set": definition.get("prompt_set", "unspecified"),
                "ids": frame["example_id"].to_numpy(str),
                "units": units,
                "train": split_by_unit(units, seed),
                "ties": tie_uniform(frame["example_id"].to_numpy(str), seed),
            }
            context.update(
                {
                    f"{prefix}_{label}": frame[f"{prefix}_{label}"].to_numpy(float)
                    for label in settings["labels"]
                    for prefix in ("p", "y")
                }
            )
            split_table = pd.DataFrame(
                {
                    "example_id": context["ids"],
                    "text_unit": units,
                    "split": np.where(context["train"], "calibration", "evaluation"),
                }
            )
            atomic_write_parquet(split_table, root / "splits" / f"{name}.parquet")
            pending = [
                job
                for job in jobs
                if job[0] in selected_stages
                and not _checkpoint_valid(root / "checkpoints" / name / _job_key(job), fingerprint)
            ]
            print(
                f"{name}: {len(pending)} pending jobs; {len(frame):,} rows; workers={workers}",
                flush=True,
            )
            if workers == 1:
                _init_worker(context)
                for job in tqdm(pending, desc=name, unit="job"):
                    tables, draws = _run_job(job)
                    _save_job(
                        root / "checkpoints" / name / _job_key(job), fingerprint, tables, draws
                    )
            elif pending:
                with ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_worker, initargs=(context,)
                ) as pool:
                    futures = {pool.submit(_run_job, job): job for job in pending}
                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc=name, unit="job"
                    ):
                        job = futures[future]
                        tables, draws = future.result()
                        _save_job(
                            root / "checkpoints" / name / _job_key(job), fingerprint, tables, draws
                        )
        collections: dict[str, list[pd.DataFrame]] = {}
        missing = []
        for definition in definitions:
            for job in jobs:
                directory = root / "checkpoints" / definition["name"] / _job_key(job)
                if not _checkpoint_valid(directory, fingerprint):
                    missing.append(f"{definition['name']}/{_job_key(job)}")
                    continue
                for path in directory.glob("*.parquet"):
                    collections.setdefault(path.stem, []).append(pd.read_parquet(path))
        tables_dir = root / "tables"
        tables_dir.mkdir(exist_ok=True)
        for name, pieces in collections.items():
            combined = pd.concat(pieces, ignore_index=True)
            if name == "complete_matrix":
                for metric in ("signed_error", "brier"):
                    combined[f"q_{metric}"] = np.nan
                    for indices in combined.groupby("prediction_set").groups.values():
                        # Only publish FDR for a COMPLETE planned family; unsupported cells have p=1.
                        if len(indices) == len(settings["labels"]) * (
                            len(settings["labels"]) - 1
                        ) * len(settings["matrix_rates"]):
                            combined.loc[indices, f"q_{metric}"] = benjamini_hochberg(
                                combined.loc[indices, f"p_{metric}"].fillna(1).to_numpy()
                            )
            atomic_write_parquet(combined, tables_dir / f"{name}.parquet")
            combined.to_csv(tables_dir / f"{name}.csv", index=False)
        manifest["status"] = "complete" if not missing else "partial"
        manifest["missing_jobs"] = missing
        manifest["table_rows"] = {
            name: sum(map(len, pieces)) for name, pieces in collections.items()
        }
        atomic_write_json(manifest, manifest_path)
        from .strengthen_reporting import generate_strengthening_report

        generate_strengthening_report(root)
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        atomic_write_json(manifest, manifest_path)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return {
        "output_dir": str(root),
        "status": manifest["status"],
        "missing_jobs": len(manifest["missing_jobs"]),
        "fingerprint": fingerprint,
    }
