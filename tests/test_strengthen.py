from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from jevcal.cli import build_parser
from jevcal.config import load_config
from jevcal.statistics import conditional_shift, pooled_quantile_bins, top_fraction_mask
from jevcal.strengthen import (
    DEFAULTS,
    _calibration_job,
    load_settings,
    read_predictions,
    run_strengthening,
)
from jevcal.strengthen_stats import (
    apply_policy,
    bin_coefficients,
    binned_metrics,
    fit_calibrator,
    fit_policy,
    interval,
    local_balance,
    routing_weights,
    split_by_unit,
    tie_uniform,
    weighted_bins,
)


def _frame(size=800):
    rng = np.random.default_rng(172)
    p = rng.uniform(0.05, 0.85, size)
    up = rng.uniform(0, 1, size)
    probability = np.clip(0.3 * p - 0.02 * (up > 0.8), 0.01, 0.95)
    return pd.DataFrame(
        {
            "example_id": [f"row-{i:06d}" for i in range(size)],
            "text": [f"text {i // 2}" for i in range(size)],
            "p_toxicity": up,
            "p_identity_attack": p,
            "y_toxicity": rng.binomial(1, up),
            "y_identity_attack": rng.binomial(1, probability),
        }
    )


def test_weighted_bins_and_routing_match_expanded_bootstrap():
    p = np.array([0.2, 0.1, 0.2, 0.7, 0.9, 0.8])
    up = np.array([0.5, 0.6, 0.6, 0.4, 0.3, 0.6])
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    w = np.array([2, 0, 3, 1, 2, 1], float)
    indices = np.repeat(np.arange(len(p)), w.astype(int))
    selected = top_fraction_mask(up[indices], 0.3, indices)
    actual = routing_weights(up, w, 0.3)
    expected = np.bincount(indices, weights=selected, minlength=len(p))
    np.testing.assert_array_equal(actual, expected)
    bins = weighted_bins(p, w, 3)
    np.testing.assert_array_equal(bins[indices], pooled_quantile_bins(p[indices], 3))
    metrics = binned_metrics(p, y, actual, w - actual, 3, 1)
    delta, support, used = conditional_shift(
        p[indices] - y[indices], selected, pooled_quantile_bins(p[indices], 3), 1
    )
    assert metrics["signed_error"] == pytest.approx(delta)
    assert metrics["common_support"] == pytest.approx(support)
    assert metrics["supported_bins"] == used
    assert metrics["signed_error"] == pytest.approx(metrics["score_gap"] - metrics["outcome_gap"])


def test_bin_coefficients_zero_under_constant_residual_and_no_support():
    p = np.linspace(0, 1, 100)
    r = (np.arange(100) % 2).astype(float)
    c, _, _ = bin_coefficients(p, r, 1 - r, 5, 3)
    assert c.sum() == pytest.approx(0, abs=1e-12)
    assert c @ np.ones(100) == pytest.approx(0, abs=1e-12)
    c, info, _ = bin_coefficients(p, r, 1 - r, 5, 100)
    assert np.isnan(c).all()
    assert info["common_support"] == 0


def test_local_balance_recovers_known_group_gap_and_common_support():
    p = np.repeat(np.linspace(0.05, 0.95, 601), 2)
    r = np.tile([0.0, 1.0], 601)
    # Identical group score distributions, controlled expected-label difference.
    y = 0.2 + 0.3 * p - 0.1 * r
    rows = local_balance(p, y, r, 1 - r, [0.01, 0.02, 0.05], 1001, 5)
    common_supports = []
    for row in rows:
        assert row["signed_error"] == pytest.approx(0.1, abs=1e-10)
        assert row["absolute_local_score_gap"] < 1e-12
        if row["support_scope"] == "common_grid":
            common_supports.append(row["common_support"])
    assert len(set(common_supports)) == 1


def test_fine_local_balance_reduces_known_coarsening_artifact():
    rng = np.random.default_rng(19)
    p = rng.uniform(0.05, 0.95, 30000)
    r = (rng.random(len(p)) < p).astype(float)
    y = 0.2 * p  # Exact invariance in conditional mean, despite miscalibration.
    broad = binned_metrics(p, y, r, 1 - r, 5, 10)["signed_error"]
    fine = local_balance(p, y, r, 1 - r, [0.005, 0.02], 1001, 10)
    narrow = next(
        row for row in fine if row["bandwidth"] == 0.005 and row["support_scope"] == "common_grid"
    )
    assert broad > 0.005
    assert abs(narrow["signed_error"]) < abs(broad) / 4


def test_duplicate_text_split_and_policy_boundary_ties():
    units = np.array(["a", "a", "b", "c", "d", "e", "f", "f"])
    train = split_by_unit(units, 17)
    assert train[0] == train[1] and train[-1] == train[-2]
    assert set(units[train]).isdisjoint(set(units[~train]))
    up = np.ones(20)
    ties = tie_uniform(np.array([str(i) for i in range(20)]), 18)
    policy = fit_policy(up, ties, np.ones(20), 0.2)
    assert apply_policy(up, ties, policy).sum() == 4
    np.testing.assert_array_equal(train, split_by_unit(units, 17))


def test_calibrator_is_finite_on_endpoint_probabilities_and_one_class():
    p = np.r_[np.zeros(20), np.linspace(0.01, 0.99, 100), np.ones(20)]
    for kind in ("platt", "spline", "spline_route"):
        r = np.arange(len(p)) % 2 == 0
        fit = fit_calibrator(p, np.zeros(len(p)), np.ones(len(p)), kind, r)
        assert fit.success
        q = fit.predict(p, r)
        assert np.isfinite(q).all()
        assert ((q >= 0) & (q <= 1)).all()


def test_bad_draw_coverage_suppresses_ci():
    result = interval(np.array([1.0, 2.0, np.nan, np.nan]), minimum_valid=0.95)
    assert not result["ci_supported"]
    assert np.isnan(result["ci_low"])


def test_validation_rejects_duplicate_ids_and_invalid_probabilities(tmp_path):
    frame = _frame()
    path = tmp_path / "predictions.parquet"
    frame.loc[1, "example_id"] = frame.loc[0, "example_id"]
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="unique"):
        read_predictions(path, ["toxicity", "identity_attack"])
    frame = _frame()
    frame.loc[1, "p_toxicity"] = np.nan
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="Invalid probability"):
        read_predictions(path, ["toxicity", "identity_attack"])


def test_heldout_predictions_do_not_fit_evaluation_labels():
    frame = _frame(500)
    settings = {**DEFAULTS, "bootstrap_resamples": 2, "calibration_grid_size": 257}
    units = frame.text.to_numpy()
    train = split_by_unit(units, 4)
    context = {
        "settings": settings,
        "train": train,
        "units": units,
        "ties": tie_uniform(frame.example_id.to_numpy(), 4),
    }
    y = frame.y_identity_attack.to_numpy(float)
    changed = y.copy()
    changed[~train] = 1 - changed[~train]
    tables, _ = _calibration_job(
        frame.p_toxicity.to_numpy(),
        frame.p_identity_attack.to_numpy(),
        y,
        0.2,
        context,
        np.random.default_rng(3),
    )
    other, _ = _calibration_job(
        frame.p_toxicity.to_numpy(),
        frame.p_identity_attack.to_numpy(),
        changed,
        0.2,
        context,
        np.random.default_rng(3),
    )
    np.testing.assert_allclose(
        tables["heldout_reliability"].mean_probability,
        other["heldout_reliability"].mean_probability,
    )
    for table in tables.values():
        assert len(table)


def _integration_config(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/reproduce.yaml")
    path = tmp_path / "predictions.parquet"
    _frame().to_parquet(path, index=False)
    config["robustness"]["prediction_sets"] = [
        {"name": "fixture", "model": "synthetic", "prompt_set": "synthetic", "path": str(path)}
    ]
    settings = {
        **DEFAULTS,
        "labels": ["toxicity", "identity_attack"],
        "audit_edges": [["toxicity", "identity_attack"]],
        "rates": [0.2],
        "matrix_rates": [0.2],
        "bootstrap_resamples": 3,
        "null_resamples": 3,
        "bin_counts": [5, 10],
        "bandwidths": [0.02, 0.04],
        "minimum_kernel_ess": 5,
        "min_per_group": 5,
        "calibration_grid_size": 257,
        "minimum_valid_fraction": 0.5,
    }
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(settings))
    return config, settings_path


def test_end_to_end_all_stages_resume_and_fingerprint_guard(tmp_path):
    config, settings_path = _integration_config(tmp_path)
    output = tmp_path / "outputs"
    result = run_strengthening(config, settings_path=settings_path, output_dir=output)
    assert result["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["table_rows"]["complete_matrix"] == 2
    assert manifest["table_rows"]["heldout_calibration"] == 12
    assert (output / "figures/fixture/local_balance.pdf").exists()
    assert (output / "figures/fixture/binning_null_r0.2.pdf").exists()
    assert (output / "figures/fixture/heldout_toxicity_to_identity_attack_r0.2.png").exists()
    assert (output / "publication_tables/fixture_binning_null.tex").exists()
    assert not (output / "RUNNING.lock").exists()
    done = next((output / "checkpoints").rglob("done.json"))
    timestamp = done.stat().st_mtime_ns
    repeat = run_strengthening(config, settings_path=settings_path, output_dir=output)
    assert repeat["fingerprint"] == result["fingerprint"]
    assert done.stat().st_mtime_ns == timestamp
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="NEW --output-dir"):
        run_strengthening(changed, settings_path=settings_path, output_dir=output)
    assert done.stat().st_mtime_ns == timestamp


def test_missing_prediction_sets_fail_before_writing_outputs(tmp_path):
    config, settings_path = _integration_config(tmp_path)
    config["robustness"]["prediction_sets"].append(
        {"name": "missing", "path": str(tmp_path / "missing.parquet")}
    )
    output = tmp_path / "outputs"
    with pytest.raises(FileNotFoundError, match="Missing prediction set"):
        run_strengthening(config, settings_path=settings_path, output_dir=output)
    assert not output.exists()


def test_parallel_matrix_matches_serial_and_partial_status(tmp_path):
    config, settings_path = _integration_config(tmp_path)
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    result = run_strengthening(
        config, settings_path=settings_path, output_dir=serial, stages=["matrix"], workers=1
    )
    other = run_strengthening(
        config, settings_path=settings_path, output_dir=parallel, stages=["matrix"], workers=2
    )
    assert result["status"] == other["status"] == "partial"
    pd.testing.assert_frame_equal(
        pd.read_parquet(serial / "tables/complete_matrix.parquet"),
        pd.read_parquet(parallel / "tables/complete_matrix.parquet"),
    )


def test_mismatched_run_ids_and_labels_rejected(tmp_path):
    config, settings_path = _integration_config(tmp_path)
    other = _frame()
    other.loc[0, "y_toxicity"] = 1 - other.loc[0, "y_toxicity"]
    path = tmp_path / "other.parquet"
    other.to_parquet(path, index=False)
    config["robustness"]["prediction_sets"].append({"name": "other", "path": str(path)})
    with pytest.raises(ValueError, match="identical IDs"):
        run_strengthening(config, settings_path=settings_path, output_dir=tmp_path / "out")


def test_settings_and_cli():
    settings = load_settings(Path(__file__).parents[1] / "configs/strengthen.yaml")
    assert settings["bootstrap_resamples"] == 2000
    args = build_parser().parse_args(["strengthen", "--workers", "8", "--stage", "null"])
    assert args.workers == 8 and args.stage == ["null"]


def test_plot_decimal_rate_names_are_not_overwritten(tmp_path):
    import matplotlib.pyplot as plt

    from jevcal.strengthen_reporting import _save

    for rate in (0.2, 0.1, 0.05, 0.02):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [rate, rate])
        _save(fig, tmp_path / f"rate{rate:g}")
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4
