"""Score-resolution and held-out diagnostics, independent of legacy spline CIs.

All fitting uses scipy/numpy. No model inference or network access is needed.
The local estimator is deliberately labelled finite-bandwidth, not an exact
conditional-independence test. Null simulations are model-based diagnostics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import convolve1d
from scipy.optimize import minimize
from scipy.special import expit, logit


def derived_seed(seed: int, *keys: object) -> int:
    payload = "|".join(map(str, (seed, *keys))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def routing_weights(up: np.ndarray, weights: np.ndarray, rate: float) -> np.ndarray:
    """Top ceil(rate * weighted N), stable input order breaking score ties.

    Integer bootstrap multiplicities may be split at the boundary, exactly as
    routing an expanded bootstrap sample containing repeated copies of a row.
    """
    order = np.argsort(-up, kind="stable")
    before = np.cumsum(weights[order]) - weights[order]
    selected = np.minimum(weights[order], np.maximum(0, np.ceil(rate * weights.sum()) - before))
    result = np.zeros_like(weights)
    result[order] = selected
    return result


def weighted_bins(p: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    """Linear quantiles of the expanded integer-weight sample; never split ties."""
    order = np.argsort(p, kind="stable")
    keep = weights[order] > 0
    values, cumulative = p[order][keep], np.cumsum(weights[order][keep])
    positions = np.linspace(0, cumulative[-1] - 1, count + 1)[1:-1]
    low = values[np.searchsorted(cumulative, np.floor(positions), side="right")]
    high = values[np.searchsorted(cumulative, np.ceil(positions), side="right")]
    edges = low + (high - low) * (positions - np.floor(positions))
    return np.searchsorted(np.unique(edges), p, side="right")


def bin_coefficients(
    p: np.ndarray, routed: np.ndarray, other: np.ndarray, count: int, minimum: int
) -> tuple[np.ndarray, dict, np.ndarray]:
    bins = weighted_bins(p, routed + other, count)
    nr = np.bincount(bins, weights=routed, minlength=count + 1)
    nc = np.bincount(bins, weights=other, minlength=count + 1)
    good = (nr >= minimum) & (nc >= minimum)
    support = nr[good].sum()
    if support == 0:
        coefficient = np.full(len(p), np.nan)
    else:
        ratio = np.divide(nr, nc, out=np.zeros_like(nr), where=nc > 0)
        coefficient = good[bins] * (routed - other * ratio[bins]) / support
    return (
        coefficient,
        {
            "routed_n": routed.sum(),
            "supported_routed_n": support,
            "common_support": support / routed.sum() if routed.sum() else 0,
            "supported_bins": int(good.sum()),
        },
        bins,
    )


def binned_metrics(
    p: np.ndarray, y: np.ndarray, routed: np.ndarray, other: np.ndarray, count: int, minimum: int
) -> dict:
    c, info, _ = bin_coefficients(p, routed, other, count, minimum)
    return {
        **info,
        "signed_error": float(c @ (p - y)),
        "brier": float(c @ (p * p - 2 * p * y + y)),
        "score_gap": float(c @ p),
        "outcome_gap": float(c @ y),
    }


def interval(draws: np.ndarray, confidence: float = 0.95, minimum_valid: float = 0.95) -> dict:
    values = np.asarray(draws, float)
    finite = values[np.isfinite(values)]
    valid = len(finite) >= max(2, np.ceil(minimum_valid * len(values)))
    alpha = (1 - confidence) / 2
    return {
        "ci_low": float(np.quantile(finite, alpha)) if valid else np.nan,
        "ci_high": float(np.quantile(finite, 1 - alpha)) if valid else np.nan,
        "bootstrap_valid": len(finite),
        "ci_supported": bool(valid),
    }


def local_balance(
    p: np.ndarray,
    y: np.ndarray,
    routed: np.ndarray,
    other: np.ndarray,
    bandwidths: list[float],
    grid_size: int,
    minimum_ess: int,
) -> list[dict]:
    """Gaussian local standardization to the SAME routed score-grid population.

    Group regressions use a truncated (4 sigma) Gaussian kernel. Native support
    requires both kernel ESS >= minimum_ess. Common-grid support intersects
    these sets over ALL specified bandwidths before comparing their estimates.
    Bootstrap callers refit routing, support, and all local regressions.
    """
    cell = np.rint(p * (grid_size - 1)).astype(int)
    hist = np.array(
        [
            np.bincount(cell, weights=w, minlength=grid_size)
            for w in (routed, other, routed * p, other * p, routed * y, other * y)
        ]
    )
    fits = []
    for h in bandwidths:
        sigma = h * (grid_size - 1)
        radius = int(np.ceil(4 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        totals = convolve1d(hist, kernel, axis=1, mode="constant", cval=0)
        squares = convolve1d(hist[:2], kernel**2, axis=1, mode="constant", cval=0)
        ess = np.divide(totals[:2] ** 2, squares, out=np.zeros_like(squares), where=squares > 0)
        means = np.divide(
            totals[2:],
            np.tile(totals[:2], (2, 1)),
            out=np.zeros_like(totals[2:]),
            where=np.tile(totals[:2], (2, 1)) > 0,
        )
        supported = (ess.min(axis=0) >= minimum_ess) & (hist[0] > 0)
        fits.append((h, means, ess, supported))
    common = np.logical_and.reduce([fit[3] for fit in fits])
    rows = []
    for h, means, ess, native in fits:
        for scope, mask in (("native", native), ("common_grid", common)):
            target = hist[0] * mask
            n = target.sum()
            w = target / n if n else target
            score_gap = means[0] - means[1]
            outcome_gap = means[2] - means[3]
            rows.append(
                {
                    "bandwidth": h,
                    "support_scope": scope,
                    "signed_error": float(w @ (score_gap - outcome_gap)) if n else np.nan,
                    "score_gap": float(w @ score_gap) if n else np.nan,
                    "absolute_local_score_gap": float(w @ abs(score_gap)) if n else np.nan,
                    "routed_prevalence": float(w @ means[2]) if n else np.nan,
                    "non_routed_prevalence": float(w @ means[3]) if n else np.nan,
                    "common_support": n / routed.sum() if routed.sum() else 0,
                    "supported_routed_n": n,
                    "supported_routed_positives": float(hist[4, mask].sum()),
                    "min_kernel_ess": float(ess[:, mask].min()) if n else np.nan,
                }
            )
    return rows


def split_by_unit(units: np.ndarray, seed: int) -> np.ndarray:
    unique, inverse = np.unique(units, return_inverse=True)
    if len(unique) < 4:
        raise ValueError("At least four independent text units are required")
    order = np.random.default_rng(derived_seed(seed, "heldout-split")).permutation(len(unique))
    train = np.zeros(len(unique), bool)
    train[order[: len(unique) // 2]] = True
    return train[inverse]


def tie_uniform(ids: np.ndarray, seed: int) -> np.ndarray:
    return np.array([derived_seed(seed, "policy-tie", value) / 2**64 for value in ids])


def fit_policy(
    up: np.ndarray, tie: np.ndarray, weights: np.ndarray, rate: float
) -> tuple[float, float]:
    order = np.lexsort((tie, -up))
    index = np.searchsorted(np.cumsum(weights[order]), np.ceil(rate * weights.sum()))
    chosen = order[index]
    return float(up[chosen]), float(tie[chosen])


def apply_policy(up: np.ndarray, tie: np.ndarray, policy: tuple[float, float]) -> np.ndarray:
    threshold, tie_cut = policy
    return (up > threshold) | ((up == threshold) & (tie <= tie_cut))


def logit_grid(p: np.ndarray, size: int = 2049) -> np.ndarray:
    z = np.clip(logit(np.clip(p, 1e-12, 1 - 1e-12)), -12, 12)
    return np.rint((z + 12) / 24 * (size - 1)).astype(int)


def calibration_design(p: np.ndarray, kind: str, route: np.ndarray | None = None) -> np.ndarray:
    z = np.clip(logit(np.clip(p, 1e-12, 1 - 1e-12)), -12, 12) / 6
    if kind == "platt":
        return np.column_stack((np.ones(len(p)), z))
    knots = np.r_[np.repeat(-2.0, 4), np.linspace(-2, 2, 9)[1:-1], np.repeat(2.0, 4)]
    basis = BSpline.design_matrix(z, knots, 3).toarray()
    if kind == "spline_route":
        if route is None:
            raise ValueError("Route-aware calibration requires routing indicators")
        basis = np.column_stack((basis, route, route * z))
    elif kind != "spline":
        raise ValueError(kind)
    return basis


@dataclass
class Calibrator:
    kind: str
    beta: np.ndarray
    success: bool
    gradient_max: float

    def predict(self, p: np.ndarray, route: np.ndarray | None = None) -> np.ndarray:
        return expit(calibration_design(p, self.kind, route) @ self.beta)


def fit_calibrator(
    p: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    kind: str,
    route: np.ndarray | None = None,
    ridge: float = 1.0,
    grid_size: int = 2049,
) -> Calibrator:
    """Penalized logistic fit, aggregated on a fixed logit grid for CPU efficiency.

    Fit uses cell-center logits; evaluation predicts at the actual input score.
    This approximation and clipping are recorded in the protocol, not hidden.
    No evaluation labels are accepted by the fitting API.
    """
    cell = logit_grid(p, grid_size)
    if kind == "spline_route":
        if route is None:
            raise ValueError("Missing route")
        cell = cell + grid_size * route.astype(int)
        size = 2 * grid_size
    else:
        size = grid_size
    counts = np.bincount(cell, weights=weights, minlength=size)
    successes = np.bincount(cell, weights=weights * y, minlength=size)
    grid_p = expit(np.linspace(-12, 12, grid_size))
    design = calibration_design(
        np.tile(grid_p, size // grid_size),
        kind,
        np.repeat([0.0, 1.0], grid_size) if size > grid_size else None,
    )
    active = counts > 0
    x, n, s = design[active], counts[active], successes[active]
    if n.sum() == 0:
        raise ValueError("Empty calibration training sample")
    dimension = x.shape[1]
    penalty = np.eye(dimension) * 1e-6
    if kind == "platt":
        penalty[1, 1] += ridge * 0.01
    else:
        base = dimension - (2 if kind == "spline_route" else 0)
        difference = np.diff(np.eye(base), n=2, axis=0)
        penalty[:base, :base] += ridge * difference.T @ difference
        if kind == "spline_route":
            penalty[-2:, -2:] += ridge * np.eye(2)
    initial = np.zeros(dimension)
    prevalence = np.clip((s.sum() + 0.5) / (n.sum() + 1), 1e-8, 1 - 1e-8)
    if kind == "platt":
        initial[0] = logit(prevalence)
    else:
        initial[: dimension - (2 if kind == "spline_route" else 0)] = logit(prevalence)

    def objective(beta):
        eta = x @ beta
        value = np.sum(n * np.logaddexp(0, eta) - s * eta) + 0.5 * beta @ penalty @ beta
        gradient = x.T @ (n * expit(eta) - s) + penalty @ beta
        return value, gradient

    def hessian(beta):
        q = expit(x @ beta)
        return x.T @ ((n * q * (1 - q))[:, None] * x) + penalty

    result = minimize(
        objective,
        initial,
        jac=True,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-5, "maxiter": 150},
    )
    gradient = float(np.max(np.abs(objective(result.x)[1])))
    success = bool(np.isfinite(result.x).all() and (result.success or gradient < 1e-3))
    return Calibrator(kind, result.x, success, gradient)


def calibration_metrics(
    q: np.ndarray, y: np.ndarray, weights: np.ndarray, route: np.ndarray, ece_bins: int = 15
) -> list[dict]:
    """ECE is fixed-width descriptive; Brier/log loss are proper held-out losses."""
    rows = []
    for group, mask in (("all", np.ones(len(y), bool)), ("routed", route), ("non_routed", ~route)):
        w = weights * mask
        total = w.sum()
        if total == 0:
            rows.append(
                {
                    "group": group,
                    "n": 0,
                    **{
                        key: np.nan
                        for key in (
                            "prevalence",
                            "mean_probability",
                            "signed_error",
                            "brier",
                            "log_loss",
                            "ece",
                            "positive_n",
                        )
                    },
                }
            )
            continue
        bins = np.minimum((q * ece_bins).astype(int), ece_bins - 1)
        residuals = np.bincount(bins, weights=w * (q - y), minlength=ece_bins)
        clipped = np.clip(q, 1e-12, 1 - 1e-12)
        rows.append(
            {
                "group": group,
                "n": total,
                "positive_n": float(w @ y),
                "prevalence": float(w @ y / total),
                "mean_probability": float(w @ q / total),
                "signed_error": float(w @ (q - y) / total),
                "brier": float(w @ ((q - y) ** 2) / total),
                "log_loss": float(
                    -w @ (y * np.log(clipped) + (1 - y) * np.log1p(-clipped)) / total
                ),
                "ece": float(abs(residuals).sum() / total),
            }
        )
    return rows
