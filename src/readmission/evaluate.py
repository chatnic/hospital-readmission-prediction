"""Model evaluation: discrimination, calibration, subgroup checks and business impact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class InterventionAssumptions:
    """Assumptions for the transitional-care program the model would target.

    These are illustrative planning numbers, not clinical evidence. Change them to match
    a specific hospital's costs and the program's measured effectiveness.
    """

    cost_per_patient: float = 500.0  # nurse follow-up call + early clinic visit
    readmission_cost: float = 15_000.0  # average cost of a 30-day readmission
    relative_risk_reduction: float = 0.20  # share of readmissions the program prevents


def core_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "base_rate": float(np.mean(y_true)),
    }


def lift_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Readmission rate by predicted-risk decile (decile 1 = highest risk)."""
    df = pd.DataFrame({"y": y_true, "p": p})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    base = df["y"].mean()
    out = (
        df.groupby("decile", observed=True)
        .agg(patients=("y", "size"), readmissions=("y", "sum"), mean_predicted=("p", "mean"))
        .assign(observed_rate=lambda t: t.readmissions / t.patients)
        .assign(lift=lambda t: t.observed_rate / base)
    )
    out["cum_share_of_readmissions"] = out["readmissions"].cumsum() / out["readmissions"].sum()
    return out.reset_index()


def intervention_value(
    y_true: np.ndarray,
    p: np.ndarray,
    assumptions: InterventionAssumptions = InterventionAssumptions(),
    fractions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Net savings from enrolling the top-k% highest-risk patients in the program."""
    if fractions is None:
        fractions = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    order = np.argsort(-p)
    y_sorted = np.asarray(y_true)[order]
    n = len(y_sorted)
    rows = []
    for f in fractions:
        k = int(round(f * n))
        caught = y_sorted[:k].sum()
        prevented = caught * assumptions.relative_risk_reduction
        savings = prevented * assumptions.readmission_cost
        cost = k * assumptions.cost_per_patient
        rows.append(
            {
                "fraction_enrolled": f,
                "patients_enrolled": k,
                "readmissions_in_group": int(caught),
                "readmissions_prevented": prevented,
                "program_cost": cost,
                "gross_savings": savings,
                "net_savings": savings - cost,
                "net_savings_per_1k_discharges": (savings - cost) / n * 1000,
            }
        )
    return pd.DataFrame(rows)


def subgroup_report(df: pd.DataFrame, y_col: str, p_col: str, group_cols: list[str]) -> pd.DataFrame:
    """Discrimination and calibration-in-the-large by subgroup."""
    rows = []
    for col in group_cols:
        for level, g in df.groupby(col, observed=True):
            if g[y_col].nunique() < 2 or len(g) < 100:
                continue
            rows.append(
                {
                    "attribute": col,
                    "group": level,
                    "n": len(g),
                    "observed_rate": g[y_col].mean(),
                    "mean_predicted": g[p_col].mean(),
                    "roc_auc": roc_auc_score(g[y_col], g[p_col]),
                }
            )
    out = pd.DataFrame(rows)
    out["calibration_gap"] = out["mean_predicted"] - out["observed_rate"]
    return out


def calibration_points(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=bins, strategy="quantile")
    return mean_pred, frac_pos
