"""End-to-end training and evaluation pipeline.

Run from the repo root:
    python -m readmission.pipeline --data data/raw/discharges.csv --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from readmission.evaluate import (  # noqa: E402
    InterventionAssumptions,
    calibration_points,
    core_metrics,
    intervention_value,
    lift_table,
    subgroup_report,
)
from readmission.features import build_preprocessor, split_xy  # noqa: E402

SEED = 42


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("prep", build_preprocessor(scale=True)), ("clf", LogisticRegression(C=0.5, max_iter=2000))]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400, min_samples_leaf=25, max_features=0.4, n_jobs=-1, random_state=SEED
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=400,
                        max_leaf_nodes=15,
                        min_samples_leaf=60,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in candidate_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "average_precision", "neg_brier_score"])
        rows.append(
            {
                "model": name,
                "cv_roc_auc": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "cv_pr_auc": scores["test_average_precision"].mean(),
                "cv_brier": -scores["test_neg_brier_score"].mean(),
            }
        )
        print(f"  {name:<20} ROC AUC {rows[-1]['cv_roc_auc']:.3f}  PR AUC {rows[-1]['cv_pr_auc']:.3f}")
    return pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(y: np.ndarray, p_by_model: dict[str, np.ndarray], best: str, value: pd.DataFrame,
                 importance: pd.DataFrame, lift: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    # ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, p in p_by_model.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        lw = 2.4 if name == best else 1.2
        axes[0].plot(fpr, tpr, lw=lw, label=name.replace("_", " "))
        axes[1].plot(rec, prec, lw=lw, label=name.replace("_", " "))
    axes[0].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    axes[1].axhline(y.mean(), ls="--", c="grey", lw=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    axes[1].legend(frameon=False)
    _save(fig, fig_dir / "roc_pr_curves.png")

    # Calibration
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    mp, fp = calibration_points(y, p_by_model[best])
    ax.plot([0, 0.6], [0, 0.6], ls="--", c="grey", lw=1)
    ax.plot(mp, fp, marker="o", lw=2)
    ax.set(xlabel="Mean predicted risk", ylabel="Observed readmission rate", title="Calibration (final model)")
    _save(fig, fig_dir / "calibration.png")

    # Lift by decile
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(lift["decile"].astype(int), lift["observed_rate"], color="#3b6ea8")
    ax.axhline(y.mean(), ls="--", c="grey", lw=1, label=f"Overall rate {y.mean():.1%}")
    ax.set(xlabel="Predicted risk decile (1 = highest)", ylabel="Observed 30-day readmission rate",
           title="Readmission rate by risk decile")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False)
    _save(fig, fig_dir / "lift_by_decile.png")

    # Net savings curve
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(value["fraction_enrolled"], value["net_savings_per_1k_discharges"], marker="o", lw=2)
    best_row = value.loc[value["net_savings_per_1k_discharges"].idxmax()]
    ax.axvline(best_row["fraction_enrolled"], ls=":", c="grey")
    ax.axhline(0, c="black", lw=0.8)
    ax.set(xlabel="Share of discharges enrolled (highest risk first)", ylabel="Net savings per 1,000 discharges ($)",
           title="Program value vs. enrollment level")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    _save(fig, fig_dir / "net_savings_curve.png")

    # Permutation importance
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#3b6ea8")
    ax.set(xlabel="Drop in PR AUC when feature is shuffled", title="Permutation importance (test set)")
    _save(fig, fig_dir / "feature_importance.png")


def run(data_path: Path, out_dir: Path) -> dict:
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["readmitted_30d"], random_state=SEED)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    print("Cross-validating candidate models (5-fold)...")
    comparison = compare_models(X_train, y_train)
    best = comparison.loc[0, "model"]
    print(f"Selected: {best}")

    # Fit every candidate on the full training set so the test-set curves can be compared.
    models = candidate_models()
    p_by_model = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p_by_model[name] = model.predict_proba(X_test)[:, 1]

    # Recalibrate the selected model with isotonic regression on out-of-fold predictions.
    final = CalibratedClassifierCV(candidate_models()[best], method="isotonic", cv=5)
    final.fit(X_train, y_train)
    p_final = final.predict_proba(X_test)[:, 1]
    p_by_model[best] = p_final

    metrics = {"selected_model": best, "test": core_metrics(y_test.values, p_final)}
    lift = lift_table(y_test.values, p_final)
    assumptions = InterventionAssumptions()
    value = intervention_value(y_test.values, p_final, assumptions)
    best_row = value.loc[value["net_savings_per_1k_discharges"].idxmax()]
    metrics["business"] = {
        "assumptions": assumptions.__dict__,
        "top_decile_lift": float(lift.loc[0, "lift"]),
        "top_20pct_share_of_readmissions": float(lift.loc[1, "cum_share_of_readmissions"]),
        "best_enrollment_fraction": float(best_row["fraction_enrolled"]),
        "net_savings_per_1k_discharges_at_best": float(best_row["net_savings_per_1k_discharges"]),
    }

    # Subgroup checks — age band, sex, insurance.
    sub_df = test_df.assign(
        p=p_final,
        age_band=pd.cut(test_df["age"], [17, 44, 64, 79, 120], labels=["18-44", "45-64", "65-79", "80+"]),
    )
    subgroups = subgroup_report(sub_df, "readmitted_30d", "p", ["age_band", "sex", "insurance"])

    print("Computing permutation importance...")
    fitted_best = models[best]
    pi = permutation_importance(
        fitted_best, X_test, y_test, scoring="average_precision", n_repeats=8, random_state=SEED, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": X_test.columns, "importance": pi.importances_mean, "std": pi.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)
    comparison.to_csv(tables / "model_comparison.csv", index=False)
    lift.to_csv(tables / "lift_by_decile.csv", index=False)
    value.to_csv(tables / "intervention_value.csv", index=False)
    subgroups.to_csv(tables / "subgroup_performance.csv", index=False)
    importance.to_csv(tables / "permutation_importance.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    make_figures(y_test.values, p_by_model, best, value, importance, lift, out_dir / "figures")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(final, model_dir / "readmission_model.joblib")

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/discharges.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
