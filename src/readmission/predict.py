"""Score new discharges with the trained model.

    python -m readmission.predict --input new_discharges.csv --output scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from readmission.features import CATEGORICAL, ENGINEERED, NUMERIC, add_features


def score(df: pd.DataFrame, model_path: Path = Path("models/readmission_model.joblib")) -> pd.DataFrame:
    model = joblib.load(model_path)
    X = add_features(df)[NUMERIC + ENGINEERED + CATEGORICAL]
    out = df.copy()
    out["readmission_risk"] = model.predict_proba(X)[:, 1]
    out["risk_tier"] = pd.cut(out["readmission_risk"], [0, 0.12, 0.25, 1.0], labels=["low", "medium", "high"],
                              include_lowest=True)
    return out.sort_values("readmission_risk", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/readmission_model.joblib"))
    args = parser.parse_args()
    scored = score(pd.read_csv(args.input), args.model)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored):,} discharges -> {args.output}")


if __name__ == "__main__":
    main()
