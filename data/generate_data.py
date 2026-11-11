"""Generate a synthetic hospital discharge dataset for 30-day readmission modeling.

All records are fictional. The generator encodes clinically plausible relationships
(prior utilization, comorbidity burden, discharge disposition, follow-up care) so the
modeling exercise behaves like a real readmission problem, including class imbalance,
missing values and a few non-linear effects.

Usage:
    python data/generate_data.py --n 25000 --seed 42 --out data/raw/discharges.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIAGNOSES = {
    # name: (prevalence, log-odds contribution)
    "heart_failure": (0.14, 0.55),
    "copd": (0.10, 0.45),
    "pneumonia": (0.12, 0.25),
    "diabetes_complication": (0.09, 0.35),
    "sepsis": (0.08, 0.40),
    "hip_knee_replacement": (0.10, -0.60),
    "chest_pain": (0.12, -0.35),
    "other": (0.25, 0.0),
}

DISPOSITIONS = {
    "home": (0.58, 0.0),
    "home_health": (0.20, 0.20),
    "skilled_nursing": (0.17, 0.35),
    "left_against_advice": (0.05, 0.90),
}

INSURANCE = {
    "medicare": (0.45, 0.10),
    "medicaid": (0.18, 0.25),
    "commercial": (0.31, -0.15),
    "self_pay": (0.06, 0.20),
}


def _choice(rng: np.random.Generator, table: dict, n: int) -> np.ndarray:
    names = list(table)
    probs = np.array([table[k][0] for k in names])
    return rng.choice(names, size=n, p=probs / probs.sum())


def generate(n: int = 25_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(64, 16, n), 18, 100).round().astype(int)
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    diagnosis = _choice(rng, DIAGNOSES, n)
    disposition = _choice(rng, DISPOSITIONS, n)
    insurance = _choice(rng, INSURANCE, n)
    # Older patients skew Medicare.
    insurance = np.where((age >= 66) & (rng.random(n) < 0.7), "medicare", insurance)

    admission_type = rng.choice(["emergency", "urgent", "elective"], size=n, p=[0.62, 0.23, 0.15])
    admission_type = np.where(diagnosis == "hip_knee_replacement", "elective", admission_type)

    charlson = rng.poisson(lam=np.clip((age - 30) / 18, 0.2, None))
    prior_admits = rng.poisson(lam=0.3 + 0.25 * charlson)
    ed_visits = rng.poisson(lam=0.5 + 0.3 * prior_admits)
    los = np.clip(rng.gamma(shape=2.0, scale=2.2, size=n) + 0.4 * charlson, 1, 45).round(1)
    n_meds = np.clip(rng.poisson(lam=4 + 1.6 * charlson), 0, 40)
    abnormal_labs = rng.poisson(lam=1 + 0.4 * charlson)
    lives_alone = rng.random(n) < np.where(age > 75, 0.38, 0.22)
    followup_scheduled = rng.random(n) < 0.63
    hba1c = np.where(
        diagnosis == "diabetes_complication",
        rng.normal(8.9, 1.5, n),
        rng.normal(6.0, 1.1, n),
    ).round(1)

    # --- outcome model (log-odds) ---
    logit = -3.1
    logit += np.vectorize(lambda d: DIAGNOSES[d][1])(diagnosis)
    logit += np.vectorize(lambda d: DISPOSITIONS[d][1])(disposition)
    logit += np.vectorize(lambda d: INSURANCE[d][1])(insurance)
    logit += 0.38 * np.log1p(prior_admits) * 2
    logit += 0.12 * np.minimum(ed_visits, 6)
    logit += 0.11 * charlson
    logit += 0.035 * np.minimum(n_meds, 20) - 0.1
    logit += 0.06 * abnormal_labs
    # U-shaped length-of-stay effect: very short and very long stays are riskier.
    logit += 0.012 * (los - 5) ** 2 / 4 - 0.05
    logit += 0.012 * np.maximum(age - 70, 0)
    logit += 0.25 * lives_alone
    # Follow-up matters more for high-burden patients (interaction).
    logit -= np.where(followup_scheduled, 0.25 + 0.08 * np.minimum(charlson, 5), 0)
    logit += np.where(hba1c > 9, 0.3, 0)
    logit += rng.normal(0, 0.35, n)  # unobserved heterogeneity

    p = 1 / (1 + np.exp(-logit))
    readmitted = rng.random(n) < p

    df = pd.DataFrame(
        {
            "encounter_id": [f"E{100000 + i}" for i in range(n)],
            "age": age,
            "sex": sex,
            "admission_type": admission_type,
            "primary_diagnosis": diagnosis,
            "length_of_stay_days": los,
            "charlson_index": charlson,
            "prior_admissions_12m": prior_admits,
            "ed_visits_6m": ed_visits,
            "num_medications": n_meds,
            "abnormal_lab_count": abnormal_labs,
            "hba1c_last": hba1c,
            "discharge_disposition": disposition,
            "insurance": insurance,
            "lives_alone": lives_alone.astype(int),
            "followup_scheduled": followup_scheduled.astype(int),
            "readmitted_30d": readmitted.astype(int),
        }
    )

    # Realistic missingness: HbA1c often not drawn; social history sometimes not recorded.
    df.loc[rng.random(n) < 0.35, "hba1c_last"] = np.nan
    df.loc[rng.random(n) < 0.08, "lives_alone"] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw/discharges.csv"))
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} (readmission rate {df.readmitted_30d.mean():.1%})")


if __name__ == "__main__":
    main()
