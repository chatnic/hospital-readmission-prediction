import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate  # noqa: E402
from readmission.evaluate import InterventionAssumptions, intervention_value, lift_table  # noqa: E402
from readmission.features import TARGET, add_features, split_xy  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return generate(n=3_000, seed=7)


def test_generator_is_reproducible():
    a, b = generate(500, seed=1), generate(500, seed=1)
    assert a.equals(b)


def test_generator_plausible_rate(df):
    assert 0.08 < df[TARGET].mean() < 0.25
    assert df["encounter_id"].is_unique


def test_no_target_leakage(df):
    X, y = split_xy(df)
    assert TARGET not in X.columns
    assert "encounter_id" not in X.columns
    assert len(X) == len(y)


def test_engineered_features(df):
    out = add_features(df)
    assert set(out["hba1c_missing"].unique()) <= {0, 1}
    assert (out["hba1c_missing"] == df["hba1c_last"].isna()).all()


def test_lift_table_sums(df):
    rng = np.random.default_rng(0)
    p = rng.random(len(df))
    lt = lift_table(df[TARGET].values, p)
    assert lt["patients"].sum() == len(df)
    assert lt["cum_share_of_readmissions"].iloc[-1] == pytest.approx(1.0)


def test_intervention_value_math():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.3])
    a = InterventionAssumptions(cost_per_patient=100, readmission_cost=1000, relative_risk_reduction=0.5)
    v = intervention_value(y, p, a, fractions=np.array([0.25, 1.0]))
    # Top 25% = 1 patient, who is readmitted: 0.5 * 1000 - 100 = 400
    assert v.loc[0, "net_savings"] == pytest.approx(400)
    # Everyone: 0.5 * 1000 - 400 = 100
    assert v.loc[1, "net_savings"] == pytest.approx(100)
