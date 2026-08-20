import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.pvt import bg_from_pressure, build_pvt_table, compute_pb_from_rsi, standing_Bo
from src.material_balance import (
    compute_drive_indices,
    compute_ho_terms,
    fit_joint_drive_parameters,
    fit_zero_intercept,
    oil_and_dissolved_gas_expansion,
    rock_and_water_expansion,
)
from src.synthetic import generate_field_case

T = 200.0
API = 35.0
GAMMA_G = 0.75
RSI = 600.0
CF = 4e-6
CW = 3e-6
SWC = 0.2
CO = 1.5e-5
N_TRUE = 50_000_000.0


@pytest.fixture
def clean_depletion_case():
    """
    A self-contained, mass-balance-consistent depletion-only case
    (m=0, no aquifer): F = N_true * (Eo + Efw), Gp = Np*Rs, Wp = 0.
    Reservoir starts exactly at bubble point (Pi = Pb) so Eo = 0 at Pi.
    """
    Pi = compute_pb_from_rsi(RSI, T, API, GAMMA_G)
    Rsb = RSI
    Boi = standing_Bo(Rsb, T, API, GAMMA_G)
    Bgi = bg_from_pressure(Pi, T)

    pressures = list(np.linspace(Pi, 0.35 * Pi, 20))
    pvt_df = build_pvt_table(Pi, Pi, Rsb, Boi, T, API, GAMMA_G, CO, pressures)

    Eo = oil_and_dissolved_gas_expansion(pvt_df["Bo"], Boi, RSI, pvt_df["Rs"], pvt_df["Bg"])
    Efw = rock_and_water_expansion(Boi, CF, CW, SWC, Pi, pvt_df["P"])
    F = N_TRUE * (Eo + Efw)
    Np = F / pvt_df["Bo"]
    Gp = Np * pvt_df["Rs"]
    Wp = pd.Series(0.0, index=pvt_df.index)

    production_df = pd.DataFrame({"Np": Np, "Gp": Gp, "Wp": Wp, "P_avg": pvt_df["P"]})
    constants = dict(Pi=Pi, Boi=Boi, Bgi=Bgi, Rsi=RSI, cf=CF, cw=CW, Swc=SWC, m=0.0)

    ho_df = compute_ho_terms(production_df, pvt_df, constants)
    return production_df, pvt_df, ho_df


def test_f_positive_and_non_decreasing(clean_depletion_case):
    _, _, ho_df = clean_depletion_case
    chronological = ho_df.sort_values("P", ascending=False)
    assert (chronological["F"] >= 0).all()
    assert (chronological["F"].diff().dropna() >= -1e-6).all()


def test_eo_zero_at_initial_pressure(clean_depletion_case):
    _, _, ho_df = clean_depletion_case
    Pi = ho_df["P"].max()
    assert ho_df.loc[ho_df["P"] == Pi, "Eo"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_efw_zero_at_initial_pressure(clean_depletion_case):
    _, _, ho_df = clean_depletion_case
    Pi = ho_df["P"].max()
    assert ho_df.loc[ho_df["P"] == Pi, "Efw"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_free_gas_term_non_negative(clean_depletion_case):
    production_df, pvt_df, _ = clean_depletion_case
    pvt_sorted = pvt_df.sort_values("P")
    Rs = np.interp(production_df["P_avg"], pvt_sorted["P"], pvt_sorted["Rs"])
    free_gas = production_df["Gp"] - production_df["Np"] * Rs
    assert (free_gas >= -1e-6).all()


def test_fit_zero_intercept_recovers_n_true_on_clean_depletion(clean_depletion_case):
    _, _, ho_df = clean_depletion_case
    N_fit, R2 = fit_zero_intercept(ho_df["Eo"].values, ho_df["F"].values)
    assert abs(N_fit - N_TRUE) / N_TRUE < 0.02
    assert R2 > 0.90


def test_fit_zero_intercept_exact_on_et_by_construction(clean_depletion_case):
    """
    The fixture defines F = N_true * (Eo + Efw) = N_true * Et (m=0).
    Regressing F against Et is therefore an algebraic identity: it should
    recover N_true to floating-point precision. This isolates and confirms
    fit_zero_intercept's own correctness, separately from the small,
    expected Efw bias exercised by the F-vs-Eo depletion diagnostic above.
    """
    _, _, ho_df = clean_depletion_case
    Et = ho_df["Eo"] + ho_df["Efw"]
    N_fit, R2 = fit_zero_intercept(Et.values, ho_df["F"].values)
    assert abs(N_fit - N_TRUE) / N_TRUE < 1e-8
    assert R2 == pytest.approx(1.0, abs=1e-9)


def test_fit_zero_intercept_matches_dot_product_formula():
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 10, 50)
    y = 3.7 * x + rng.normal(0, 0.05, 50)
    N_fit, _ = fit_zero_intercept(x, y)
    assert N_fit == pytest.approx(np.dot(x, y) / np.dot(x, x))


def test_r2_never_exceeds_one():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.uniform(-10, 10, 30)
        y = rng.uniform(-10, 10, 30)
        _, R2 = fit_zero_intercept(x, y)
        assert R2 <= 1.0


def test_r2_high_for_good_depletion_fit(clean_depletion_case):
    _, _, ho_df = clean_depletion_case
    _, R2 = fit_zero_intercept(ho_df["Eo"].values, ho_df["F"].values)
    assert R2 > 0.90


DRIVE_INDEX_CASES = ["depletion", "gas_cap", "water_drive", "combination"]


def _joint_fit_indices_for(case):
    """Builds ho_df for a synthetic case (seed=42, matching the rest of the
    suite's convention) and returns (N_est, m_est, C_est, r2, indices_df)
    from the NNLS joint fit."""
    prod_df, pvt_df, tp = generate_field_case(case, 50_000_000.0, 30, True, seed=42)
    Pi = tp["Pi"]
    constants = dict(Pi=Pi, Boi=tp["Boi"], Bgi=tp["Bgi"], Rsi=RSI,
                      cf=CF, cw=CW, Swc=SWC, m=0.0)
    ho_df = compute_ho_terms(prod_df, pvt_df, constants)
    N_est, m_est, C_est, r2 = fit_joint_drive_parameters(ho_df, Pi)
    indices_df = compute_drive_indices(ho_df, N_est, m_est, C_est, Pi)
    return tp, N_est, m_est, C_est, r2, indices_df


@pytest.mark.parametrize("case", DRIVE_INDEX_CASES)
def test_drive_indices_are_non_negative(case):
    """NNLS constrains all three coefficients (N, Q=N*m, C) to >= 0, so DDI,
    SDI, and WDI can never be negative -- unlike the unconstrained
    np.linalg.lstsq joint fit this replaced, which could assign a negative
    coefficient to whichever of Eg / (Pi-P) was redundant given collinearity
    between them."""
    _, _, _, _, _, indices_df = _joint_fit_indices_for(case)
    valid = indices_df.dropna()
    assert (valid >= -1e-9).all().all()


@pytest.mark.parametrize("case", DRIVE_INDEX_CASES)
def test_drive_indices_sum_to_one_at_final_step(case):
    """DDI + SDI + WDI should sum close to 1.0 at the final (most depleted,
    least noise-sensitive) row -- this is what the app's summary metrics and
    interpretation text actually use. Early rows near F=0 have much larger
    *relative* deviation (a tiny absolute residual against a tiny F), which
    is expected numerical behavior, not a defect -- see
    compute_drive_indices' safe-division handling."""
    _, _, _, _, _, indices_df = _joint_fit_indices_for(case)
    final_row_sum = indices_df.dropna().iloc[-1].sum()
    assert abs(final_row_sum - 1.0) < 0.05


def test_joint_fit_n_recovery_for_depletion_and_gas_cap():
    """The NNLS joint fit should recover OOIP much more tightly for
    depletion and gas-cap cases (no aquifer to confuse with collinear Eg)
    than the >100% bias the Eo-only Section 3A fit shows for gas-cap data
    (see integration_matrix.md) or the unconstrained joint fit's negative-C
    /inflated-N failure mode this replaced."""
    tp, N_est, m_est, C_est, r2, _ = _joint_fit_indices_for("depletion")
    assert abs(N_est - tp["N_true"]) / tp["N_true"] < 0.05
    assert C_est == pytest.approx(0.0, abs=1.0)  # no real aquifer in this case

    tp, N_est, m_est, C_est, r2, _ = _joint_fit_indices_for("gas_cap")
    assert abs(N_est - tp["N_true"]) / tp["N_true"] < 0.15
    assert C_est == pytest.approx(0.0, abs=1.0)  # no real aquifer in this case
