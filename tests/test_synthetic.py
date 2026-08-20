import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.material_balance import compute_ho_terms, fit_zero_intercept
from src.synthetic import BW, CASE_PARAMS, CF, CW, RSI, SWC, generate_field_case

N_TRUE = 50_000_000.0
CASES = list(CASE_PARAMS)


def _constants_for(true_params):
    # Pi/Boi/Bgi differ per case (undersaturated headroom above Pb varies
    # by case — see CASE_PARAMS), so they come from true_params, not a
    # single global constant.
    return dict(
        Pi=true_params["Pi"], Boi=true_params["Boi"], Bgi=true_params["Bgi"],
        Rsi=RSI, cf=CF, cw=CW, Swc=SWC, m=true_params["m_true"],
    )


@pytest.mark.parametrize("case", CASES)
def test_np_monotonically_non_decreasing(case):
    prod_df, _, _ = generate_field_case(case, N_TRUE, 30, True, seed=42)
    Np = prod_df["Np"].to_numpy()
    assert np.all(np.diff(Np) >= -1e-6)


@pytest.mark.parametrize("case", CASES)
def test_gp_monotonically_non_decreasing(case):
    prod_df, _, _ = generate_field_case(case, N_TRUE, 30, True, seed=42)
    Gp = prod_df["Gp"].to_numpy()
    assert np.all(np.diff(Gp) >= -1e-6)


@pytest.mark.parametrize("case", CASES)
def test_pressure_monotonically_decreasing(case):
    # Noise-free: additive pressure noise (gauge uncertainty) is allowed to
    # break strict monotonicity by design and is not what this check targets.
    prod_df, _, _ = generate_field_case(case, N_TRUE, 30, False, seed=42)
    P = prod_df["P_avg"].to_numpy()
    assert np.all(np.diff(P) <= 1e-6)


@pytest.mark.parametrize("case", CASES)
def test_f_remaining_positive_at_every_step(case):
    prod_df, pvt_df, true_params = generate_field_case(case, N_TRUE, 30, False, seed=42)
    ho_df = compute_ho_terms(prod_df, pvt_df, _constants_for(true_params))
    F_synthetic = ho_df["F"].to_numpy()
    Wp = prod_df["Wp"].to_numpy()
    F_remaining = F_synthetic - Wp * BW
    assert np.all(F_remaining >= -1e-6)


@pytest.mark.parametrize("case", CASES)
def test_wp_times_bw_below_f_synthetic(case):
    prod_df, pvt_df, true_params = generate_field_case(case, N_TRUE, 30, False, seed=42)
    ho_df = compute_ho_terms(prod_df, pvt_df, _constants_for(true_params))
    F_synthetic = ho_df["F"].to_numpy()
    Wp = prod_df["Wp"].to_numpy()
    assert np.all(Wp * BW < F_synthetic + 1e-6)


@pytest.mark.parametrize("case", CASES)
def test_all_case_types_generate_without_error(case):
    prod_df, pvt_df, true_params = generate_field_case(case, N_TRUE, 30, True, seed=42)
    assert len(prod_df) > 0
    assert list(prod_df.columns) == [
        "Date", "Np", "Gp", "Wp", "P_avg",
        "oil_rates_noisy", "gas_rates_noisy", "water_rates_noisy",
    ]
    assert true_params["case"] == case
    assert true_params["N_true"] == N_TRUE


def test_ho_analysis_recovers_n_true_for_depletion_case():
    prod_df, pvt_df, true_params = generate_field_case("depletion", N_TRUE, 30, False, seed=42)
    ho_df = compute_ho_terms(prod_df, pvt_df, _constants_for(true_params))
    N_fit, R2 = fit_zero_intercept(ho_df["Eo"].values, ho_df["F"].values)
    assert abs(N_fit - N_TRUE) / N_TRUE < 0.02
    assert R2 > 0.90
