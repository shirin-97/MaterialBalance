"""Mass-balance-corrected synthetic data generator (spec v4, Section 4.5).

Generates physically consistent synthetic field cases for a chosen drive
mechanism scenario. Water production is subtracted from total available
expansion energy before oil production is allocated, and noise is injected
via rates (not cumulatives) to avoid flatline/spike artifacts.
"""

import numpy as np
import pandas as pd

from .material_balance import (
    aquifer_constant_from_strength,
    compute_we,
    gas_cap_expansion,
    oil_and_dissolved_gas_expansion,
    rock_and_water_expansion,
)
from .pvt import bg_from_pressure, build_pvt_table, compute_pb_from_rsi, get_pvt_at_pressure, standing_Bo

# Representative fixed reservoir/PVT constants shared by every synthetic case.
# Pb is derived from Rsi via the algebraic inverse so Standing's Rs/Bo stay
# internally consistent. Pi and Pb are kept decoupled: Pi = Pb + pi_above_pb
# (per-case, see CASE_PARAMS) so the reservoir starts genuinely
# undersaturated, and the top of every decline curve exercises
# get_pvt_at_pressure's undersaturated (co-driven) branch before crossing
# into the saturated region below Pb.
T = 200.0
API = 35.0
GAMMA_G = 0.75
RSI = 600.0
CO = 1.5e-5
CF = 4e-6
CW = 3e-6
SWC = 0.20
BW = 1.03
PB = compute_pb_from_rsi(RSI, T, API, GAMMA_G)

# Abandonment pressure fraction of Pi (Pab = pab_fraction * Pi), and the
# undersaturated headroom above Pb (Pi = Pb + pi_above_pb), both per case:
# - pab_fraction: the spec's literal default of 0.35 (65% pressure
#   depletion) works well for the depletion case (deep drawdown minimizes
#   the small Efw bias in the F-vs-Eo regression test), but for the other
#   three cases it leaves solution-gas expansion (Eo) alone consuming >90%
#   of OOIP by abandonment for these PVT constants -- no headroom for a gas
#   cap or aquifer contribution without exceeding 100% recovery. Those
#   three cases use 0.5 instead, keeping depletion-only recovery around
#   ~45% and leaving realistic headroom.
# - pi_above_pb: a larger undersaturated segment more thoroughly exercises
#   get_pvt_at_pressure's co-driven branch, but it also grows the Efw bias
#   in the depletion case's mandated 2%-recovery test (Efw grows linearly
#   with (Pi-P) from the very top, while Eo grows slowly in that region).
#   Depletion uses a modest 100 psi offset to stay safely under the 2%
#   bound; the other three cases (no such accuracy requirement) use 500.
CASE_PARAMS = {
    "depletion":   dict(alpha=1.20, m_true=0.00, aquifer_strength="None",  water_fraction=0.08, pab_fraction=0.35, pi_above_pb=100.0),
    "gas_cap":     dict(alpha=0.85, m_true=0.25, aquifer_strength="None",  water_fraction=0.08, pab_fraction=0.50, pi_above_pb=500.0),
    "combination": dict(alpha=0.65, m_true=0.15, aquifer_strength="Weak",  water_fraction=0.70, pab_fraction=0.50, pi_above_pb=500.0),
    "water_drive": dict(alpha=0.35, m_true=0.00, aquifer_strength="Weak",  water_fraction=0.75, pab_fraction=0.50, pi_above_pb=500.0),
}


def pressure_decline(Pi, Pab, alpha, n_steps):
    """
    P(t) = Pi - (Pi - Pab) * t^alpha
    t runs from 0 to 1 linearly over n_steps.
    """
    t = np.linspace(0.0, 1.0, n_steps)
    return Pi - (Pi - Pab) * t ** alpha


def forward_mbe_corrected(N, Eo, Eg, Efw, We, m, Bo, Rs, Bw, water_fraction=0.08):
    """
    Compute Np, Gp, Wp at a single pressure step using corrected mass balance.

    Steps:
    1. Compute total available expansion energy (F_synthetic)
    2. Compute Wp first from water influx
    3. Subtract Wp energy cost from F_synthetic to get F_remaining
    4. Allocate F_remaining to oil (and dissolved gas, tracked via Rs)

    Parameters:
        water_fraction: fraction of water influx that is produced (default 0.08 = 8%)
    """

    # Step 1: Total available expansion energy from the reservoir
    F_synthetic = N * (Eo + m * Eg + Efw) + We

    # Step 2: Water production first — driven by water influx
    Wp = (We * water_fraction) / Bw

    # Step 3: Subtract water's energy cost from total available
    F_remaining = F_synthetic - (Wp * Bw)

    # Step 4: Remaining energy drives oil production
    # Assumes producing GOR = Rs (no free gas production above bubble point)
    # Gp = Np * Rs follows from this assumption
    Np = F_remaining / Bo
    Gp = Np * Rs

    return Np, Gp, Wp


def inject_noise_via_rates(Np_clean, Gp_clean, Wp_clean, P_clean, rng):
    """
    Correct noise injection pipeline.
    Noise is applied to incremental rates, not to cumulatives.
    This preserves physically realistic rate variation without flatline artifacts.
    """

    # Step 1: Convert cumulatives to incremental rates (delta per time step)
    oil_rates  = np.diff(Np_clean, prepend=0)   # STB per time step
    gas_rates  = np.diff(Gp_clean, prepend=0)   # Scf per time step
    water_rates = np.diff(Wp_clean, prepend=0)  # STB per time step

    # Step 2: Apply multiplicative noise to rates
    oil_rates_noisy   = oil_rates  * (1 + rng.normal(0, 0.03, len(oil_rates)))
    gas_rates_noisy   = gas_rates  * (1 + rng.normal(0, 0.04, len(gas_rates)))
    water_rates_noisy = water_rates * (1 + rng.normal(0, 0.05, len(water_rates)))

    # Step 3: Floor all rates at zero — rates cannot be negative
    # This guard is mandatory: without it, a large negative noise draw on a
    # small rate produces a negative rate, and cumsum gives a decreasing cumulative.
    oil_rates_noisy   = np.maximum(oil_rates_noisy,   0)
    gas_rates_noisy   = np.maximum(gas_rates_noisy,   0)
    water_rates_noisy = np.maximum(water_rates_noisy, 0)

    # Step 4: Rebuild cumulatives from noisy rates via cumsum
    # Monotonicity is guaranteed because all rates are >= 0
    Np_noisy = np.cumsum(oil_rates_noisy)
    Gp_noisy = np.cumsum(gas_rates_noisy)
    Wp_noisy = np.cumsum(water_rates_noisy)

    # Step 5: Apply additive noise to pressure (gauge measurement uncertainty)
    P_noisy = P_clean + rng.normal(0, 20, len(P_clean))   # +/- 20 psia

    return Np_noisy, Gp_noisy, Wp_noisy, P_noisy


def _drop_middle_rows(df, rng, n_drop=2, guard=3):
    """Randomly drops n_drop rows from the middle of df (simulated missing months);
    never touches the first/last `guard` rows."""
    droppable = np.arange(guard, len(df) - guard)
    if len(droppable) < n_drop:
        return df.reset_index(drop=True)
    drop_idx = rng.choice(droppable, size=n_drop, replace=False)
    return df.drop(index=drop_idx).reset_index(drop=True)


def generate_field_case(case, N, n_steps, noise, seed):
    """
    Generates a mass-balance-consistent synthetic field case.

    Parameters:
        case    : one of "depletion", "gas_cap", "water_drive", "combination"
        N       : true OOIP in STB
        n_steps : number of time steps (pressure survey points)
        noise   : whether to inject noise via inject_noise_via_rates
        seed    : random seed (drives np.random.default_rng(seed) exclusively)

    Returns:
        production_df, pvt_df, true_params
    """
    params = CASE_PARAMS[case]
    alpha = params["alpha"]
    m_true = params["m_true"]
    aquifer_strength = params["aquifer_strength"]
    water_fraction = params["water_fraction"]
    pab_fraction = params["pab_fraction"]
    pi_above_pb = params["pi_above_pb"]

    rng = np.random.default_rng(seed)

    Rsb = RSI
    Pb = PB
    Pi = Pb + pi_above_pb
    Bob = standing_Bo(Rsb, T, API, GAMMA_G)
    Bgi = bg_from_pressure(Pi, T)

    # Boi is Bo AT INITIAL PRESSURE Pi, which differs from Bob (Bo at Pb)
    # whenever the reservoir starts undersaturated (Pi > Pb).
    _, Boi, _, _ = get_pvt_at_pressure(Pi, Pb, Rsb, Bob, T, API, GAMMA_G, CO)

    Pab = pab_fraction * Pi
    P_clean = pressure_decline(Pi, Pab, alpha, n_steps)

    pvt_df = build_pvt_table(Pi, Pb, Rsb, Bob, T, API, GAMMA_G, CO, list(P_clean))

    Rs = pvt_df["Rs"].to_numpy()
    Bo = pvt_df["Bo"].to_numpy()
    Bg = pvt_df["Bg"].to_numpy()
    P_sorted = pvt_df["P"].to_numpy()

    Eo = oil_and_dissolved_gas_expansion(Bo, Boi, RSI, Rs, Bg)
    Eg = gas_cap_expansion(Boi, Bg, Bgi)
    Efw = rock_and_water_expansion(Boi, CF, CW, SWC, Pi, P_sorted)

    C = aquifer_constant_from_strength(aquifer_strength)
    We = compute_we(C, Pi, P_sorted)

    Np_clean, Gp_clean, Wp_clean = forward_mbe_corrected(
        N, Eo, Eg, Efw, We, m_true, Bo, Rs, BW, water_fraction=water_fraction
    )

    if noise:
        Np_final, Gp_final, Wp_final, P_final = inject_noise_via_rates(
            Np_clean, Gp_clean, Wp_clean, P_sorted, rng
        )
    else:
        Np_final, Gp_final, Wp_final, P_final = Np_clean, Gp_clean, Wp_clean, P_sorted

    oil_rates_noisy = np.diff(Np_final, prepend=0)
    gas_rates_noisy = np.diff(Gp_final, prepend=0)
    water_rates_noisy = np.diff(Wp_final, prepend=0)

    dates = pd.date_range(start="2020-01-01", periods=n_steps, freq="MS")

    production_df = pd.DataFrame({
        "Date": dates,
        "Np": Np_final,
        "Gp": Gp_final,
        "Wp": Wp_final,
        "P_avg": P_final,
        "oil_rates_noisy": oil_rates_noisy,
        "gas_rates_noisy": gas_rates_noisy,
        "water_rates_noisy": water_rates_noisy,
    })

    production_df = _drop_middle_rows(production_df, rng, n_drop=2, guard=3)

    true_params = dict(
        N_true=N, m_true=m_true, aquifer_strength=aquifer_strength, case=case,
        Pi=Pi, Pb=Pb, Boi=Boi, Bgi=Bgi,
    )

    return production_df, pvt_df, true_params
