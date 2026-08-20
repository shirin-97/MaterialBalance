import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.pvt import (
    build_pvt_table,
    compute_pb_from_rsi,
    get_pvt_at_pressure,
    interpolate_pvt_for_pressures,
    standing_Bo,
    standing_Rs,
)

T = 200.0
API = 35.0
GAMMA_G = 0.75
RSI = 600.0
CO = 1.5e-5


@pytest.fixture
def bubble_point_params():
    Pb = compute_pb_from_rsi(RSI, T, API, GAMMA_G)
    Rsb = standing_Rs(Pb, T, API, GAMMA_G)
    Bob = standing_Bo(Rsb, T, API, GAMMA_G)
    return Pb, Rsb, Bob


def test_rs_locked_above_bubble_point(bubble_point_params):
    Pb, Rsb, Bob = bubble_point_params
    for P in [Pb, Pb + 200, Pb + 1000]:
        Rs, _, _, _ = get_pvt_at_pressure(P, Pb, Rsb, Bob, T, API, GAMMA_G, CO)
        assert Rs == pytest.approx(Rsb)


def test_bo_above_pb_follows_compressibility_trend(bubble_point_params):
    Pb, Rsb, Bob = bubble_point_params
    for P in [Pb, Pb + 250, Pb + 1000]:
        _, Bo, _, _ = get_pvt_at_pressure(P, Pb, Rsb, Bob, T, API, GAMMA_G, CO)
        assert Bo == pytest.approx(Bob * np.exp(CO * (Pb - P)))

    # Bo must decrease monotonically as pressure rises above Pb
    pressures_above = [Pb, Pb + 250, Pb + 1000]
    bo_values = [
        get_pvt_at_pressure(P, Pb, Rsb, Bob, T, API, GAMMA_G, CO)[1]
        for P in pressures_above
    ]
    assert all(bo_values[i] > bo_values[i + 1] for i in range(len(bo_values) - 1))


def test_rs_does_not_increase_below_bubble_point(bubble_point_params):
    Pb, Rsb, Bob = bubble_point_params
    pressures = list(np.linspace(Pb - 50, 500, 20))
    df = build_pvt_table(Pb + 300, Pb, Rsb, Bob, T, API, GAMMA_G, CO, pressures)
    below_pb = df[df["P"] < Pb].sort_values("P", ascending=False)
    assert below_pb["Rs"].is_monotonic_decreasing or below_pb["Rs"].diff().dropna().le(0).all()


def test_bo_greater_than_one_for_all_pressures(bubble_point_params):
    Pb, Rsb, Bob = bubble_point_params
    pressures = list(np.linspace(500, Pb + 500, 25))
    df = build_pvt_table(Pb + 500, Pb, Rsb, Bob, T, API, GAMMA_G, CO, pressures)
    assert (df["Bo"] > 1.0).all()


def test_bg_increases_monotonically_as_pressure_decreases(bubble_point_params):
    Pb, Rsb, Bob = bubble_point_params
    pressures = list(np.linspace(500, Pb + 300, 15))
    df = build_pvt_table(Pb + 300, Pb, Rsb, Bob, T, API, GAMMA_G, CO, pressures)
    df_desc = df.sort_values("P", ascending=False)
    assert df_desc["Bg"].is_monotonic_increasing


def test_build_pvt_table_columns_and_first_row(bubble_point_params):
    Pb, Rsb, Bob = bubble_point_params
    Pi = Pb + 300
    pressures = list(np.linspace(500, Pb + 200, 10))
    df = build_pvt_table(Pi, Pb, Rsb, Bob, T, API, GAMMA_G, CO, pressures)
    assert list(df.columns) == ["P", "Rs", "Bo", "Bg", "Bw"]
    assert df.iloc[0]["P"] == pytest.approx(Pi)


@pytest.fixture
def reference_pvt_table(bubble_point_params):
    Pb, Rsb, Bob = bubble_point_params
    Pi = Pb + 300
    pressures = list(np.linspace(500, Pi, 15))
    return build_pvt_table(Pi, Pb, Rsb, Bob, T, API, GAMMA_G, CO, pressures)


def test_interpolate_pvt_matches_table_at_exact_pressures(reference_pvt_table):
    result = interpolate_pvt_for_pressures(reference_pvt_table["P"].to_numpy(), reference_pvt_table)
    assert result["Rs"].to_numpy() == pytest.approx(reference_pvt_table["Rs"].to_numpy())
    assert result["Bo"].to_numpy() == pytest.approx(reference_pvt_table["Bo"].to_numpy())
    assert result["Bg"].to_numpy() == pytest.approx(reference_pvt_table["Bg"].to_numpy())


def test_interpolate_pvt_between_table_points_is_bounded(reference_pvt_table):
    p_sorted = reference_pvt_table.sort_values("P", ascending=False).reset_index(drop=True)
    midpoint = (p_sorted["P"].iloc[0] + p_sorted["P"].iloc[1]) / 2
    result = interpolate_pvt_for_pressures([midpoint], reference_pvt_table)
    lo, hi = sorted([p_sorted["Bo"].iloc[0], p_sorted["Bo"].iloc[1]])
    assert lo - 1e-9 <= result["Bo"].iloc[0] <= hi + 1e-9


def test_interpolate_pvt_unsorted_input_table_still_works(reference_pvt_table):
    shuffled = reference_pvt_table.sample(frac=1, random_state=0).reset_index(drop=True)
    result_shuffled = interpolate_pvt_for_pressures(reference_pvt_table["P"].to_numpy(), shuffled)
    result_sorted = interpolate_pvt_for_pressures(reference_pvt_table["P"].to_numpy(), reference_pvt_table)
    assert result_shuffled["Rs"].to_numpy() == pytest.approx(result_sorted["Rs"].to_numpy())
