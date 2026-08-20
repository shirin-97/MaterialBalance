import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.validation import sanitize_mapped_data


@pytest.fixture
def raw_production_df():
    return pd.DataFrame({
        "Date_col": ["2020-01-01", "2020-02-01", "2020-03-01", "bad-date"],
        "Oil Cum (stb)": ["500", "1000", "not_a_number", "3000"],
        "Gas Cum": [250_000, 500_000, 1_000_000, 1_500_000],
        "Water Cum": [5, 10, 20, 30],
        "Pressure (psi)": ["3000", "2900", "2800", None],
    })


@pytest.fixture
def raw_pvt_df():
    # Deliberately unsorted (ascending) to verify PVT sorting.
    return pd.DataFrame({
        "Pressure": [2600, 2800, 3000],
        "GOR": [512, 558, 600],
        "Oil FVF": [1.258, 1.285, 1.310],
        "Gas FVF": [0.001040, 0.000960, 0.000892],
        "Water FVF": [1.030, 1.030, 1.030],
    })


PRODUCTION_MAPPING = {
    "Date": "Date_col", "Np": "Oil Cum (stb)", "Gp": "Gas Cum",
    "Wp": "Water Cum", "P_avg": "Pressure (psi)",
}
PVT_MAPPING = {
    "P": "Pressure", "Rs": "GOR", "Bo": "Oil FVF", "Bg": "Gas FVF", "Bw": "Water FVF",
}


def test_sanitize_renames_to_standard_columns(raw_production_df, raw_pvt_df):
    production_df, pvt_df = sanitize_mapped_data(
        raw_production_df, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING
    )
    assert list(production_df.columns) == ["Date", "Np", "Gp", "Wp", "P_avg"]
    assert list(pvt_df.columns) == ["P", "Rs", "Bo", "Bg", "Bw"]


def test_sanitize_coerces_numeric_and_drops_invalid_np_rows(raw_production_df, raw_pvt_df):
    production_df, pvt_df = sanitize_mapped_data(
        raw_production_df, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING
    )
    # Row 2 has Np="not_a_number" -> NaN -> dropped. Row 3 has P_avg=None -> dropped.
    assert len(production_df) == 2
    assert production_df["Np"].dtype.kind == "f"
    assert not production_df["Np"].isna().any()
    assert not production_df["P_avg"].isna().any()


def test_sanitize_parses_date_and_flags_bad_dates(raw_production_df, raw_pvt_df):
    production_df, _ = sanitize_mapped_data(
        raw_production_df, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING
    )
    assert pd.api.types.is_datetime64_any_dtype(production_df["Date"])


def test_sanitize_sorts_pvt_descending_by_pressure(raw_production_df, raw_pvt_df):
    _, pvt_df = sanitize_mapped_data(
        raw_production_df, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING
    )
    assert list(pvt_df["P"]) == sorted(pvt_df["P"], reverse=True)


def test_sanitize_handles_already_sorted_pvt(raw_production_df):
    already_sorted = pd.DataFrame({
        "Pressure": [3000, 2800, 2600],
        "GOR": [600, 558, 512],
        "Oil FVF": [1.310, 1.285, 1.258],
        "Gas FVF": [0.000892, 0.000960, 0.001040],
        "Water FVF": [1.030, 1.030, 1.030],
    })
    _, pvt_df = sanitize_mapped_data(
        raw_production_df, already_sorted, PRODUCTION_MAPPING, PVT_MAPPING
    )
    assert list(pvt_df["P"]) == [3000, 2800, 2600]


def test_sanitize_drops_zero_np_and_zero_pressure_rows(raw_pvt_df):
    raw = pd.DataFrame({
        "Date_col": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "Oil Cum (stb)": [0, 1000, 2000],  # row 0: shut-in / not-yet-producing
        "Gas Cum": [0, 500_000, 1_000_000],
        "Water Cum": [0, 10, 20],
        "Pressure (psi)": [3000, 0, 2800],  # row 1: bad/missing pressure sentinel
    })
    production_df, _ = sanitize_mapped_data(raw, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING)
    assert len(production_df) == 1
    assert production_df["Np"].iloc[0] == 2000


def test_sanitize_sorts_production_by_date_ascending(raw_pvt_df):
    raw = pd.DataFrame({
        "Date_col": ["2020-03-01", "2020-01-01", "2020-02-01"],
        "Oil Cum (stb)": [3000, 1000, 2000],
        "Gas Cum": [3_000_000, 1_000_000, 2_000_000],
        "Water Cum": [30, 10, 20],
        "Pressure (psi)": [2800, 3000, 2900],
    })
    production_df, _ = sanitize_mapped_data(raw, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING)
    assert list(production_df["Np"]) == [1000, 2000, 3000]
    assert production_df["Date"].is_monotonic_increasing


def test_sanitize_cumsum_when_incremental(raw_pvt_df):
    # Daily incremental rates, not cumulative -- zigzags rather than only increasing.
    raw = pd.DataFrame({
        "Date_col": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Oil Cum (stb)": [100, 50, 80],
        "Gas Cum": [1000, 500, 800],
        "Water Cum": [1, 2, 3],
        "Pressure (psi)": [3000, 2950, 2900],
    })
    production_df, _ = sanitize_mapped_data(
        raw, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING, is_incremental=True
    )
    assert list(production_df["Np"]) == [100, 150, 230]
    assert list(production_df["Gp"]) == [1000, 1500, 2300]
    assert production_df["Np"].is_monotonic_increasing


def test_sanitize_no_cumsum_when_not_incremental(raw_pvt_df):
    raw = pd.DataFrame({
        "Date_col": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Oil Cum (stb)": [100, 150, 230],  # already cumulative
        "Gas Cum": [1000, 1500, 2300],
        "Water Cum": [1, 2, 3],
        "Pressure (psi)": [3000, 2950, 2900],
    })
    production_df, _ = sanitize_mapped_data(
        raw, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING, is_incremental=False
    )
    assert list(production_df["Np"]) == [100, 150, 230]


@pytest.fixture
def raw_production_df_metric():
    return pd.DataFrame({
        "Date_col": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Oil (Sm3)": [100.0, 150.0, 230.0],
        "Gas (Sm3)": [1000.0, 1500.0, 2300.0],
        "Water (Sm3)": [1.0, 2.0, 3.0],
        "Pressure (Bar)": [206.8, 200.0, 193.0],
    })


METRIC_PRODUCTION_MAPPING = {
    "Date": "Date_col", "Np": "Oil (Sm3)", "Gp": "Gas (Sm3)",
    "Wp": "Water (Sm3)", "P_avg": "Pressure (Bar)",
}


def test_sanitize_converts_metric_to_imperial(raw_production_df_metric, raw_pvt_df):
    production_df, _ = sanitize_mapped_data(
        raw_production_df_metric, raw_pvt_df, METRIC_PRODUCTION_MAPPING, PVT_MAPPING,
        is_metric=True,
    )
    assert production_df["Np"].to_numpy() == pytest.approx([100.0 * 6.2898, 150.0 * 6.2898, 230.0 * 6.2898])
    assert production_df["Wp"].to_numpy() == pytest.approx([1.0 * 6.2898, 2.0 * 6.2898, 3.0 * 6.2898])
    assert production_df["Gp"].to_numpy() == pytest.approx([1000.0 * 35.315, 1500.0 * 35.315, 2300.0 * 35.315])
    assert production_df["P_avg"].to_numpy() == pytest.approx(
        [206.8 * 14.5038, 200.0 * 14.5038, 193.0 * 14.5038]
    )
    # Converted Bar->psia should land in the same realistic reservoir range
    # as the (already-Imperial) PVT template, e.g. roughly 1000-4000 psia.
    assert (production_df["P_avg"] > 1000).all() and (production_df["P_avg"] < 4000).all()


def test_sanitize_no_conversion_when_not_metric(raw_production_df_metric, raw_pvt_df):
    production_df, _ = sanitize_mapped_data(
        raw_production_df_metric, raw_pvt_df, METRIC_PRODUCTION_MAPPING, PVT_MAPPING,
        is_metric=False,
    )
    assert list(production_df["Np"]) == [100.0, 150.0, 230.0]
    assert list(production_df["P_avg"]) == [206.8, 200.0, 193.0]


def test_sanitize_metric_conversion_leaves_pvt_table_untouched(raw_production_df_metric, raw_pvt_df):
    _, pvt_df = sanitize_mapped_data(
        raw_production_df_metric, raw_pvt_df, METRIC_PRODUCTION_MAPPING, PVT_MAPPING,
        is_metric=True,
    )
    assert list(pvt_df["P"]) == [3000, 2800, 2600]
    assert list(pvt_df["Rs"]) == [600, 558, 512]


@pytest.fixture
def noisy_daily_production_df():
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    trend = 3000 - np.linspace(0, 500, n)
    pressure = trend + rng.normal(0, 80, n)
    pressure[10:13] = 0  # simulated shut-in days
    return pd.DataFrame({
        "Date_col": dates,
        "Oil Cum (stb)": rng.uniform(50, 150, n),
        "Gas Cum": rng.uniform(500, 1500, n),
        "Water Cum": rng.uniform(1, 5, n),
        "Pressure (psi)": pressure,
    })


def test_rolling_pressure_smoothing_reduces_day_to_day_noise(noisy_daily_production_df, raw_pvt_df):
    raw_std = np.std(np.diff(noisy_daily_production_df["Pressure (psi)"].to_numpy()))
    production_df, _ = sanitize_mapped_data(
        noisy_daily_production_df, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING,
        is_incremental=True,
    )
    smoothed_std = np.std(np.diff(production_df["P_avg"].to_numpy()))
    assert smoothed_std < raw_std / 5


def test_rolling_pressure_smoothing_fills_shutin_zeros_no_rows_dropped(
    noisy_daily_production_df, raw_pvt_df
):
    production_df, _ = sanitize_mapped_data(
        noisy_daily_production_df, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING,
        is_incremental=True,
    )
    assert len(production_df) == len(noisy_daily_production_df)
    assert not (production_df["P_avg"] == 0).any()
    assert not production_df["P_avg"].isna().any()


def test_rolling_pressure_smoothing_only_applies_when_incremental(
    noisy_daily_production_df, raw_pvt_df
):
    # is_incremental=False: P_avg is untouched, so the pre-existing zero-drop
    # step purges the shut-in rows instead of the smoother filling them.
    production_df, _ = sanitize_mapped_data(
        noisy_daily_production_df, raw_pvt_df, PRODUCTION_MAPPING, PVT_MAPPING,
        is_incremental=False,
    )
    assert len(production_df) == len(noisy_daily_production_df) - 3
