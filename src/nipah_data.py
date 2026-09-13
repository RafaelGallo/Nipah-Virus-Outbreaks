"""
Shared data-access layer for the Nipah Virus Outbreaks 2025-2026 project.

Every loader returns a tidy, analysis-ready pandas DataFrame so that the
notebook (``notebook/``) and the Streamlit dashboard (``app/``) always read the
exact same numbers from the exact same raw CSVs in ``input/``.

Raw files consumed
------------------
input/nipah_historical_outbreaks_updated.csv : annual outbreak roll-up 1998-2026
input/nipah_wb_2026_cases.csv                : West Bengal (Barasat) 2026 cases
input/nipah_wb_2026_contacts.csv             : West Bengal 2026 contact tracing
input/nipah_kerala_2026_cases.csv            : Kerala (Kozhikode) 2026 case
input/nipah_kerala_2026_contacts.csv         : Kerala 2026 contact tracing
input/nipah_bangladesh_2026.csv              : Naogaon/Rajshahi 2026 case + contacts
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------


def find_project_root(start: Path | str | None = None) -> Path:
    """Walk upwards from ``start`` until a folder containing ``input/`` is found."""
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "input").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the project root (no 'input' folder found).")


PROJECT_ROOT = find_project_root()
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
IMG_DIR = PROJECT_ROOT / "img"

# Analytical constants ------------------------------------------------------
FIRST_YEAR, LAST_YEAR = 1998, 2026          # full span covered by the raw file
FOCUS_COUNTRIES = ("India", "Bangladesh")   # the two countries this project targets

# Age brackets such as "20-30" are mapped to their midpoint so banded ages and
# exact ages can live in the same numeric column.
_AGE_BAND_MID = {
    "0-10": 5.0, "10-20": 15.0, "20-30": 25.0, "30-40": 35.0,
    "40-50": 45.0, "50-60": 55.0, "60-70": 65.0, "70-80": 75.0,
}

# The three raw sources use different vocabularies for the same exposure roles.
_CONTACT_TYPE_MAP = {
    "Colleague (Healthcare)": "Healthcare Worker",
    "Healthcare Worker": "Healthcare Worker",
    "Family/Household": "Family/Household",
    "Community": "Community",
    "Workplace": "Workplace",
    "Patient": "Co-patient (Nosocomial)",
}


def age_band_to_mid(value) -> float:
    """Convert ``'20-30'`` to ``25.0``; numeric ages pass straight through."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in _AGE_BAND_MID:
        return _AGE_BAND_MID[text]
    if "-" in text:
        lo, _, hi = text.partition("-")
        try:
            return (float(lo) + float(hi)) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def age_to_band(age) -> str:
    """Bucket an exact age into the same 10-year bands used by the raw files."""
    if pd.isna(age):
        return "Unknown"
    lower = int(min(max(float(age), 0), 79) // 10 * 10)
    return f"{lower}-{lower + 10}"


# --------------------------------------------------------------------------
# 1. Historical outbreak record (the backbone of the time series)
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_historical_events() -> pd.DataFrame:
    """One row per (year, location) outbreak event, 1998-2026, all countries."""
    df = pd.read_csv(INPUT_DIR / "nipah_historical_outbreaks_updated.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["year"] = df["year"].astype(int)
    df["cases"] = df["cases"].astype(int)
    df["deaths"] = df["deaths"].astype(int)
    # Recompute the CFR from raw counts rather than trusting the stored column:
    # it keeps the ratio internally consistent with cases/deaths everywhere.
    df["cfr_recomputed"] = np.where(df["cases"] > 0, df["deaths"] / df["cases"] * 100.0, np.nan)
    df["location"] = df["location"].str.strip()
    df["country"] = df["country"].str.strip()
    df["is_focus_country"] = df["country"].isin(FOCUS_COUNTRIES)
    df["multi_district"] = df["location"].str.contains("Multiple", case=False, na=False)
    return df.sort_values(["year", "country"]).reset_index(drop=True)


def build_annual_series(
    countries: tuple[str, ...] | list[str] | None = FOCUS_COUNTRIES,
    first_year: int = FIRST_YEAR,
    last_year: int = LAST_YEAR,
) -> pd.DataFrame:
    """
    Aggregate outbreak events into a *gap-free* annual time series.

    Years with no reported outbreak (2000, 2002, 2006, 2009, 2016, 2017, 2020)
    are simply absent from the raw file. Re-indexing over the full calendar and
    filling those with zero is essential: without it every trend, correlation
    and rolling statistic would be computed on a distorted, irregular index.
    """
    events = load_historical_events()
    if countries is not None:
        events = events[events["country"].isin(list(countries))]

    grouped = (
        events.groupby("year")
        .agg(cases=("cases", "sum"), deaths=("deaths", "sum"), outbreak_events=("year", "size"))
        .reindex(range(first_year, last_year + 1), fill_value=0)
    )
    grouped.index.name = "year"
    out = grouped.reset_index()

    out["cfr"] = np.where(out["cases"] > 0, out["deaths"] / out["cases"] * 100.0, np.nan)
    out["survivors"] = out["cases"] - out["deaths"]
    out["had_outbreak"] = (out["cases"] > 0).astype(int)
    out["cum_cases"] = out["cases"].cumsum()
    out["cum_deaths"] = out["deaths"].cumsum()
    out["years_since_start"] = out["year"] - first_year

    # Rolling views (min_periods=1 so the early years are not dropped).
    for window in (3, 5):
        out[f"cases_ma{window}"] = out["cases"].rolling(window, min_periods=1).mean()
        out[f"deaths_ma{window}"] = out["deaths"].rolling(window, min_periods=1).mean()
    out["cases_std5"] = out["cases"].rolling(5, min_periods=2).std()
    # A rolling CFR is a ratio of rolling sums, never a mean of yearly ratios.
    roll_cases = out["cases"].rolling(5, min_periods=1).sum()
    roll_deaths = out["deaths"].rolling(5, min_periods=1).sum()
    out["cfr_ma5"] = np.where(roll_cases > 0, roll_deaths / roll_cases * 100.0, np.nan)

    out["cases_yoy"] = out["cases"].diff()
    out["cases_pct_change"] = out["cases"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
    return out


def build_country_annual_series(
    first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR
) -> pd.DataFrame:
    """Long-format annual series split by country (India / Bangladesh)."""
    frames = []
    for country in FOCUS_COUNTRIES:
        series = build_annual_series((country,), first_year, last_year)
        series.insert(0, "country", country)
        frames.append(series)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 2. Case-level records for the 2026 outbreaks
# --------------------------------------------------------------------------

_DATE_COLS = ["date_onset", "date_admitted", "date_confirmed", "date_death"]


def _finalise_case_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and derive the clinical / surveillance delay intervals."""
    for col in _DATE_COLS:
        df[col] = pd.to_datetime(df.get(col), errors="coerce")

    df["onset_to_admission_days"] = (df["date_admitted"] - df["date_onset"]).dt.days
    df["admission_to_confirm_days"] = (df["date_confirmed"] - df["date_admitted"]).dt.days
    df["onset_to_confirm_days"] = (df["date_confirmed"] - df["date_onset"]).dt.days
    df["onset_to_death_days"] = (df["date_death"] - df["date_onset"]).dt.days
    df["confirm_to_death_days"] = (df["date_death"] - df["date_confirmed"]).dt.days
    df["age_mid"] = df["age_group"].map(age_band_to_mid)
    df["died"] = df["final_status"].str.contains("Deceased", case=False, na=False).astype(int)
    df["onset_month"] = df["date_onset"].dt.month
    df["onset_year"] = df["date_onset"].dt.year
    return df


def load_case_timeline() -> pd.DataFrame:
    """
    All four laboratory-confirmed 2026 cases (West Bengal x2, Kerala x1,
    Bangladesh x1) harmonised into a single case-level frame.
    """
    wb_raw = pd.read_csv(INPUT_DIR / "nipah_wb_2026_cases.csv")
    wb = pd.DataFrame({
        "case_id": wb_raw["Case_ID"],
        "outbreak": "West Bengal 2026 (Barasat)",
        "country": "India",
        "region": wb_raw["State"],
        "district": wb_raw["District"],
        "age_group": wb_raw["Age_Group"],
        "gender": wb_raw["Gender"],
        "occupation": wb_raw["Occupation"],
        "date_onset": wb_raw["Date_Onset"],
        "date_admitted": wb_raw["Date_Admitted"],
        "date_confirmed": wb_raw["Date_Confirmed"],
        "date_death": wb_raw["Date_Death"],
        "exposure_source": wb_raw["Exposure_Source"],
        "final_status": wb_raw["Final_Status"],
    })

    kl_raw = pd.read_csv(INPUT_DIR / "nipah_kerala_2026_cases.csv")
    kl = pd.DataFrame({
        "case_id": kl_raw["Case_ID"],
        "outbreak": "Kerala 2026 (Kozhikode)",
        "country": "India",
        "region": kl_raw["State"],
        "district": kl_raw["District"],
        "age_group": kl_raw["Age"].map(age_to_band),
        "gender": kl_raw["Gender"],
        "occupation": kl_raw["Occupation"],
        "date_onset": kl_raw["Date_Onset"],
        "date_admitted": kl_raw["Date_Admitted"],
        "date_confirmed": kl_raw["Date_Confirmed"],
        "date_death": kl_raw["Date_Death"],
        "exposure_source": kl_raw["Exposure_Source"],
        "final_status": kl_raw["Final_Status"],
    })

    bd_raw = pd.read_csv(INPUT_DIR / "nipah_bangladesh_2026.csv")
    bd_rows = bd_raw[bd_raw["Record_Type"] == "Case"].copy()
    bd = pd.DataFrame({
        "case_id": bd_rows["Record_ID"],
        "outbreak": "Bangladesh 2026 (Naogaon)",
        "country": "Bangladesh",
        "region": bd_rows["Division"],
        "district": bd_rows["District"],
        "age_group": bd_rows["Age_Group"],
        "gender": bd_rows["Gender"],
        "occupation": "Not reported",
        "date_onset": bd_rows["Date_Onset"],
        "date_admitted": bd_rows["Date_Admitted"],
        "date_confirmed": bd_rows["Date_Confirmed"],
        "date_death": bd_rows["Date_Death"],
        "exposure_source": bd_rows["Transmission_Source"],
        "final_status": bd_rows["Final_Status"],
    })

    cases = pd.concat([wb, kl, bd], ignore_index=True)
    return _finalise_case_frame(cases).sort_values("date_onset").reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Contact tracing (the daily surveillance signal)
# --------------------------------------------------------------------------


def load_contacts() -> pd.DataFrame:
    """
    All 331 traced contacts (West Bengal 192 + Kerala 104 + Bangladesh 35)
    harmonised into one frame. Bangladesh contacts carry no sampling date, so
    they are excluded from the daily curves but kept for the demographics.
    """
    wb_raw = pd.read_csv(INPUT_DIR / "nipah_wb_2026_contacts.csv")
    wb = pd.DataFrame({
        "contact_id": wb_raw["Contact_ID"],
        "outbreak": "West Bengal 2026 (Barasat)",
        "country": "India",
        "district": wb_raw["District"],
        "linked_case_id": wb_raw["Linked_Case_ID"],
        "contact_type_raw": wb_raw["Contact_Type"],
        "risk_category": np.nan,
        "age": wb_raw["Age"].astype(float),
        "gender": wb_raw["Gender"],
        "date_sample": wb_raw["Date_Sample_Collected"],
        "test_result_raw": wb_raw["Test_Result"],
    })

    kl_raw = pd.read_csv(INPUT_DIR / "nipah_kerala_2026_contacts.csv")
    kl = pd.DataFrame({
        "contact_id": kl_raw["Contact_ID"],
        "outbreak": "Kerala 2026 (Kozhikode)",
        "country": "India",
        "district": kl_raw["District"],
        "linked_case_id": kl_raw["Linked_Case_ID"],
        "contact_type_raw": kl_raw["Contact_Type"],
        "risk_category": kl_raw["Risk_Category"],
        "age": kl_raw["Age"].astype(float),
        "gender": kl_raw["Gender"],
        "date_sample": kl_raw["Date_Sample_Collected"],
        "test_result_raw": kl_raw["Test_Result"],
    })

    bd_raw = pd.read_csv(INPUT_DIR / "nipah_bangladesh_2026.csv")
    bd_rows = bd_raw[bd_raw["Record_Type"] == "Contact"].copy()
    bd = pd.DataFrame({
        "contact_id": bd_rows["Record_ID"],
        "outbreak": "Bangladesh 2026 (Naogaon)",
        "country": "Bangladesh",
        "district": bd_rows["District"],
        "linked_case_id": "BD-2026-001",
        "contact_type_raw": "Community",
        "risk_category": np.nan,
        "age": bd_rows["Age_Group"].map(age_band_to_mid),
        "gender": bd_rows["Gender"],
        "date_sample": pd.NaT,
        "test_result_raw": bd_rows["Test_Result"],
    })

    contacts = pd.concat([wb, kl, bd], ignore_index=True)
    contacts["contact_type"] = (
        contacts["contact_type_raw"].map(_CONTACT_TYPE_MAP).fillna(contacts["contact_type_raw"])
    )
    contacts["date_sample"] = pd.to_datetime(contacts["date_sample"], errors="coerce")
    contacts["age_band"] = contacts["age"].map(age_to_band)
    # "Negative (Symptomatic)" is still a negative NiV test, but the person did
    # develop symptoms - a materially different surveillance signal.
    contacts["is_positive"] = contacts["test_result_raw"].str.contains(
        "Positive", case=False, na=False)
    contacts["was_symptomatic"] = contacts["test_result_raw"].str.contains(
        "Symptomatic", case=False, na=False)
    return contacts.reset_index(drop=True)


def build_surveillance_curve() -> pd.DataFrame:
    """
    Daily contact-sampling counts per outbreak, re-indexed over every calendar
    day of the campaign so that idle days appear as explicit zeros.
    """
    contacts = load_contacts().dropna(subset=["date_sample"])
    frames = []
    for outbreak, group in contacts.groupby("outbreak"):
        span = pd.date_range(group["date_sample"].min(), group["date_sample"].max(), freq="D")
        daily = (
            group.groupby("date_sample").size()
            .reindex(span, fill_value=0)
            .rename("samples_collected")
        )
        frame = daily.rename_axis("date").reset_index()
        frame["outbreak"] = outbreak
        frame["country"] = group["country"].iloc[0]
        frame["cum_samples"] = frame["samples_collected"].cumsum()
        total = frame["samples_collected"].sum()
        frame["pct_cohort_cleared"] = frame["cum_samples"] / total * 100.0
        frame["campaign_day"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[
        ["outbreak", "country", "date", "campaign_day", "samples_collected",
         "cum_samples", "pct_cohort_cleared"]
    ]


def build_tracing_efficiency() -> pd.DataFrame:
    """Contacts traced per confirmed case and campaign duration, per outbreak."""
    cases = load_case_timeline()
    contacts = load_contacts()
    rows = []
    for outbreak, group in contacts.groupby("outbreak"):
        case_rows = cases[cases["outbreak"] == outbreak]
        dated = group.dropna(subset=["date_sample"])
        n_cases = max(len(case_rows), 1)
        rows.append({
            "outbreak": outbreak,
            "country": group["country"].iloc[0],
            "confirmed_cases": len(case_rows),
            "deaths": int(case_rows["died"].sum()),
            "contacts_traced": len(group),
            "contacts_per_case": len(group) / n_cases,
            "campaign_days": int(dated["date_sample"].nunique()) if len(dated) else np.nan,
            "samples_per_day": (len(dated) / dated["date_sample"].nunique()) if len(dated) else np.nan,
            "first_sample": dated["date_sample"].min() if len(dated) else pd.NaT,
            "last_sample": dated["date_sample"].max() if len(dated) else pd.NaT,
            "secondary_positives": int(group["is_positive"].sum()),
            "symptomatic_contacts": int(group["was_symptomatic"].sum()),
        })
    return pd.DataFrame(rows).sort_values("outbreak").reset_index(drop=True)


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience bundle used by both the notebook and the dashboard."""
    return {
        "events": load_historical_events(),
        "annual": build_annual_series(),
        "annual_by_country": build_country_annual_series(),
        "cases": load_case_timeline(),
        "contacts": load_contacts(),
        "surveillance": build_surveillance_curve(),
        "tracing": build_tracing_efficiency(),
    }


if __name__ == "__main__":  # quick smoke test: python src/nipah_data.py
    for name, frame in load_all().items():
        print(f"{name:20s} {frame.shape}")
