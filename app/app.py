"""
Nipah Virus Outbreaks 2025-2026 (India & Bangladesh)
Interactive time-series dashboard.

Run from the project root:

    streamlit run app/app.py

The dashboard imports ``src/nipah_data.py`` and ``src/train_models.py`` - the same
modules the analysis notebook uses - so every figure here is guaranteed to agree
with `notebook/nipah_time_series_analysis.ipynb` by construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.stattools import acf, adfuller


# --------------------------------------------------------------------------
# Bootstrapping
# --------------------------------------------------------------------------


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "input").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the project root (no 'input' folder found).")


ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "src"))

import nipah_data as nd          # noqa: E402
import train_models as tm        # noqa: E402

st.set_page_config(
    page_title="Nipah Virus Outbreaks 2025-2026",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design tokens - the same palette the notebook uses
# --------------------------------------------------------------------------

C = {
    "india": "#2A6F97", "bangladesh": "#E07A5F", "accent": "#3D5A80",
    "case": "#457B9D", "death": "#B23A48", "good": "#2A9D8F",
    "warn": "#E9C46A", "dark": "#22333B", "muted": "#8D99AE", "grid": "#E4E7EB",
}
OUTBREAK_COLORS = {
    "West Bengal 2026 (Barasat)": C["india"],
    "Kerala 2026 (Kozhikode)": C["good"],
    "Bangladesh 2026 (Naogaon)": C["bangladesh"],
}
# NOTE: no `title` key here - every caller passes its own `title=`, and having
# both would collide on the same keyword in update_layout().
PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13,
              color=C["dark"]),
    margin=dict(l=10, r=10, t=56, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    title_x=0.0,
    title_xanchor="left",
    title_font=dict(size=16),
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetricValue"] {font-size: 1.75rem;}
      [data-testid="stMetricLabel"] {font-size: .8rem; letter-spacing: .02em;}
      h1 {letter-spacing: -.02em;}
      .insight {
        background: #F4F6F8; border-left: 4px solid #3D5A80;
        padding: .85rem 1.1rem; border-radius: 4px; margin: .6rem 0 1.1rem 0;
        font-size: .93rem; line-height: 1.55;
      }
      .warn-box {
        background: #FDF6E7; border-left: 4px solid #E9C46A;
        padding: .85rem 1.1rem; border-radius: 4px; margin: .6rem 0 1.1rem 0;
        font-size: .93rem; line-height: 1.55;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def insight(text: str, kind: str = "insight") -> None:
    """Render a short interpretation block beneath a chart."""
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading outbreak data…")
def load_data() -> dict[str, pd.DataFrame]:
    return nd.load_all()


@st.cache_data(show_spinner="Loading trend models…")
def load_model_artefacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Read the persisted model outputs, training them first if absent."""
    summary_path = ROOT / "output" / "model_performance_summary.csv"
    if not summary_path.exists() or not (ROOT / "models" / "model_registry.json").exists():
        tm.train_and_save(verbose=False)

    summary = pd.read_csv(summary_path)
    curves = pd.read_csv(ROOT / "output" / "model_fitted_curves.csv")
    grid = pd.read_csv(ROOT / "output" / "model_degree_search.csv")
    with open(ROOT / "models" / "model_registry.json", encoding="utf-8") as fh:
        registry = json.load(fh)
    return summary, curves, grid, registry


data = load_data()
annual_full = data["annual"]
by_country_full = data["annual_by_country"]
cases = data["cases"]
contacts = data["contacts"]
surveillance = data["surveillance"]
tracing = data["tracing"]
events = data["events"]

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🦠 Nipah Virus")
    st.caption("Outbreak surveillance & time-series analysis · India and Bangladesh, 1998–2026")
    st.divider()

    st.subheader("Filters")
    year_range = st.slider(
        "Year range",
        int(nd.FIRST_YEAR), int(nd.LAST_YEAR),
        (int(nd.FIRST_YEAR), int(nd.LAST_YEAR)),
        help="Applies to the historical time-series views.",
    )
    countries = st.multiselect(
        "Countries", list(nd.FOCUS_COUNTRIES), default=list(nd.FOCUS_COUNTRIES),
        help="The dataset also contains Malaysia, Singapore and the Philippines; "
             "this project focuses on the India–Bangladesh belt.",
    )
    if not countries:
        st.warning("Select at least one country.")
        st.stop()

    smoothing = st.select_slider(
        "Trend smoothing window", options=[1, 3, 5, 7], value=5,
        help="Centred moving average applied to the annual series.",
    )

    st.divider()
    st.subheader("Dataset")
    st.metric("Outbreak events", len(events))
    st.metric("2026 confirmed cases", len(cases))
    st.metric("2026 contacts traced", len(contacts))
    st.caption(
        "Source: WHO Disease Outbreak News, CDC and peer-reviewed outbreak reports. "
        "Loaded through `src/nipah_data.py`, the same module the analysis notebook uses."
    )


@st.cache_data
def filtered_series(countries: tuple, lo: int, hi: int) -> pd.DataFrame:
    series = nd.build_annual_series(tuple(countries), lo, hi)
    return series


annual = filtered_series(tuple(countries), year_range[0], year_range[1])
annual = annual.copy()
annual["trend"] = (annual["cases"].rolling(smoothing, center=True, min_periods=1).mean()
                   if smoothing > 1 else annual["cases"])

by_country = by_country_full[
    by_country_full["country"].isin(countries)
    & by_country_full["year"].between(*year_range)
]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Nipah Virus Outbreaks 2025–2026")
st.markdown(
    "**India & Bangladesh · a complete time-series analysis.** Nipah virus kills roughly "
    "three of every four people it infects, but arrives in small sporadic bursts rather "
    "than sustained epidemics — a zero-inflated, spike-dominated count process observed "
    "only 29 times."
)

total_cases = int(annual["cases"].sum())
total_deaths = int(annual["deaths"].sum())
pooled_cfr = total_deaths / total_cases * 100 if total_cases else 0.0
active_years = int(annual["had_outbreak"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Confirmed cases", f"{total_cases:,}",
          help=f"{year_range[0]}–{year_range[1]}, {', '.join(countries)}")
k2.metric("Deaths", f"{total_deaths:,}")
k3.metric("Pooled CFR", f"{pooled_cfr:.1f}%",
          help="Total deaths ÷ total cases — patient-weighted, not a mean of yearly ratios.")
k4.metric("Active years", f"{active_years} / {len(annual)}",
          help="Years with at least one reported outbreak.")
k5.metric("Contacts traced (2026)", f"{len(contacts)}",
          delta=f"{int(contacts['is_positive'].sum())} positive",
          delta_color="off")

st.divider()

TAB_OVERVIEW, TAB_TS, TAB_COUNTRY, TAB_2026, TAB_MODEL, TAB_DATA = st.tabs([
    "📈 Overview",
    "🔬 Time-series diagnostics",
    "🌏 Country comparison",
    "🏥 2026 outbreaks",
    "🤖 Polynomial model",
    "📄 Data & exports",
])

# ==========================================================================
# TAB 1 — OVERVIEW
# ==========================================================================
with TAB_OVERVIEW:
    st.subheader("Annual burden and the underlying trend")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=annual["year"], y=annual["cases"], name="Confirmed cases",
        marker_color=C["case"], hovertemplate="%{x}: %{y} cases<extra></extra>"))
    fig.add_trace(go.Bar(
        x=annual["year"], y=annual["deaths"], name="Deaths",
        marker_color=C["death"], hovertemplate="%{x}: %{y} deaths<extra></extra>"))
    if smoothing > 1:
        fig.add_trace(go.Scatter(
            x=annual["year"], y=annual["trend"], name=f"{smoothing}-yr centred mean",
            mode="lines", line=dict(color=C["dark"], width=3),
            hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["cfr"], name="CFR (%)", mode="markers",
        marker=dict(color=C["warn"], size=9, line=dict(color=C["dark"], width=1)),
        hovertemplate="%{x}: CFR %{y:.1f}%<extra></extra>"), secondary_y=True)

    fig.update_layout(barmode="overlay", height=460,
                      title="Cases, deaths and case fatality ratio by year", **PLOT_LAYOUT)
    fig.update_traces(opacity=0.95, selector=dict(type="bar"))
    fig.update_yaxes(title_text="People", secondary_y=False)
    fig.update_yaxes(title_text="CFR (%)", secondary_y=True, range=[0, 110], showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    insight(
        "<b>Deaths track cases almost perfectly.</b> The CFR markers hover near 70–100% in "
        "nearly every outbreak year and show no downward drift across 28 years — lethality "
        "is effectively fixed. Note the <b>zero years</b> (2000, 2002, 2006, 2009, 2016, "
        "2017, 2020): they carry no row in the raw file and are re-inserted here as explicit "
        "zeros. Without that step every trend statistic below would be computed on a "
        "distorted index."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Burden concentration")
        ranked = annual.sort_values("cases", ascending=False).reset_index(drop=True)
        ranked["share"] = ranked["cases"] / ranked["cases"].sum() * 100
        ranked["cum_share"] = ranked["share"].cumsum()
        top = ranked.head(12)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=top["year"].astype(str), y=top["cases"], name="Cases",
            marker_color=C["case"], hovertemplate="%{x}: %{y} cases<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=top["year"].astype(str), y=top["cum_share"], name="Cumulative share",
            mode="lines+markers", line=dict(color=C["death"], width=2.5),
            hovertemplate="%{x}: %{y:.1f}% cumulative<extra></extra>"), secondary_y=True)
        fig.add_hline(y=80, line=dict(color=C["muted"], dash="dot"), secondary_y=True)
        fig.update_layout(height=380, title="Pareto — the worst 12 years", **PLOT_LAYOUT)
        fig.update_yaxes(title_text="Cases", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True,
                         range=[0, 105], showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        def gini(values) -> float:
            v = np.sort(np.asarray(values, dtype=float))
            n = len(v)
            if v.sum() == 0:
                return 0.0
            idx = np.arange(1, n + 1)
            return float((2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)

        g = gini(annual["cases"])
        top3 = ranked.head(3)["share"].sum()
        insight(
            f"<b>Gini = {g:.3f}</b> — a more unequal distribution than household income in "
            f"most countries. The three worst years hold <b>{top3:.1f}%</b> of all cases. "
            "The largest years (Siliguri 2001, Faridpur 2004) were both amplified inside "
            "hospitals — which is why <b>surge capacity, not average throughput</b>, is the "
            "binding planning constraint."
        )

    with right:
        st.subheader("Recurrence interval")
        active = annual.loc[annual["cases"] > 0, "year"].to_numpy()
        if len(active) > 1:
            gaps = np.diff(active)
            vals, counts = np.unique(gaps, return_counts=True)
            fig = go.Figure(go.Bar(
                x=[f"{v} yr" for v in vals], y=counts, marker_color=C["accent"],
                text=counts, textposition="outside",
                hovertemplate="%{x} gap: %{y} times<extra></extra>"))
            fig.update_layout(height=380, title="Years between consecutive active years",
                              **PLOT_LAYOUT)
            fig.update_yaxes(title_text="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            one_year = int(counts[vals == 1][0]) if 1 in vals else 0
            recent = annual[annual["year"] >= 2021]
            insight(
                f"<b>{one_year} of {len(gaps)} gaps are exactly one year</b>, and the longest "
                f"silence on record is {gaps.max()} years. Recent activity: "
                f"<b>{int(recent['had_outbreak'].sum())} of {len(recent)} years since 2021</b>. "
                "Nipah is no longer episodic — it is an annual seasonal function, and response "
                "capacity should be <b>standing rather than on-call</b>."
            )
        else:
            st.info("Widen the year range to see the recurrence structure.")

    st.subheader("Cumulative burden")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["cum_cases"], name="Cumulative cases",
        mode="lines", line=dict(color=C["case"], width=3), fill="tozeroy",
        fillcolor="rgba(69,123,157,0.18)"))
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["cum_deaths"], name="Cumulative deaths",
        mode="lines", line=dict(color=C["death"], width=3), fill="tozeroy",
        fillcolor="rgba(178,58,72,0.18)"))
    fig.update_layout(height=340, title="Cumulative cases and deaths", **PLOT_LAYOUT)
    fig.update_yaxes(title_text="People")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 2 — TIME-SERIES DIAGNOSTICS
# ==========================================================================
with TAB_TS:
    st.subheader("Is there a trend? Three tests, three answers")

    years = annual["year"].to_numpy()
    counts_arr = annual["cases"].to_numpy(float)

    if len(annual) >= 8 and counts_arr.sum() > 0:
        tau, p_tau = stats.kendalltau(years, counts_arr)
        ts_slope, ts_int, ts_lo, ts_hi = stats.theilslopes(counts_arr, years, 0.95)
        dispersion = counts_arr.var(ddof=1) / counts_arr.mean() if counts_arr.mean() else np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("Mann-Kendall τ", f"{tau:+.3f}", f"p = {p_tau:.3f}", delta_color="off",
                  help="Rank-based. Assumes neither linearity nor normality.")
        m2.metric("Theil-Sen slope", f"{ts_slope:+.2f}/yr",
                  f"95% CI {ts_lo:+.2f} to {ts_hi:+.2f}", delta_color="off",
                  help="Median of all pairwise slopes — robust to the 2001/2004 spikes.")
        m3.metric("Overdispersion", f"{dispersion:.1f}×",
                  "Poisson assumes 1.0×", delta_color="off",
                  help="Variance ÷ mean. Anything far above 1 invalidates Poisson p-values.")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=counts_arr, name="Cases",
                             marker_color=C["muted"], opacity=0.55))
        fig.add_trace(go.Scatter(
            x=years, y=ts_int + ts_slope * years, name="Theil-Sen trend",
            mode="lines", line=dict(color=C["dark"], width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(
            x=annual["year"], y=annual["trend"], name=f"{smoothing}-yr centred mean",
            mode="lines", line=dict(color=C["accent"], width=3.5)))
        fig.update_layout(height=400, title="Raw counts vs robust trend vs smoothed level",
                          **PLOT_LAYOUT)
        fig.update_yaxes(title_text="Cases")
        st.plotly_chart(fig, use_container_width=True)

        verdict = ("no monotonic trend" if p_tau > 0.05
                   else ("a significant decline" if tau < 0 else "a significant increase"))
        insight(
            f"Mann-Kendall reports <b>{verdict}</b> (τ = {tau:+.3f}, p = {p_tau:.3f}). "
            f"With overdispersion at <b>{dispersion:.1f}×</b> the Poisson assumption, a "
            "Poisson GLM on this same series returns p ≈ 10⁻⁹ — an artefact of standard "
            "errors roughly √26 ≈ 5× too small. <b>The rank-based null result is the honest "
            "reading.</b> What the smoothed line shows is not a steady decline but a "
            "<b>two-wave shape</b> ending in a sustained low — which a monotonic test "
            "correctly declines to call a trend."
        )
    else:
        st.info("Select a wider year range to run the trend diagnostics.")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Does one year predict the next?")
        series = annual.set_index("year")["cases"]
        if len(series) >= 12:
            max_lag = min(10, len(series) // 2)
            acf_vals = acf(series, nlags=max_lag, fft=False)
            band = 1.96 / np.sqrt(len(series))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(max_lag + 1)), y=acf_vals, marker_color=C["case"],
                name="ACF", hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"))
            fig.add_hrect(y0=-band, y1=band, fillcolor=C["muted"], opacity=0.22, line_width=0)
            fig.add_hline(y=0, line=dict(color=C["dark"], width=1))
            fig.update_layout(height=340, title="Autocorrelation function",
                              **{**PLOT_LAYOUT, "hovermode": "closest"})
            fig.update_xaxes(title_text="Lag (years)")
            fig.update_yaxes(title_text="ACF")
            st.plotly_chart(fig, use_container_width=True)

            adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
            sig = [i for i in range(1, max_lag + 1) if abs(acf_vals[i]) > band]
            insight(
                f"Lag-1 ACF = <b>{acf_vals[1]:+.3f}</b> against a ±{band:.3f} band; "
                f"significant lags: <b>{sig if sig else 'none'}</b>. ADF p = "
                f"<b>{adf_p:.4f}</b> ({'stationary' if adf_p < 0.05 else 'non-stationary'}). "
                "<b>The series has no memory</b> — each outbreak is an independent spillover "
                "from the bat reservoir, not a continuation of the last one. That is the "
                "statistical fingerprint of contained transmission, and it is why ARIMA and "
                "lag-feature models are inapplicable here."
            )
        else:
            st.info("Need at least 12 years for autocorrelation diagnostics.")

    with col_b:
        st.subheader("Is volatility falling?")
        vol = annual.copy()
        vol["rolling_sd"] = vol["cases"].rolling(5, min_periods=2).std()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vol["year"], y=vol["rolling_sd"], name="5-yr rolling SD", mode="lines+markers",
            line=dict(color=C["death"], width=2.5), fill="tozeroy",
            fillcolor="rgba(178,58,72,0.15)"))
        fig.update_layout(height=340, title="Rolling standard deviation", **PLOT_LAYOUT)
        fig.update_yaxes(title_text="SD (cases)")
        st.plotly_chart(fig, use_container_width=True)

        # Split into equal halves, with the earlier half taking floor(n/2) years.
        # Over the full 1998-2026 range this gives 1998-2011 vs 2012-2026, the
        # same boundary the analysis notebook uses.
        mid = int(annual["year"].iloc[len(annual) // 2 - 1])
        early = annual.loc[annual["year"] <= mid, "cases"]
        late = annual.loc[annual["year"] > mid, "cases"]
        if len(early) > 2 and len(late) > 2 and late.std() > 0:
            bf_stat, bf_p = stats.levene(early, late, center="median")
            insight(
                f"SD fell from <b>{early.std():.2f}</b> ({year_range[0]}–{mid}) to "
                f"<b>{late.std():.2f}</b> ({mid + 1}–{year_range[1]}) — a "
                f"<b>{early.std() / late.std():.2f}× reduction</b>. Brown-Forsythe "
                f"p = {bf_p:.3f}. Because Nipah planning is governed by the tail rather than "
                "the average (see the Pareto chart), <b>shrinking volatility is a bigger "
                "operational win than a falling mean</b> — the worst year fell from 79 cases "
                "to 30."
            )

    st.divider()
    st.subheader("Was there a structural break?")

    # Scanned over the full selected range, matching the notebook's methodology
    # so the default 1998-2026 view reproduces its figures exactly.
    if len(annual) >= 12:
        scan = []
        lo_lim, hi_lim = annual["year"].min() + 4, annual["year"].max() - 4
        for split in range(int(lo_lim), int(hi_lim) + 1):
            pre = annual.loc[annual["year"] <= split, "cases"]
            post = annual.loc[annual["year"] > split, "cases"]
            if len(pre) < 3 or len(post) < 3:
                continue
            t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
            scan.append({"split": split, "t": abs(t_stat), "p": p_val,
                         "pre": pre.mean(), "post": post.mean()})
        scan = pd.DataFrame(scan)

        if len(scan):
            best = scan.loc[scan["t"].idxmax()]
            bonf = min(best["p"] * len(scan), 1.0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scan["split"], y=scan["t"], mode="lines+markers", name="|Welch t|",
                line=dict(color=C["accent"], width=2.5), fill="tozeroy",
                fillcolor="rgba(61,90,128,0.15)"))
            fig.add_vline(x=best["split"], line=dict(color=C["death"], dash="dash", width=2))
            fig.update_layout(height=330,
                              title=f"Break scan — strongest split at {int(best['split'])}",
                              **PLOT_LAYOUT)
            fig.update_xaxes(title_text="Candidate split year")
            fig.update_yaxes(title_text="|Welch t|")
            st.plotly_chart(fig, use_container_width=True)

            b1, b2, b3 = st.columns(3)
            b1.metric("Break year", f"{int(best['split'])}")
            b2.metric("Mean before → after",
                      f"{best['pre']:.1f} → {best['post']:.1f}",
                      f"{(best['post'] / best['pre'] - 1) * 100:+.0f}%", delta_color="inverse")
            b3.metric("Bonferroni p", f"{bonf:.3f}",
                      f"raw p = {best['p']:.3f}", delta_color="off",
                      help=f"Corrected for the {len(scan)} candidate splits tested.")

            insight(
                f"The strongest split is <b>{int(best['split'])}</b>, with the mean falling "
                f"from <b>{best['pre']:.1f}</b> to <b>{best['post']:.1f}</b> cases/year. But "
                f"because {len(scan)} candidate years were scanned, the honest p-value is the "
                f"Bonferroni-corrected <b>{bonf:.3f}</b>, not the raw {best['p']:.3f}. "
                "<b>Treat the post-break level as a reasonable operating baseline, not an "
                "established fact</b> — and re-test it as each year's data arrives.",
                kind="warn-box",
            )

# ==========================================================================
# TAB 3 — COUNTRY COMPARISON
# ==========================================================================
with TAB_COUNTRY:
    st.subheader("India vs Bangladesh — same lethality, different rhythm")

    profile = []
    for country in countries:
        frame = by_country[by_country["country"] == country]
        if not len(frame) or frame["cases"].sum() == 0:
            continue
        active = frame[frame["cases"] > 0]
        profile.append({
            "Country": country,
            "Cases": int(frame["cases"].sum()),
            "Deaths": int(frame["deaths"].sum()),
            "Pooled CFR (%)": round(frame["deaths"].sum() / frame["cases"].sum() * 100, 1),
            "Active years": int(frame["had_outbreak"].sum()),
            "Activity rate (%)": round(frame["had_outbreak"].mean() * 100, 1),
            "Median size": float(active["cases"].median()),
            "Worst year": int(frame["cases"].max()),
        })
    profile = pd.DataFrame(profile)

    if len(profile):
        st.dataframe(profile, use_container_width=True, hide_index=True)

    fig = go.Figure()
    for country in countries:
        frame = by_country[by_country["country"] == country]
        colour = C["bangladesh"] if country == "Bangladesh" else C["india"]
        fig.add_trace(go.Bar(x=frame["year"], y=frame["cases"], name=country,
                             marker_color=colour,
                             hovertemplate=f"{country} %{{x}}: %{{y}} cases<extra></extra>"))
    fig.update_layout(barmode="group", height=400,
                      title="Annual cases by country", **PLOT_LAYOUT)
    fig.update_yaxes(title_text="Cases")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        for country in countries:
            frame = by_country[by_country["country"] == country]
            colour = C["bangladesh"] if country == "Bangladesh" else C["india"]
            fig.add_trace(go.Scatter(
                x=frame["year"], y=frame["cum_cases"], name=country, mode="lines",
                line=dict(color=colour, width=3)))
        fig.update_layout(height=360, title="Cumulative burden", **PLOT_LAYOUT)
        fig.update_yaxes(title_text="Cumulative cases")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = go.Figure()
        for country in countries:
            frame = by_country[by_country["country"] == country]
            sizes = frame.loc[frame["cases"] > 0, "cases"]
            if not len(sizes):
                continue
            colour = C["bangladesh"] if country == "Bangladesh" else C["india"]
            fig.add_trace(go.Box(y=sizes, name=country, marker_color=colour,
                                 boxpoints="all", jitter=0.4, pointpos=-1.6))
        fig.update_layout(height=360, title="Outbreak size in an active year",
                          **{**PLOT_LAYOUT, "hovermode": "closest"})
        fig.update_yaxes(title_text="Cases")
        st.plotly_chart(fig, use_container_width=True)

    if len(profile) == 2:
        bd = by_country[by_country["country"] == "Bangladesh"]
        ind = by_country[by_country["country"] == "India"]
        try:
            _, p_size = stats.mannwhitneyu(bd.loc[bd["cases"] > 0, "cases"],
                                           ind.loc[ind["cases"] > 0, "cases"])
            _, p_cfr = proportions_ztest([bd["deaths"].sum(), ind["deaths"].sum()],
                                         [bd["cases"].sum(), ind["cases"].sum()])
            _, p_act = proportions_ztest(
                [int(bd["had_outbreak"].sum()), int(ind["had_outbreak"].sum())],
                [len(bd), len(ind)])
            insight(
                f"<b>The significant difference is frequency, not severity.</b> Activity rate "
                f"differs at <b>p = {p_act:.3f}</b>, while outbreak size (p = {p_size:.3f}) and "
                f"pooled CFR (p = {p_cfr:.3f}) do not. Both countries face a virus that kills "
                "roughly three in four. What differs is the route: Bangladesh's near-annual "
                "<b>raw date-palm sap</b> spillover during the December–April harvest, versus "
                "India's rare but explosive <b>nosocomial amplification</b>. Same outcome, "
                "different intervention point — so a single regional strategy will "
                "misallocate."
            )
        except Exception:
            pass

# ==========================================================================
# TAB 4 — 2026 OUTBREAKS
# ==========================================================================
with TAB_2026:
    st.subheader("The three 2026 outbreak investigations")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confirmed cases", len(cases), f"{int(cases['died'].sum())} deaths",
              delta_color="inverse")
    m2.metric("Contacts traced", len(contacts))
    m3.metric("Secondary cases", int(contacts["is_positive"].sum()),
              "0.00% attack rate", delta_color="off")
    m4.metric("Mean onset → confirmation",
              f"{cases['onset_to_confirm_days'].mean():.1f} d")

    insight(
        "<b>Zero of 331 traced contacts tested positive.</b> Every one of the four confirmed "
        "2026 cases was a terminal node — Nipah did not transmit onward anywhere in India or "
        "Bangladesh in 2026. Against Siliguri 2001, where one introduction produced 66 cases "
        "through hospital amplification, that is a categorical change in outcome."
    )

    st.markdown("#### Detection speed — where the delay actually sits")

    fig = go.Figure()
    ordered = cases.sort_values("date_onset")
    for _, row in ordered.iterrows():
        colour = OUTBREAK_COLORS.get(row["outbreak"], C["muted"])
        fig.add_trace(go.Scatter(
            x=[row["date_onset"], row["date_admitted"]], y=[row["case_id"]] * 2,
            mode="lines", line=dict(color=colour, width=11), opacity=0.35,
            name="Onset → admission", showlegend=False,
            hovertemplate=f"{row['case_id']}: {row['onset_to_admission_days']:.0f}d to "
                          "admission<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[row["date_admitted"], row["date_confirmed"]], y=[row["case_id"]] * 2,
            mode="lines", line=dict(color=colour, width=11), opacity=0.85,
            name="Admission → confirmation", showlegend=False,
            hovertemplate=f"{row['case_id']}: {row['admission_to_confirm_days']:.0f}d to "
                          "confirmation<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[row["date_confirmed"]], y=[row["case_id"]], mode="markers",
            marker=dict(symbol="diamond", size=12, color=C["dark"]),
            name="Lab confirmation", showlegend=False,
            hovertemplate=f"Confirmed {row['date_confirmed']:%Y-%m-%d}<extra></extra>"))
        if pd.notna(row["date_death"]):
            fig.add_trace(go.Scatter(
                x=[row["date_death"]], y=[row["case_id"]], mode="markers",
                marker=dict(symbol="x", size=14, color=C["death"]),
                name="Death", showlegend=False,
                hovertemplate=f"Died {row['date_death']:%Y-%m-%d}<extra></extra>"))
    fig.update_layout(height=330, title="Clinical timeline of every 2026 case",
                      **{**PLOT_LAYOUT, "hovermode": "closest"})
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cases[["case_id", "outbreak", "age_group", "gender", "exposure_source",
               "onset_to_admission_days", "admission_to_confirm_days",
               "onset_to_confirm_days", "final_status"]],
        use_container_width=True, hide_index=True,
    )

    insight(
        "<b>Three of the four patients took exactly 7 days to reach hospital</b> — Nipah "
        "begins as undifferentiated fever, and no laboratory investment shortens that week. "
        "The <b>controllable</b> interval is admission → confirmation, and it varies fivefold: "
        "Bangladesh and Kerala confirmed in <b>1 day</b>; West Bengal took <b>8 and 9</b>. "
        "The difference is clinical suspicion, not capacity — and during those 8 days an "
        "unrecognised Nipah patient sat in a general ward, which is exactly the configuration "
        "that produced Siliguri's 66 cases. Note also <b>BD-2026-001, confirmed the day "
        "after the patient died.</b>"
    )

    st.divider()
    st.markdown("#### Surveillance response — how fast was each cohort cleared?")

    c_left, c_right = st.columns(2)

    with c_left:
        fig = go.Figure()
        for outbreak, group in surveillance.groupby("outbreak"):
            fig.add_trace(go.Scatter(
                x=group["campaign_day"], y=group["samples_collected"],
                name=outbreak.split(" 2026")[0], mode="lines+markers",
                line=dict(color=OUTBREAK_COLORS[outbreak], width=2.5)))
        fig.update_layout(height=360, title="Daily sampling throughput", **PLOT_LAYOUT)
        fig.update_xaxes(title_text="Campaign day")
        fig.update_yaxes(title_text="Samples collected")
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        fig = go.Figure()
        for outbreak, group in surveillance.groupby("outbreak"):
            fig.add_trace(go.Scatter(
                x=group["campaign_day"], y=group["pct_cohort_cleared"],
                name=outbreak.split(" 2026")[0], mode="lines+markers",
                line=dict(color=OUTBREAK_COLORS[outbreak], width=2.5)))
        fig.add_hline(y=90, line=dict(color=C["muted"], dash="dot"))
        fig.add_vrect(x0=0.5, x1=14, fillcolor=C["good"], opacity=0.07, line_width=0,
                      annotation_text="incubation window", annotation_position="bottom right")
        fig.update_layout(height=360, title="Cumulative cohort clearance", **PLOT_LAYOUT)
        fig.update_xaxes(title_text="Campaign day")
        fig.update_yaxes(title_text="% cleared", range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        tracing[["outbreak", "confirmed_cases", "contacts_traced", "contacts_per_case",
                 "campaign_days", "samples_per_day", "secondary_positives",
                 "symptomatic_contacts"]],
        use_container_width=True, hide_index=True,
    )

    insight(
        "<b>Both Indian campaigns cleared their entire cohort inside the 4–14 day incubation "
        "window</b> — West Bengal in 7 days at 27.4 samples/day, Kerala in 10 at 10.4/day, "
        "with <b>zero idle days</b> and activation 0–1 days after confirmation. West Bengal's "
        "steeper curve is not better work: a nosocomial cluster hands investigators a "
        "pre-enumerated cohort from duty rosters, while Kerala had to <i>discover</i> contacts "
        "through community investigation. <b>Plan ~100 contacts per confirmed case.</b> "
        "Bangladesh's 35 is a tracing shortfall, not a smaller outbreak — the patient died "
        "before anyone could interview them."
    )

    st.divider()
    st.markdown("#### Who was exposed?")

    e_left, e_right = st.columns([1.3, 1])

    with e_left:
        share = pd.crosstab(contacts["outbreak"], contacts["contact_type"],
                            normalize="index") * 100
        fig = go.Figure()
        type_colors = {"Healthcare Worker": C["india"], "Co-patient (Nosocomial)": C["accent"],
                       "Community": C["bangladesh"], "Family/Household": C["warn"],
                       "Workplace": C["good"]}
        for ctype in share.columns:
            fig.add_trace(go.Bar(
                y=[o.split(" 2026")[0] for o in share.index], x=share[ctype],
                name=ctype, orientation="h",
                marker_color=type_colors.get(ctype, C["muted"]),
                hovertemplate="%{y} · " + ctype + ": %{x:.1f}%<extra></extra>"))
        fig.update_layout(barmode="stack", height=340,
                          title="Exposure setting by outbreak (%)",
                          **{**PLOT_LAYOUT, "hovermode": "closest"})
        fig.update_xaxes(title_text="% of traced contacts")
        st.plotly_chart(fig, use_container_width=True)

    with e_right:
        fig = px.histogram(contacts, x="age", color="outbreak", nbins=18,
                           color_discrete_map=OUTBREAK_COLORS, opacity=0.7)
        fig.update_layout(height=340, title="Age of traced contacts",
                          **{**PLOT_LAYOUT, "hovermode": "closest"})
        fig.update_xaxes(title_text="Age (years)")
        fig.update_yaxes(title_text="Contacts")
        st.plotly_chart(fig, use_container_width=True)

    hospital = contacts[
        (contacts["outbreak"].str.contains("West Bengal"))
        & (contacts["contact_type"].isin(["Healthcare Worker", "Co-patient (Nosocomial)"]))]
    insight(
        f"<b>{len(hospital)} of 192 West Bengal contacts ({len(hospital) / 192 * 100:.1f}%) "
        "were hospital-linked</b> — healthcare workers plus co-patients — confirming the "
        "stated nosocomial route. Kerala's profile is entirely different (no co-patients, "
        "21% workplace), matching an environmental index case. Bangladesh's cohort is 100% "
        "community. <b>The traced cohorts independently corroborate each outbreak's stated "
        "exposure history</b>, which is a meaningful internal validation of the dataset."
    )

    kerala = contacts[contacts["outbreak"].str.contains("Kerala")].copy()
    if len(kerala) and kerala["risk_category"].notna().any():
        st.markdown("#### Did risk stratification drive testing priority?")
        kerala["campaign_day"] = (
            kerala["date_sample"] - kerala["date_sample"].min()).dt.days + 1
        order = ["Very High Risk", "High Risk", "Low Risk"]
        risk_colors = {"Very High Risk": C["death"], "High Risk": C["warn"],
                       "Low Risk": C["good"]}

        fig = go.Figure()
        for risk in order:
            sub = kerala.loc[kerala["risk_category"] == risk, "campaign_day"]
            if len(sub):
                fig.add_trace(go.Box(y=sub, name=risk.replace(" Risk", ""),
                                     marker_color=risk_colors[risk],
                                     boxpoints="all", jitter=0.45, pointpos=-1.6))
        fig.update_layout(height=330, title="Campaign day sampled, by WHO risk category",
                          **{**PLOT_LAYOUT, "hovermode": "closest"})
        fig.update_yaxes(title_text="Campaign day")
        st.plotly_chart(fig, use_container_width=True)

        groups = [kerala.loc[kerala["risk_category"] == r, "campaign_day"].to_numpy()
                  for r in order]
        groups = [g for g in groups if len(g)]
        if len(groups) >= 2:
            h_stat, p_kw = stats.kruskal(*groups)
            vh = kerala.loc[kerala["risk_category"] == "Very High Risk", "campaign_day"]
            lo = kerala.loc[kerala["risk_category"] == "Low Risk", "campaign_day"]
            insight(
                f"<b>Kruskal-Wallis p = {p_kw:.3f} — risk category did not drive sampling "
                f"order</b>, and the sign runs the wrong way: the {len(vh)} Very High Risk "
                f"contacts were sampled on average <b>{vh.mean() - lo.mean():+.2f} days "
                "later</b> than Low Risk ones. It did not matter here — 104 contacts clear "
                "comfortably in 10 days regardless of order, and the result was 0 positives. "
                "But <b>a cohort of 500 cannot</b>, and a prioritisation protocol never "
                "exercised on a small outbreak will not materialise on a large one. Making "
                "risk the explicit sort key is a zero-cost protocol change.",
                kind="warn-box",
            )

# ==========================================================================
# TAB 5 — POLYNOMIAL MODEL
# ==========================================================================
with TAB_MODEL:
    st.subheader("Polynomial regression — what can actually be forecast?")

    summary, curves, grid, registry = load_model_artefacts()

    st.markdown(
        "Each model searches **degree (1–6) × ridge penalty × variance-stabilising "
        "transform**, selected by leave-one-out cross-validation and always scored back on "
        "the original scale. Every one is then compared against a **“predict the historical "
        "mean” baseline** — the bar any trend model must clear to be worth using."
    )

    beats = int(summary["beats_mean"].sum())
    n_models = len(summary)
    s1, s2, s3 = st.columns(3)
    s1.metric("Models trained", n_models)
    s2.metric("Beat the mean baseline", f"{beats} / {n_models}",
              f"{n_models - beats} failed", delta_color="inverse")
    s3.metric("Best CV R²", f"{summary['cv_r2'].max():+.3f}",
              summary.loc[summary["cv_r2"].idxmax(), "target"], delta_color="off")

    fig = go.Figure()
    sc = summary.sort_values("cv_r2")
    fig.add_trace(go.Bar(
        x=sc["cv_r2"], y=sc["target"], orientation="h",
        marker_color=[C["good"] if v > 0 else C["death"] for v in sc["cv_r2"]],
        text=[f"{v:+.3f}" for v in sc["cv_r2"]], textposition="outside",
        hovertemplate="%{y}: CV R² = %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=C["dark"], width=1.5))
    fig.update_layout(height=380, title="Cross-validated R² by target",
                      **{**PLOT_LAYOUT, "hovermode": "closest"})
    fig.update_xaxes(title_text="CV R²  (negative = worse than predicting the mean)")
    st.plotly_chart(fig, use_container_width=True)

    insight(
        f"<b>{n_models - beats} of {n_models} models fail to beat the historical mean.</b> "
        "That is not a defect — it is the finding. The autocorrelation diagnostic already "
        "showed the series has no memory and 26:1 overdispersion, so year-to-year variation "
        "<i>is</i> noise. The two targets that do carry signal are revealing: the "
        "<b>smoothed trend</b> (confirming the two-wave shape is real) and <b>outbreak size "
        "given an outbreak occurred</b>. Separating <i>whether</i> an outbreak happens from "
        "<i>how big</i> it gets makes one of the two components tractable; modelling raw "
        "annual counts blends them and predicts neither."
    )

    st.divider()

    target = st.selectbox(
        "Inspect a model", summary["target"].tolist(),
        format_func=lambda t: f"{t} — "
                              f"{summary.loc[summary['target'] == t, 'label'].iloc[0]}",
    )
    row = summary[summary["target"] == target].iloc[0]
    curve = curves[curves["target"] == target]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Degree", int(row["degree"]), f"transform: {row['transform']}",
              delta_color="off")
    d2.metric("CV R²", f"{row['cv_r2']:+.3f}",
              "beats mean" if row["beats_mean"] else "worse than mean",
              delta_color="normal" if row["beats_mean"] else "inverse")
    d3.metric("CV RMSE", f"{row['cv_rmse']:.2f}",
              f"baseline {row['baseline_rmse']:.2f}", delta_color="off")
    d4.metric("Train R²", f"{row['train_r2']:+.3f}",
              f"adj. {row['adj_r2']:+.3f}", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=curve["year"], y=curve["observed"], name="Observed",
        marker_color=C["muted"], opacity=0.5))
    fig.add_trace(go.Scatter(
        x=curve["year"], y=curve["fitted"], name=f"Polynomial fit (deg {int(row['degree'])})",
        mode="lines", line=dict(color=C["accent"], width=3)))
    fig.add_trace(go.Scatter(
        x=curve["year"], y=curve["loo_pred"], name="Leave-one-out prediction",
        mode="markers", marker=dict(color=C["death"], size=7, symbol="circle-open",
                                    line=dict(width=2))))
    fig.add_hline(y=curve["observed"].mean(), line=dict(color=C["dark"], dash="dot"),
                  annotation_text=f"mean = {curve['observed'].mean():.1f}")

    if bool(row["forecastable"]) and pd.notna(row["next_year_pred"]):
        fig.add_trace(go.Scatter(
            x=[row["next_year"]], y=[row["next_year_pred"]], name=f"{int(row['next_year'])} projection",
            mode="markers", marker=dict(color=C["death"], size=13, symbol="diamond"),
            error_y=dict(type="data", symmetric=False,
                         array=[row["next_year_hi95"] - row["next_year_pred"]],
                         arrayminus=[row["next_year_pred"] - row["next_year_lo95"]],
                         color=C["death"], thickness=2, width=8)))

    fig.update_layout(height=420, title=row["label"], **PLOT_LAYOUT)
    fig.update_yaxes(title_text="Value")
    st.plotly_chart(fig, use_container_width=True)

    g_left, g_right = st.columns(2)

    with g_left:
        sub = grid[grid["target"] == target].groupby("degree")["cv_rmse"].min()
        fig = go.Figure(go.Scatter(
            x=sub.index, y=sub.to_numpy(), mode="lines+markers",
            line=dict(color=C["accent"], width=2.5),
            marker=dict(size=9),
            hovertemplate="degree %{x}: CV RMSE %{y:.2f}<extra></extra>"))
        fig.add_hline(y=row["baseline_rmse"], line=dict(color=C["death"], dash="dash"),
                      annotation_text="mean baseline")
        fig.update_layout(height=330, title="Degree selection (best alpha per degree)",
                          **{**PLOT_LAYOUT, "hovermode": "closest"})
        fig.update_xaxes(title_text="Polynomial degree")
        fig.update_yaxes(title_text="CV RMSE")
        st.plotly_chart(fig, use_container_width=True)

    with g_right:
        fig = go.Figure(go.Scatter(
            x=curve["fitted"], y=curve["residual"], mode="markers",
            marker=dict(color=C["case"], size=9, line=dict(color=C["dark"], width=1)),
            text=curve["year"],
            hovertemplate="%{text}: fitted %{x:.1f}, residual %{y:.1f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=C["dark"], width=1.5))
        fig.update_layout(height=330, title="Residuals vs fitted",
                          **{**PLOT_LAYOUT, "hovermode": "closest"})
        fig.update_xaxes(title_text="Fitted value")
        fig.update_yaxes(title_text="Residual")
        st.plotly_chart(fig, use_container_width=True)

    if not bool(row["forecastable"]):
        insight(
            "<b>This target is deliberately excluded from forecasting.</b> It is a centred "
            "moving average, so its final points are already built from truncated windows — "
            "extrapolating them would produce an artefact rather than a projection. Its value "
            "is confirming that the smoothed two-wave shape is real structure, not a "
            "smoothing artefact.",
            kind="warn-box",
        )
    else:
        insight(
            f"<b>{int(row['next_year'])} projection: {row['next_year_pred']:.1f}</b> with a 95% "
            f"prediction interval of <b>{row['next_year_lo95']:.1f} – "
            f"{row['next_year_hi95']:.1f}</b>. That interval covers nearly the whole historical "
            "range — a plan that must work across it is not being informed by this model. "
            + ("<b>This model does not beat the mean baseline, so its point estimate should "
               "not be used.</b> " if not row["beats_mean"] else "")
            + ("Degree ≥ 4 also means the curve bends violently outside the fitted window. "
               if row["extrapolation_risk"] else "")
            + "All models fit 1998–2026 as one homogeneous process, but the structural-break "
              "scan shows it is not — so the estimate is pulled upward by a regime that "
              "ended a decade ago.",
            kind="warn-box",
        )

    st.divider()
    st.markdown("#### What number should actually go into the 2027 plan?")
    st.markdown(
        """
        **Not the model's.** The defensible figures come from the descriptive analysis:

        | Quantity | Value | Source |
        |---|---|---|
        | Expected cases in 2027 | **~6–7** | Post-2015 regime mean (6.45/yr) |
        | Probability 2027 is active | **~80%** | Recent activity rate |
        | Expected deaths | **~5** | 6.5 cases × 73.8% pooled CFR |
        | Surge design point | **30–40 cases** | Post-break maximum + headroom |
        | Contacts to trace per case | **~100** | 2026 outbreak investigations |

        The polynomial models earn their place by establishing what is **not** knowable:
        *the level of any individual year is not predictable from calendar time.* That
        result protects a planner from a false-precision forecast, and it is worth more
        than a confident number would have been.
        """
    )

    with st.expander("Model registry (models/model_registry.json)"):
        st.json(registry)

# ==========================================================================
# TAB 6 — DATA
# ==========================================================================
with TAB_DATA:
    st.subheader("Analytical tables")

    frames = {
        "Annual time series": annual_full,
        "Annual by country": by_country_full,
        "Outbreak events (all countries)": events,
        "2026 case timelines": cases,
        "2026 contacts (harmonised)": contacts,
        "Surveillance daily curve": surveillance,
        "Tracing efficiency": tracing,
    }

    choice = st.selectbox("Table", list(frames))
    frame = frames[choice]

    st.caption(f"{len(frame):,} rows × {frame.shape[1]} columns")
    st.dataframe(frame, use_container_width=True, height=440)
    st.download_button(
        f"Download {choice.lower().replace(' ', '_')}.csv",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown(
        """
        #### Data-quality notes carried through the analysis

        - **Nine silent years** (1998–2000, 2002, 2006, 2009, 2016, 2017, 2020) have no row
          in the raw file. They are re-inserted as explicit zeros, because a silent year is a
          year with zero reported cases — not a year that did not happen.
        - **`Date_Death` is empty for survivors.** That absence *is* the outcome; imputing it
          would invent deaths.
        - **`BD-2026-001` died on 2026-01-28 but was confirmed on 2026-01-29** — laboratory
          confirmation arrived one day after death. The negative interval is retained because
          it measures something real.
        - **Bangladesh contacts carry no sampling dates**, so that outbreak appears in the
          totals but not in any daily surveillance curve.
        - **CFR is always recomputed** from deaths ÷ cases rather than read from the stored
          column, and pooled ratios are patient-weighted, never a mean of yearly ratios.

        #### Project layout

        ```
        input/    six raw CSVs from the Kaggle dataset
        src/      nipah_data.py (shared loaders) · train_models.py (polynomial models)
        notebook/ nipah_time_series_analysis.ipynb — 15 business questions, full commentary
        models/   seven serialised models + model_registry.json
        output/   exported analytical tables and model audit files
        img/      every figure from the notebook
        app/      this dashboard
        ```
        """
    )

st.divider()
st.caption(
    "Nipah Virus Outbreaks 2025–2026 · India & Bangladesh — built on WHO Disease Outbreak "
    "News, CDC and peer-reviewed outbreak reports. Dashboard and notebook share "
    "`src/nipah_data.py`, so every number shown here matches the analysis notebook."
)
