import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import shap


# -----------------------------
# Page config (make it feel fancy)
# -----------------------------
st.set_page_config(
    page_title="SSP Risk Explorer — Poverty Multiline",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Slightly tighter top padding */
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* Make sidebar a bit nicer */
      section[data-testid="stSidebar"] { padding-top: 1.2rem; }

      /* Metric cards spacing */
      div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 16px;
        border-radius: 16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data loading
# -----------------------------
DEFAULT_PATH = "C:\\Users\\miche\\Desktop\\SSP Risk Explorer\\project-ssp-risk-explorer\\data\\raw\\dummy\\dashboard_ready_dummy_ssp_poverty_multiline.csv"  # put next to app.py

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleanup
    df["year"] = df["year"].astype(int)
    df["scenario"] = df["scenario"].astype(str)
    return df

# Sidebar: data source
with st.sidebar:
    st.title("⚙️ Controls")
    data_path = st.text_input("Data path", value=DEFAULT_PATH, help="CSV path. If relative, place it next to app.py")

df = load_data(data_path)

# Core columns (from your dummy file)
ID_COLS = ["iso3", "country", "scenario", "year"]
POVERTY_COLS = [c for c in df.columns if c.startswith("poverty_")]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + POVERTY_COLS]


# -----------------------------
# Build a continent mapping (for regional trends)
# Uses plotly's gapminder mapping; unmatched countries become "Other"
# -----------------------------
@st.cache_data
def add_continent(df_in: pd.DataFrame) -> pd.DataFrame:
    gm = px.data.gapminder()[["iso_alpha", "continent"]].drop_duplicates()
    out = df_in.merge(gm, left_on="iso3", right_on="iso_alpha", how="left")
    out["continent"] = out["continent"].fillna("Other")
    out = out.drop(columns=["iso_alpha"])
    return out

df = add_continent(df)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    scenario = st.segmented_control(
        "Scenario",
        options=["SSP1", "SSP2", "SSP3", "SSP5"],
        default="SSP2",
        key="scenario_selector",
    )

    available_years = sorted(df["year"].unique().tolist())
    # Typical years – if dataset differs, we intersect with what's available
    preferred_years = [2030, 2050, 2100]
    year_options = [y for y in preferred_years if y in available_years] or available_years

    year = st.segmented_control(
        "Year",
        options=year_options,
        default=year_options[0],
        key="year_selector",
    )

    poverty_metric = st.selectbox(
        "Risk metric (poverty line)",
        options=POVERTY_COLS,
        index=0,
        help="Used for map coloring + charts + SHAP model target",
    )

    map_scale = st.selectbox(
        "Map color scale",
        options=["Continuous", "Categorical risk bands"],
        index=0,
    )

    # Optional focus
    continents = ["Global"] + sorted(df["continent"].unique().tolist())
    region_focus = st.selectbox("Region focus", options=continents, index=0)

    show_top_n = st.slider("Show top N countries in rankings", 5, 30, 10, 1)

    st.divider()
    st.caption("Tip: This app reruns on every interaction; caching keeps it snappy.")


# -----------------------------
# Filtered slices
# -----------------------------
base = df[(df["scenario"] == scenario) & (df["year"] == year)].copy()
if region_focus != "Global":
    base = base[base["continent"] == region_focus].copy()

# Handle missing values gracefully for metric
base_metric = base.dropna(subset=[poverty_metric]).copy()

# Population-weighted mean (if population exists)
def pop_weighted_mean(d: pd.DataFrame, value_col: str, weight_col: str = "population") -> float | None:
    if weight_col not in d.columns:
        return None
    x = d[[value_col, weight_col]].dropna()
    if x.empty:
        return None
    w = x[weight_col].astype(float).values
    v = x[value_col].astype(float).values
    if np.sum(w) == 0:
        return float(np.nanmean(v))
    return float(np.sum(v * w) / np.sum(w))


# -----------------------------
# Header + KPIs
# -----------------------------
title_left, title_right = st.columns([3, 2])
with title_left:
    st.title("🌍 SSP Risk Explorer — Poverty Multiline")
    st.caption("Interactive scenario/year exploration with global mapping, comparisons, trends, and explainability.")

with title_right:
    st.markdown(
        f"""
        <div style="text-align:right; padding-top: 6px;">
          <div style="font-size: 12px; opacity: 0.75;">Current view</div>
          <div style="font-size: 20px; font-weight: 650;">{scenario} · {year} · {poverty_metric}</div>
          <div style="font-size: 12px; opacity: 0.7;">Region: {region_focus}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

mean_val = float(base_metric[poverty_metric].mean()) if not base_metric.empty else np.nan
median_val = float(base_metric[poverty_metric].median()) if not base_metric.empty else np.nan
pw_mean = pop_weighted_mean(base_metric, poverty_metric)

# "High risk share" (e.g., > 25% for a poverty metric; arbitrary but useful)
high_risk_threshold = 25.0
high_risk_share = (
    float((base_metric[poverty_metric] > high_risk_threshold).mean() * 100.0)
    if not base_metric.empty else np.nan
)

kpi1.metric("Avg risk (%)", f"{mean_val:,.2f}" if np.isfinite(mean_val) else "—")
kpi2.metric("Median risk (%)", f"{median_val:,.2f}" if np.isfinite(median_val) else "—")
kpi3.metric("Pop-weighted avg (%)", f"{pw_mean:,.2f}" if (pw_mean is not None and np.isfinite(pw_mean)) else "—")
kpi4.metric(f"Share > {high_risk_threshold:.0f}% (%)", f"{high_risk_share:,.1f}" if np.isfinite(high_risk_share) else "—")


# -----------------------------
# Tabs: Map / Comparisons / Trends / Explainability
# -----------------------------
tab_map, tab_compare, tab_trends, tab_xai = st.tabs(["🗺️ Map", "📊 Scenario Comparison", "📈 Regional Trends", "🧠 Explainability (SHAP)"])


# -----------------------------
# MAP TAB
# -----------------------------
with tab_map:
    left, right = st.columns([2.2, 1])

    with left:
        if base_metric.empty:
            st.warning("No data for the selected filters.")
        else:
            # Optional risk categorization
            if map_scale == "Categorical risk bands":
                # You can tweak bins to your domain
                bins = [-np.inf, 5, 15, 30, 50, np.inf]
                labels = ["Very Low", "Low", "Moderate", "High", "Extreme"]
                base_metric["risk_band"] = pd.cut(base_metric[poverty_metric], bins=bins, labels=labels)

                fig = px.choropleth(
                    base_metric,
                    locations="iso3",
                    color="risk_band",
                    hover_name="country",
                    category_orders={"risk_band": labels},
                    title=f"Risk bands by country — {scenario} {year}",
                )
            else:
                fig = px.choropleth(
                    base_metric,
                    locations="iso3",
                    color=poverty_metric,
                    hover_name="country",
                    title=f"{poverty_metric} by country — {scenario} {year}",
                    color_continuous_scale="Viridis",
                )

            fig.update_layout(
                height=620,
                margin=dict(l=0, r=0, t=60, b=0),
                title=dict(x=0.02),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("📌 Country ranking")
        if base_metric.empty:
            st.info("No ranking available for the current filters.")
        else:
            top = base_metric.sort_values(poverty_metric, ascending=False).head(show_top_n)[["country", "iso3", poverty_metric]]
            st.dataframe(top, use_container_width=True, hide_index=True)

        st.subheader("🔎 Quick data peek")
        st.dataframe(
            base_metric[["country", "iso3", "continent", poverty_metric] + FEATURE_COLS].head(12),
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------
# COMPARISON TAB
# -----------------------------
with tab_compare:
    st.subheader("Scenario comparison at selected year")

    # Compare all scenarios at selected year (and optional region)
    comp = df[df["year"] == year].copy()
    if region_focus != "Global":
        comp = comp[comp["continent"] == region_focus].copy()
    comp = comp.dropna(subset=[poverty_metric])

    c1, c2 = st.columns([1.4, 1])

    with c1:
        # Global distribution by scenario
        fig = px.box(
            comp,
            x="scenario",
            y=poverty_metric,
            points="outliers",
            title=f"Distribution of {poverty_metric} across scenarios ({year})",
        )
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=60, b=0), title=dict(x=0.02))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Average by scenario (optionally pop-weighted)
        agg_rows = []
        for s in ["SSP1", "SSP2", "SSP3", "SSP5"]:
            d = comp[comp["scenario"] == s]
            agg_rows.append(
                {
                    "scenario": s,
                    "mean": float(d[poverty_metric].mean()) if not d.empty else np.nan,
                    "pop_weighted_mean": pop_weighted_mean(d, poverty_metric),
                }
            )
        agg = pd.DataFrame(agg_rows)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["scenario"], y=agg["mean"], name="Mean"))
        if agg["pop_weighted_mean"].notna().any():
            fig.add_trace(go.Bar(x=agg["scenario"], y=agg["pop_weighted_mean"], name="Pop-weighted mean"))

        fig.update_layout(
            barmode="group",
            title=f"Scenario averages — {poverty_metric} ({year})",
            height=460,
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Country spotlight (optional)")
    country_options = sorted(df["country"].unique().tolist())
    selected_country = st.selectbox("Pick a country", options=country_options, index=country_options.index("Angola") if "Angola" in country_options else 0)

    spotlight = df[df["country"] == selected_country].copy()
    if region_focus != "Global":
        spotlight = spotlight[spotlight["continent"] == region_focus].copy()

    fig = px.line(
        spotlight,
        x="year",
        y=poverty_metric,
        color="scenario",
        markers=True,
        title=f"{selected_country}: {poverty_metric} over time by scenario",
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=60, b=0), title=dict(x=0.02))
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# TRENDS TAB
# -----------------------------
with tab_trends:
    st.subheader("Regional trend lines (continent aggregates)")

    trend = df.copy()
    if region_focus != "Global":
        # If user chose a region focus, still show that region's trends (by scenario)
        trend = trend[trend["continent"] == region_focus].copy()

    trend = trend.dropna(subset=[poverty_metric])

    # Aggregate by continent/year/scenario
    # Use population-weighted mean if available, else mean
    if "population" in trend.columns:
        g = trend.dropna(subset=["population"]).copy()
        g["w"] = g["population"].astype(float)
        g["v"] = g[poverty_metric].astype(float)
        regional = (
            g.groupby(["continent", "scenario", "year"], as_index=False)
            .apply(lambda x: np.sum(x["v"] * x["w"]) / np.sum(x["w"]) if np.sum(x["w"]) else np.nan)
            .rename(columns={None: "risk_pw_mean"})
        )
        ycol = "risk_pw_mean"
        yname = "Pop-weighted mean risk (%)"
    else:
        regional = trend.groupby(["continent", "scenario", "year"], as_index=False)[poverty_metric].mean()
        ycol = poverty_metric
        yname = "Mean risk (%)"

    c1, c2 = st.columns([1.15, 1])

    with c1:
        # Pick a continent to spotlight (default: user's focus or "Europe" if exists)
        conts = sorted(regional["continent"].unique().tolist())
        default_cont = region_focus if region_focus != "Global" else ("Europe" if "Europe" in conts else conts[0])
        chosen_cont = st.selectbox("Spotlight continent", conts, index=conts.index(default_cont))

        sub = regional[regional["continent"] == chosen_cont].copy()
        fig = px.line(
            sub,
            x="year",
            y=ycol,
            color="scenario",
            markers=True,
            title=f"{chosen_cont}: {yname} over time",
        )
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=60, b=0), title=dict(x=0.02))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Small multiples feel: show one scenario, all continents
        chosen_scenario = st.selectbox("Scenario for multi-continent view", ["SSP1", "SSP2", "SSP3", "SSP5"], index=1)
        sub = regional[regional["scenario"] == chosen_scenario].copy()
        fig = px.line(
            sub,
            x="year",
            y=ycol,
            color="continent",
            markers=True,
            title=f"{chosen_scenario}: {yname} by continent",
        )
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=60, b=0), title=dict(x=0.02))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: continent mapping comes from Plotly’s built-in Gapminder dataset; unmatched countries are labeled 'Other'.")


# -----------------------------
# XAI TAB (SHAP)
# -----------------------------
with tab_xai:
    st.subheader("Variable importance (SHAP)")

    st.info(
        "This tab trains a quick tree model to predict the selected poverty metric from the available features "
        "and shows SHAP-based importance. It’s a demonstration pipeline for your real model."
    )

    # Prepare modeling data (across all scenarios/years/regions, or optionally constrained)
    model_scope = st.radio(
        "Model scope",
        options=["Use full dataset (recommended)", "Use only current region focus"],
        horizontal=True,
        index=0,
    )

    model_df = df.copy()
    if model_scope == "Use only current region focus" and region_focus != "Global":
        model_df = model_df[model_df["continent"] == region_focus].copy()

    model_df = model_df.dropna(subset=[poverty_metric]).copy()

    # Build X/y
    X = model_df[FEATURE_COLS].copy()
    # basic NA handling
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = model_df[poverty_metric].astype(float).values

    if len(model_df) < 200:
        st.warning("Not much data available for SHAP modeling under the current scope. Try 'full dataset'.")
    else:
        @st.cache_resource
        def train_model(feature_cols: tuple, random_state: int = 7):
            # Train a strong-enough baseline model quickly
            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
            return model

        model = train_model(tuple(FEATURE_COLS))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=7)

        model.fit(X_train, y_train)

        # Sample rows for SHAP speed
        sample_n = min(1200, len(X_test))
        X_shap = X_test.sample(sample_n, random_state=7)

        @st.cache_data
        def compute_shap_values(model_bytes: bytes, X_in: pd.DataFrame):
            # We can’t hash the model object well; so pass bytes marker + use cache_data lightly.
            # Streamlit will still re-run if code changes.
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_in)
            base_val = explainer.expected_value
            return sv, base_val

        # "marker" for caching; not perfect but okay for this demo
        model_marker = str(model.get_params()).encode("utf-8")

        shap_values, expected_value = compute_shap_values(model_marker, X_shap)

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("#### 🔥 Global importance (bar)")
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            shap.summary_plot(
                shap_values,
                X_shap,
                plot_type="bar",
                show=False,
                max_display=min(12, len(FEATURE_COLS)),
            )
            st.pyplot(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🐝 SHAP summary (beeswarm)")
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            shap.summary_plot(
                shap_values,
                X_shap,
                show=False,
                max_display=min(12, len(FEATURE_COLS)),
            )
            st.pyplot(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Model quick check")
        y_pred = model.predict(X_test)
        mae = float(np.mean(np.abs(y_test - y_pred)))
        st.metric("MAE (on held-out test split)", f"{mae:,.3f}")

        with st.expander("See feature list used for modeling"):
            st.write(FEATURE_COLS)
