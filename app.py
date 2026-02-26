import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Model Builder",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏈 NFL Prediction Model Builder")
st.markdown("Select features, choose a target, and train a model to predict NFL game outcomes.")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("games_data_nfl.csv")
    return df

df = load_data()

# ── Column groups ──────────────────────────────────────────────────────────────
TARGET_COLS = ["team_score", "total", "result"]

# All numeric columns minus targets and ID/metadata columns
EXCLUDE = {
    "game_id", "old_game_id", "gsis", "nfl_detail_id", "pfr", "pff", "espn",
    "ftn", "stadium_id",
    # exclude the other two targets so they can't be features for the chosen target
}
EXCLUDE.update(TARGET_COLS)

FEATURE_GROUPS = {
    "Game Context": ["season", "week", "home", "div_game", "overtime"],
    "Rest & Schedule": ["rest", "opponent_rest"],
    "Weather": ["temp", "wind"],
    "Betting Lines": [
        "moneyline", "opponent_moneyline", "spread_line",
        "spread_odds", "opponent_spread_odds",
        "total_line", "under_odds", "over_odds",
        "opponent_implied", "implied",
    ],
    "Offensive EPA": ["run_epa", "pass_epa", "total_run_epa", "total_pass_epa"],
    "Defensive EPA": [
        "run_epa_defense", "pass_epa_defense",
        "total_run_epa_defense", "total_pass_epa_defense",
    ],
    "Play Counts & Pace": [
        "run_plays", "pass_plays", "pass_share",
        "run_plays_defense", "pass_plays_defense", "pass_share_defense",
        "pace", "pace_defense",
    ],
    "Roof / Surface": [
        "roofclosed", "roofdome", "roofopen", "roofoutdoors",
        "surfacea_turf", "surfaceastroturf", "surfacefieldturf",
        "surfacegrass", "surfacematrixturf", "surfacesportturf",
    ],
}

# Only keep columns that actually exist in the dataframe
for group in FEATURE_GROUPS:
    FEATURE_GROUPS[group] = [c for c in FEATURE_GROUPS[group] if c in df.columns]

ALL_NUMERIC = sorted(
    [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE and c in df.columns
    ]
)

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

# Market baseline for each target: the betting-market's best guess at that number
MARKET_BASELINES = {
    "team_score": "implied",       # implied team points from moneyline/total
    "total":      "total_line",    # over/under line
    "result":     "spread_line",   # spread (market's predicted point diff)
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Configuration")

    # Target
    st.subheader("1. Target Variable")
    target = st.selectbox(
        "What do you want to predict?",
        options=TARGET_COLS,
        format_func=lambda x: {
            "team_score": "Team Score (points scored)",
            "total": "Total Points (combined score)",
            "result": "Result (point differential)",
        }[x],
    )

    # Model type
    st.subheader("2. Model Type")
    model_name = st.selectbox("Algorithm", list(MODELS.keys()))

    # Train/test split
    st.subheader("3. Train / Test Split")
    test_size = st.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    # Season filter
    st.subheader("4. Season Filter")
    seasons = sorted(df["season"].dropna().unique().astype(int).tolist())
    selected_seasons = st.multiselect(
        "Include seasons (leave blank for all)",
        options=seasons,
        default=[],
    )

    st.divider()
    st.caption("Select features in the main panel, then click **Train Model**.")

# ── Feature selection ──────────────────────────────────────────────────────────
st.header("Select Features")

# Quick-pick buttons row
col_a, col_b, col_c = st.columns(3)
with col_a:
    betting_quick = st.checkbox("All Betting Lines", value=False)
with col_b:
    epa_quick = st.checkbox("All EPA Stats", value=False)
with col_c:
    context_quick = st.checkbox("Game Context", value=True)

# Build default selections from quick-picks
default_features = set()
if betting_quick:
    default_features.update(FEATURE_GROUPS.get("Betting Lines", []))
if epa_quick:
    default_features.update(FEATURE_GROUPS.get("Offensive EPA", []))
    default_features.update(FEATURE_GROUPS.get("Defensive EPA", []))
if context_quick:
    default_features.update(FEATURE_GROUPS.get("Game Context", []))

# Remove the target from defaults just in case
default_features.discard(target)

tab_labels = list(FEATURE_GROUPS.keys()) + ["All Numeric"]
tabs = st.tabs(tab_labels)

selected_features = set()

for tab, (group_name, group_cols) in zip(tabs[:-1], FEATURE_GROUPS.items()):
    with tab:
        if not group_cols:
            st.caption("No columns available for this group.")
            continue
        # filter out the current target
        available = [c for c in group_cols if c != target]
        defaults = [c for c in available if c in default_features]
        chosen = st.multiselect(
            f"Select {group_name} features",
            options=available,
            default=defaults,
            key=f"group_{group_name}",
        )
        selected_features.update(chosen)

with tabs[-1]:
    available_all = [c for c in ALL_NUMERIC if c != target]
    extra = st.multiselect(
        "Pick any numeric column",
        options=available_all,
        default=[],
        key="all_numeric",
    )
    selected_features.update(extra)

# Consolidated review — lets users see all selected features and remove any
all_selected_sorted = sorted(selected_features)
if all_selected_sorted:
    st.markdown("**Selected Features** — click × on any tag to remove it:")
    feature_list = st.multiselect(
        "selected_features_review",
        options=all_selected_sorted,
        default=all_selected_sorted,
        key="feature_review",
        label_visibility="collapsed",
    )
else:
    feature_list = []
    st.warning("No features selected yet. Use the tabs above to select predictor columns.")

# ── Train ──────────────────────────────────────────────────────────────────────
st.divider()
train_btn = st.button("🚀 Train Model", type="primary", disabled=len(feature_list) == 0)

if train_btn:
    # Apply season filter
    data = df.copy()
    if selected_seasons:
        data = data[data["season"].isin(selected_seasons)]

    # Stash baseline values before we narrow the columns (so they survive dropna on features)
    baseline_col = MARKET_BASELINES.get(target)
    baseline_full = data[baseline_col].copy() if baseline_col and baseline_col in data.columns else None

    # Drop rows missing the target or any selected feature
    cols_needed = feature_list + [target]
    data = data[cols_needed].dropna()

    if len(data) < 50:
        st.error("Not enough rows after filtering. Try including more seasons or fewer required features.")
        st.stop()

    X = data[feature_list]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scale for linear models
    scaler = None
    model = MODELS[model_name]
    if model_name in ("Linear Regression", "Ridge Regression", "Lasso Regression"):
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
    else:
        X_train_fit = X_train.values
        X_test_fit = X_test.values

    model.fit(X_train_fit, y_train)
    y_pred = model.predict(X_test_fit)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ── Market baseline metrics ──────────────────────────────────────────────────
    b_rmse = b_mae = b_r2 = None
    if baseline_full is not None:
        baseline_test = baseline_full.loc[y_test.index]
        valid = baseline_test.notna()
        if valid.sum() > 10:
            bt = baseline_test[valid].values
            yt = y_test[valid].values
            b_rmse = np.sqrt(mean_squared_error(yt, bt))
            b_mae  = mean_absolute_error(yt, bt)
            b_r2   = r2_score(yt, bt)

    # ── Results layout ──────────────────────────────────────────────────────────
    st.header("📊 Model Results")
    st.caption(
        f"**{model_name}** · Target: `{target}` · "
        f"Train rows: {len(X_train):,} · Test rows: {len(X_test):,}"
    )

    # Market baseline comparison banner
    if b_rmse is not None:
        beats_rmse = rmse < b_rmse
        beats_mae  = mae  < b_mae
        n_beats = sum([beats_rmse, beats_mae])
        if n_beats == 2:
            st.success(
                f"**Model beats the market baseline ({baseline_col}) on both RMSE and MAE.** "
                f"Market RMSE: {b_rmse:.3f} → Model RMSE: {rmse:.3f} | "
                f"Market MAE: {b_mae:.3f} → Model MAE: {mae:.3f}"
            )
        elif n_beats == 1:
            st.warning(
                f"**Model beats the market on one metric.** "
                f"Market RMSE: {b_rmse:.3f} → Model RMSE: {rmse:.3f} | "
                f"Market MAE: {b_mae:.3f} → Model MAE: {mae:.3f}"
            )
        else:
            st.error(
                f"**Model does NOT beat the market baseline ({baseline_col}).** "
                f"Market RMSE: {b_rmse:.3f} → Model RMSE: {rmse:.3f} | "
                f"Market MAE: {b_mae:.3f} → Model MAE: {mae:.3f}"
            )

    # Metrics: model vs market side by side
    if b_rmse is not None:
        st.markdown("#### Model vs Market Baseline")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "R²  (Model vs Market)",
            f"{r2:.4f}",
            delta=f"{r2 - b_r2:+.4f} vs market",
            delta_color="normal",
            help="Higher is better",
        )
        mc2.metric(
            "RMSE  (Model vs Market)",
            f"{rmse:.3f}",
            delta=f"{rmse - b_rmse:+.3f} vs market",
            delta_color="inverse",
            help="Lower is better — negative delta means model wins",
        )
        mc3.metric(
            "MAE  (Model vs Market)",
            f"{mae:.3f}",
            delta=f"{mae - b_mae:+.3f} vs market",
            delta_color="inverse",
            help="Lower is better — negative delta means model wins",
        )
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.4f}", help="1.0 = perfect; 0 = no better than the mean")
        m2.metric("RMSE", f"{rmse:.3f}", help="Root Mean Squared Error (same units as target)")
        m3.metric("MAE", f"{mae:.3f}", help="Mean Absolute Error (same units as target)")

    col_left, col_right = st.columns(2)

    # Actual vs Predicted
    with col_left:
        fig_scatter = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": f"Actual {target}", "y": f"Predicted {target}"},
            title="Actual vs. Predicted",
            opacity=0.6,
            color_discrete_sequence=["#1f77b4"],
        )
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        fig_scatter.add_trace(
            go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                       line=dict(color="red", dash="dash"), name="Perfect fit")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Residuals
    with col_right:
        residuals = y_test.values - y_pred
        fig_resid = px.histogram(
            residuals,
            nbins=40,
            labels={"value": "Residual", "count": "Frequency"},
            title="Residuals Distribution",
            color_discrete_sequence=["#ff7f0e"],
        )
        fig_resid.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, use_container_width=True)

    # Feature importance / coefficients
    st.subheader("Feature Importance / Coefficients")

    if hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "Feature": feature_list,
            "Coefficient": model.coef_,
        }).sort_values("Coefficient", key=abs, ascending=False)
        fig_coef = px.bar(
            coef_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Model Coefficients (scaled)",
            color="Coefficient",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
        )
        fig_coef.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_coef, use_container_width=True)

    elif hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature": feature_list,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    # Prediction table sample
    st.subheader("Sample Predictions")
    sample = X_test.copy().reset_index(drop=True).head(20)
    sample["Actual"] = y_test.values[:20]
    sample["Predicted"] = np.round(y_pred[:20], 2)
    sample["Error"] = np.round(sample["Actual"] - sample["Predicted"], 2)
    st.dataframe(sample, use_container_width=True)

    # Download predictions
    pred_df = X_test.copy().reset_index(drop=True)
    pred_df["Actual"] = y_test.values
    pred_df["Predicted"] = np.round(y_pred, 2)
    pred_df["Error"] = pred_df["Actual"] - pred_df["Predicted"]
    csv_out = pred_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Predictions CSV",
        data=csv_out,
        file_name=f"nfl_predictions_{target}_{model_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ── Data explorer ──────────────────────────────────────────────────────────────
with st.expander("🔍 Data Explorer"):
    st.markdown(f"**{len(df):,} rows × {len(df.columns)} columns**")

    col_filter, col_season = st.columns(2)
    with col_season:
        explore_seasons = st.multiselect(
            "Filter by season", options=seasons, default=[], key="explore_season"
        )
    explore_df = df if not explore_seasons else df[df["season"].isin(explore_seasons)]

    with col_filter:
        search_col = st.selectbox("Column to inspect", options=df.columns.tolist(), key="explore_col")

    st.dataframe(explore_df[[search_col, "team", "opponent", "season", "week", target if target in df.columns else "team_score"]].head(200), use_container_width=True)

    st.markdown("**Descriptive statistics (numeric columns)**")
    st.dataframe(explore_df[ALL_NUMERIC].describe().T.round(3), use_container_width=True)
