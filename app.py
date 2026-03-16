import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import norm
import requests
import io
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Model Builder",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: #08090F;
    border-right: 1px solid #1E2235;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161B27;
    border: 1px solid #1E2235;
    border-radius: 10px;
    padding: 14px 18px;
}

/* Primary button — Streamlit 1.44+ testid */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #00AEEF 0%, #0077CC 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #17C4FF 0%, #0088EE 100%) !important;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    color: #00AEEF !important;
    border-bottom: 2px solid #00AEEF !important;
}

/* Expander headers — scoped to stExpander so it doesn't hit internal details */
[data-testid="stExpander"] summary span {
    color: #00AEEF;
    font-weight: 600;
}

/* Dividers */
hr { border-color: #1E2235; }
</style>
""", unsafe_allow_html=True)

# ── Header with logos ──────────────────────────────────────────────────────────
oj_logo = "assets/oddsjam.png"
rw_logo = "assets/rotowire.png"

left_col, title_col, right_col = st.columns([1.2, 4, 1.2])
with left_col:
    if os.path.exists(oj_logo):
        st.image(oj_logo, width=140)
with title_col:
    # Inline styles keep the gradient scoped; no external CSS class needed
    st.markdown(
        """
        <div style="text-align:center; padding:8px 0;">
          <span style="font-size:1.9rem; font-weight:700;
                       background:linear-gradient(90deg,#00AEEF,#9B30FF);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       background-clip:text;">
            🏈 NFL Prediction Model Builder
          </span>
          <p style="margin:4px 0 0; color:#7A8299; font-size:0.85rem;">
            Select features, choose a target, and train a model to predict NFL game outcomes.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right_col:
    if os.path.exists(rw_logo):
        st.image(rw_logo, width=140)

st.divider()

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

# Each entry is a callable (seed) → model instance.
# Models that don't use random_state simply ignore the seed argument.
MODEL_FACTORIES = {
    "Linear Regression":    lambda s: LinearRegression(),
    "Ridge Regression":     lambda s: Ridge(),
    "Lasso Regression":     lambda s: Lasso(),
    "ElasticNet":           lambda s: ElasticNet(max_iter=5000),
    "Random Forest":        lambda s: RandomForestRegressor(n_estimators=100, random_state=s),
    "Extra Trees":          lambda s: ExtraTreesRegressor(n_estimators=100, random_state=s),
    "Gradient Boosting":    lambda s: GradientBoostingRegressor(n_estimators=100, random_state=s),
    "AdaBoost":             lambda s: AdaBoostRegressor(n_estimators=100, random_state=s),
    "Support Vector (RBF)": lambda s: SVR(kernel="rbf"),
    "K-Nearest Neighbors":  lambda s: KNeighborsRegressor(),
}

# Models that need feature scaling (distance/magnitude sensitive)
SCALE_MODELS = {
    "Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet",
    "Support Vector (RBF)", "K-Nearest Neighbors",
}

# Hyperparameter search spaces for RandomizedSearchCV.
# Keys use the pipeline step prefix "model__" so they work with Pipeline objects.
# Models without tunable parameters (Linear Regression) are omitted.
PARAM_GRIDS = {
    "Ridge Regression":     {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression":     {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "ElasticNet":           {"model__alpha": [0.001, 0.01, 0.1, 1.0],
                             "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "Random Forest":        {"model__n_estimators": [50, 100, 200, 300],
                             "model__max_depth": [None, 5, 10, 20],
                             "model__min_samples_split": [2, 5, 10],
                             "model__max_features": ["sqrt", "log2", 0.5]},
    "Extra Trees":          {"model__n_estimators": [50, 100, 200, 300],
                             "model__max_depth": [None, 5, 10, 20],
                             "model__min_samples_split": [2, 5, 10],
                             "model__max_features": ["sqrt", "log2", 0.5]},
    "Gradient Boosting":    {"model__n_estimators": [50, 100, 200],
                             "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                             "model__max_depth": [3, 4, 5, 6],
                             "model__subsample": [0.7, 0.8, 1.0]},
    "AdaBoost":             {"model__n_estimators": [50, 100, 150, 200],
                             "model__learning_rate": [0.01, 0.1, 0.5, 1.0]},
    "Support Vector (RBF)": {"model__C": [0.1, 1.0, 10.0, 100.0],
                             "model__epsilon": [0.01, 0.1, 0.5, 1.0],
                             "model__gamma": ["scale", "auto"]},
    "K-Nearest Neighbors":  {"model__n_neighbors": [3, 5, 7, 10, 15, 20],
                             "model__weights": ["uniform", "distance"],
                             "model__metric": ["euclidean", "manhattan"]},
}

# Market baseline for each target: the betting-market's best guess at that number
MARKET_BASELINES = {
    "team_score": "implied",       # implied team points from moneyline/total
    "total":      "total_line",    # over/under line
    "result":     "spread_line",   # spread (market's predicted point diff)
}

# Odds columns used to derive market implied probability for binary outcomes
# total  → P(actual total > total_line)    using over_odds / under_odds
# result → P(team covers spread)           using spread_odds / opponent_spread_odds
PROB_ODDS = {
    "total":  ("over_odds", "under_odds"),
    "result": ("spread_odds", "opponent_spread_odds"),
}


def american_to_raw_prob(odds_arr):
    """Convert American odds (array) to raw implied probability (includes vig)."""
    odds_arr = np.asarray(odds_arr, dtype=float)
    return np.where(odds_arr < 0, -odds_arr / (-odds_arr + 100), 100 / (odds_arr + 100))


def remove_vig(p_side, p_other):
    """Normalise two raw implied probabilities so they sum to 1 (removes the vig)."""
    return p_side / (p_side + p_other)

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
    model_name = st.selectbox("Algorithm", list(MODEL_FACTORIES.keys()))

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

    # Random seed — random on first load so each session builds a unique model
    st.subheader("5. Random Seed")
    if "model_seed" not in st.session_state:
        st.session_state["model_seed"] = int(np.random.randint(1, 99999))
    seed_col, btn_col = st.columns([3, 1])
    with seed_col:
        seed = st.number_input(
            "Seed", min_value=1, max_value=99999,
            value=st.session_state["model_seed"], step=1,
            help="Controls train/test split and model randomness. Same seed + same config = same model.",
        )
        st.session_state["model_seed"] = int(seed)
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲", help="Pick a new random seed"):
            st.session_state["model_seed"] = int(np.random.randint(1, 99999))
            st.rerun()

    # Cross-validation
    st.subheader("6. Cross-Validation")
    cv_enabled = st.toggle("Enable cross-validation", value=True,
                           help="Evaluate model consistency across multiple folds of training data.")
    cv_folds = st.slider("CV folds", min_value=3, max_value=10, value=5,
                         disabled=not cv_enabled,
                         help="More folds = more reliable estimate but slower.")

    # Hyperparameter tuning
    st.subheader("7. Hyperparameter Tuning")
    can_tune = model_name in PARAM_GRIDS
    tune_enabled = st.toggle(
        "Auto-tune hyperparameters",
        value=False,
        disabled=not can_tune,
        help="Searches for the best model settings using randomized search + cross-validation. "
             "Slower but often improves accuracy." if can_tune
             else f"{model_name} has no tunable hyperparameters.",
    )
    n_iter = st.slider(
        "Search iterations", min_value=10, max_value=100, value=20, step=5,
        disabled=not (tune_enabled and can_tune),
        help="Number of random hyperparameter combinations to try. More = better search, slower runtime.",
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

    # Stash baseline + odds columns before we narrow the columns (so they survive dropna on features)
    baseline_col = MARKET_BASELINES.get(target)
    baseline_full = data[baseline_col].copy() if baseline_col and baseline_col in data.columns else None
    odds_full = {
        c: data[c].copy()
        for c in PROB_ODDS.get(target, ())
        if c in data.columns
    }

    # Drop rows missing the target or any selected feature
    cols_needed = feature_list + [target]
    data = data[cols_needed].dropna()

    if len(data) < 50:
        st.error("Not enough rows after filtering. Try including more seasons or fewer required features.")
        st.stop()

    X = data[feature_list]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Build pipeline — scaler is part of the pipeline so CV folds are leak-free
    base_model = MODEL_FACTORIES[model_name](seed)
    steps = [("model", base_model)]
    if model_name in SCALE_MODELS:
        steps.insert(0, ("scaler", StandardScaler()))
    pipeline = Pipeline(steps)

    best_params = None
    cv_results  = None
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # ── Hyperparameter tuning ────────────────────────────────────────────────────
    if tune_enabled and model_name in PARAM_GRIDS:
        with st.spinner(
            f"Tuning {model_name} — {n_iter} iterations × {cv_folds}-fold CV …"
        ):
            search = RandomizedSearchCV(
                pipeline,
                PARAM_GRIDS[model_name],
                n_iter=n_iter,
                cv=kf,
                scoring="neg_mean_absolute_error",
                random_state=seed,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
        pipeline   = search.best_estimator_
        best_params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
    else:
        pipeline.fit(X_train, y_train)

    # ── Cross-validation (always on training data, with final pipeline config) ───
    if cv_enabled:
        with st.spinner(f"Running {cv_folds}-fold cross-validation …"):
            rmse_scores = -cross_val_score(
                pipeline, X_train, y_train, cv=kf,
                scoring="neg_root_mean_squared_error",
            )
            mae_scores = -cross_val_score(
                pipeline, X_train, y_train, cv=kf,
                scoring="neg_mean_absolute_error",
            )
            r2_scores = cross_val_score(
                pipeline, X_train, y_train, cv=kf, scoring="r2",
            )
        cv_results = {
            "folds":     cv_folds,
            "rmse":      rmse_scores,
            "mae":       mae_scores,
            "r2":        r2_scores,
        }

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ── Market baseline regression metrics ──────────────────────────────────────
    b_rmse = b_mae = b_r2 = None
    if baseline_full is not None:
        baseline_test = baseline_full.loc[y_test.index]
        valid_reg = baseline_test.notna()
        if valid_reg.sum() > 10:
            bt = baseline_test[valid_reg].values
            yt = y_test[valid_reg].values
            b_rmse = np.sqrt(mean_squared_error(yt, bt))
            b_mae  = mean_absolute_error(yt, bt)
            b_r2   = r2_score(yt, bt)

    # ── Probability scoring (total / result only) ────────────────────────────────
    prob_results = None
    prob_cols = PROB_ODDS.get(target, ())
    if baseline_full is not None and len(prob_cols) == 2 and all(c in odds_full for c in prob_cols):
        o1_col, o2_col = prob_cols
        line_test = baseline_full.loc[y_test.index]
        o1_test   = odds_full[o1_col].loc[y_test.index]
        o2_test   = odds_full[o2_col].loc[y_test.index]
        valid_prob = line_test.notna() & o1_test.notna() & o2_test.notna()

        # Residual std from training predictions (used for probability conversion)
        train_resid_std = max(np.std(y_train.values - pipeline.predict(X_train)), 0.01)

        # P(Over/Cover) for ALL test rows that have a valid line — used in sample table
        # For total:  P(actual > total_line)       → effective threshold = +line
        # For result: P(result + spread_line > 0)  → effective threshold = -line
        #   (spread_line is negative for favorites in nflfastR, so -spread_line
        #    is the points the team must WIN by to cover)
        model_prob_full = pd.Series(np.nan, index=y_test.index)
        line_notnull = line_test.notna()
        line_vals_full = line_test[line_notnull].values
        eff_line_full  = line_vals_full if target == "total" else -line_vals_full
        model_prob_full.loc[line_notnull[line_notnull].index] = np.clip(
            norm.cdf(
                (y_pred[line_notnull.values] - eff_line_full) / train_resid_std
            ),
            1e-6, 1 - 1e-6,
        )

        if valid_prob.sum() > 10:
            line_v = line_test[valid_prob].values
            o1_v   = o1_test[valid_prob].values
            o2_v   = o2_test[valid_prob].values
            y_v    = y_test[valid_prob].values
            p_v    = y_pred[valid_prob.values]   # positional mask into numpy array

            # Effective threshold for each target:
            #   total  → over if actual > total_line
            #   result → cover if result > -spread_line  (result + spread_line > 0)
            eff_line = line_v if target == "total" else -line_v

            y_binary = (y_v > eff_line).astype(int)

            # Market no-vig probability for "side 1"
            raw_p1      = american_to_raw_prob(o1_v)
            raw_p2      = american_to_raw_prob(o2_v)
            market_prob = np.clip(remove_vig(raw_p1, raw_p2), 1e-6, 1 - 1e-6)
            model_prob  = np.clip(norm.cdf((p_v - eff_line) / train_resid_std), 1e-6, 1 - 1e-6)

            side1_label = "Over" if target == "total" else "Cover"
            prob_results = dict(
                n=int(valid_prob.sum()),
                side1_label=side1_label,
                o1_col=o1_col, o2_col=o2_col,
                model_prob=model_prob, market_prob=market_prob,
                model_prob_full=model_prob_full,
                y_binary=y_binary,
                train_resid_std=train_resid_std,
                pct_correct_model=float(np.mean((model_prob > 0.5) == y_binary)),
                pct_correct_market=float(np.mean((market_prob > 0.5) == y_binary)),
            )

    # ── Results layout ──────────────────────────────────────────────────────────
    st.header("📊 Model Results")
    st.caption(
        f"**{model_name}** · Target: `{target}` · "
        f"Train rows: {len(X_train):,} · Test rows: {len(X_test):,} · "
        f"Seed: `{seed}`"
    )

    # ── Plain English Summary ────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append(
        f"Your **{model_name}** was trained to predict **{target}** using "
        f"**{len(feature_list)} feature(s)** and evaluated on **{len(y_test):,} held-out games**."
    )

    if target == "team_score" and b_mae is not None:
        edge = b_mae - mae
        direction = "better" if edge > 0 else "worse"
        summary_lines.append(
            f"On average the model's score predictions were off by **{mae:.2f} points**, "
            f"compared to the market's implied score being off by **{b_mae:.2f} points** — "
            f"the model is **{abs(edge):.2f} points {direction}** than the market."
        )
        if edge > 0:
            summary_lines.append(
                "The model is adding predictive information on top of what the market already prices in. "
                "That's a promising signal for building value bets around team scoring."
            )
        else:
            summary_lines.append(
                "The market implied score is still more accurate than the model. "
                "This suggests stronger features or more data are needed to overcome the market."
            )

    if prob_results is not None:
        side = prob_results["side1_label"].lower()
        acc_m   = prob_results["pct_correct_model"]  * 100
        acc_mkt = prob_results["pct_correct_market"] * 100
        edge    = acc_m - acc_mkt
        beat    = acc_m > acc_mkt
        summary_lines.append(
            f"The model's numerical predictions were converted to a probability of going "
            f"**{side}** the line (residual std = {prob_results['train_resid_std']:.2f} points). "
            f"Across {prob_results['n']:,} games with available odds, the model called the correct "
            f"side **{acc_m:.1f}%** of the time vs the market's **{acc_mkt:.1f}%** — "
            f"an edge of **{edge:+.1f} percentage points**."
        )
        if beat:
            summary_lines.append(
                f"**The model beats the market** — it correctly called the {side} side "
                f"{edge:.1f}pp more often than the market's implied favourite. "
                "That frequency edge is where betting value can be extracted."
            )
        else:
            summary_lines.append(
                f"**The model does not beat the market** — the market called the correct side "
                f"{-edge:.1f}pp more often. Try adding more informative features, a different "
                "algorithm, or more seasons of data."
            )
    elif b_rmse is not None and target != "team_score":
        if rmse < b_rmse and mae < b_mae:
            summary_lines.append(
                "The model outperforms the market baseline on both RMSE and MAE. "
                "Note: side-accuracy scoring was skipped because the required odds columns "
                f"({' / '.join(PROB_ODDS.get(target, []))}) were not found in the data."
            )
        else:
            summary_lines.append(
                "The model does not outperform the market baseline on error metrics. "
                "Consider different features or a different algorithm."
            )

    with st.expander("📝 Plain English Summary", expanded=True):
        for line in summary_lines:
            st.markdown(f"- {line}")

    # ── Verdict banner ───────────────────────────────────────────────────────────
    if prob_results is not None:
        acc_m   = prob_results["pct_correct_model"]  * 100
        acc_mkt = prob_results["pct_correct_market"] * 100
        if acc_m > acc_mkt:
            st.success(
                f"**Model beats the market (side accuracy) — "
                f"Model: {acc_m:.1f}% | Market: {acc_mkt:.1f}%**"
            )
        else:
            st.error(
                f"**Model does NOT beat the market (side accuracy) — "
                f"Model: {acc_m:.1f}% | Market: {acc_mkt:.1f}%**"
            )
    elif b_rmse is not None:
        beats_rmse = rmse < b_rmse
        beats_mae  = mae  < b_mae
        n_beats = sum([beats_rmse, beats_mae])
        if n_beats == 2:
            st.success(
                f"**Model beats the market baseline ({baseline_col}) on both RMSE and MAE — "
                f"RMSE: {b_rmse:.3f} → {rmse:.3f} | MAE: {b_mae:.3f} → {mae:.3f}**"
            )
        elif n_beats == 1:
            st.warning(
                f"**Model beats the market on one of two error metrics — "
                f"RMSE: {b_rmse:.3f} → {rmse:.3f} | MAE: {b_mae:.3f} → {mae:.3f}**"
            )
        else:
            st.error(
                f"**Model does NOT beat the market baseline ({baseline_col}) — "
                f"RMSE: {b_rmse:.3f} → {rmse:.3f} | MAE: {b_mae:.3f} → {mae:.3f}**"
            )

    # ── Probability metrics (total / result) ─────────────────────────────────────
    if prob_results is not None:
        acc_m   = prob_results["pct_correct_model"]  * 100
        acc_mkt = prob_results["pct_correct_market"] * 100
        st.markdown("#### Side Accuracy vs Market")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric(
            "Model Side Accuracy",
            f"{acc_m:.1f}%",
            delta=f"{acc_m - acc_mkt:+.1f}pp vs market",
            delta_color="normal",
            help=f"% of games where the model's P({prob_results['side1_label']}) > 50% matched the actual outcome.",
        )
        pc2.metric(
            "Market Side Accuracy",
            f"{acc_mkt:.1f}%",
            help="% of games where the market's no-vig implied favourite was correct.",
        )
        pc3.metric(
            "Games Evaluated",
            f"{prob_results['n']:,}",
            help="Test-set games where both the line and odds were available.",
        )

    # ── Regression metrics vs market ─────────────────────────────────────────────
    st.markdown("#### Regression Accuracy vs Market")
    if b_rmse is not None:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "R²", f"{r2:.4f}",
            delta=f"{r2 - b_r2:+.4f} vs market", delta_color="normal",
            help="Higher is better.",
        )
        mc2.metric(
            "RMSE", f"{rmse:.3f}",
            delta=f"{rmse - b_rmse:+.3f} vs market", delta_color="inverse",
            help="Lower is better — negative delta means model wins.",
        )
        mc3.metric(
            "MAE", f"{mae:.3f}",
            delta=f"{mae - b_mae:+.3f} vs market", delta_color="inverse",
            help="Lower is better — negative delta means model wins.",
        )
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.4f}", help="1.0 = perfect; 0 = no better than the mean.")
        m2.metric("RMSE", f"{rmse:.3f}", help="Root Mean Squared Error (same units as target).")
        m3.metric("MAE", f"{mae:.3f}", help="Mean Absolute Error (same units as target).")

    # ── Cross-validation results ──────────────────────────────────────────────────
    if cv_results is not None:
        r = cv_results
        rmse_cv, mae_cv, r2_cv = r["rmse"], r["mae"], r["r2"]
        cv_stable = (rmse_cv.std() / rmse_cv.mean()) < 0.10  # <10% CoV = stable

        st.markdown("#### Cross-Validation Results (training data)")
        if cv_stable:
            st.success(
                f"**Model is stable** — RMSE varied by less than 10% across folds "
                f"(CV: {rmse_cv.mean():.3f} ± {rmse_cv.std():.3f})"
            )
        else:
            st.warning(
                f"**Model shows variance across folds** — RMSE CoV "
                f"{rmse_cv.std()/rmse_cv.mean()*100:.1f}%. "
                "Consider more data, fewer features, or stronger regularisation."
            )

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(
            "CV RMSE", f"{rmse_cv.mean():.3f}",
            delta=f"±{rmse_cv.std():.3f} std",
            delta_color="off",
            help="Mean RMSE across all CV folds (lower is better).",
        )
        cc2.metric(
            "CV MAE", f"{mae_cv.mean():.3f}",
            delta=f"±{mae_cv.std():.3f} std",
            delta_color="off",
            help="Mean MAE across all CV folds (lower is better).",
        )
        cc3.metric(
            "CV R²", f"{r2_cv.mean():.4f}",
            delta=f"±{r2_cv.std():.4f} std",
            delta_color="off",
            help="Mean R² across all CV folds (higher is better).",
        )

        fold_df = pd.DataFrame({
            "Fold":  [f"Fold {i+1}" for i in range(r["folds"])],
            "RMSE":  rmse_cv,
            "MAE":   mae_cv,
            "R²":    r2_cv,
        })
        fig_cv = px.bar(
            fold_df, x="Fold", y="RMSE",
            title=f"{r['folds']}-Fold CV — RMSE per Fold",
            color="RMSE",
            color_continuous_scale="Blues_r",
            text=fold_df["RMSE"].round(3),
        )
        fig_cv.add_hline(
            y=rmse_cv.mean(), line_dash="dash", line_color="#00AEEF",
            annotation_text=f"Mean {rmse_cv.mean():.3f}",
        )
        fig_cv.update_traces(textposition="outside")
        fig_cv.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Best hyperparameters (if tuned) ──────────────────────────────────────────
    if best_params is not None:
        with st.expander("🔧 Auto-Tuned Hyperparameters", expanded=False):
            st.caption(
                f"Best parameters found by RandomizedSearchCV "
                f"({n_iter} iterations, {cv_folds}-fold CV, scored on MAE)."
            )
            params_df = pd.DataFrame(
                list(best_params.items()), columns=["Parameter", "Value"]
            )
            st.dataframe(params_df, use_container_width=True, hide_index=True)

    # ── Model vs Market probability scatter ──────────────────────────────────────
    if prob_results is not None:
        st.markdown("#### Model vs Market Probability (each dot = one game)")
        outcome_labels = {
            "1": prob_results["side1_label"],
            "0": f"Under / No Cover",
        }
        prob_df = pd.DataFrame({
            "Market P(Over/Cover)": prob_results["market_prob"],
            "Model P(Over/Cover)":  prob_results["model_prob"],
            "Outcome": pd.Series(prob_results["y_binary"]).astype(str).map(outcome_labels),
        })
        fig_prob = px.scatter(
            prob_df,
            x="Market P(Over/Cover)",
            y="Model P(Over/Cover)",
            color="Outcome",
            opacity=0.55,
            color_discrete_map={
                prob_results["side1_label"]: "#2ca02c",
                "Under / No Cover": "#d62728",
            },
            title="Where model and market agree — and where they diverge",
        )
        fig_prob.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(color="gray", dash="dash"),
        )
        fig_prob.add_hline(y=0.5, line_dash="dot", line_color="lightgray")
        fig_prob.add_vline(x=0.5, line_dash="dot", line_color="lightgray")
        fig_prob.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(fig_prob, use_container_width=True)
        st.caption(
            "Dots above the diagonal = model is more confident than the market. "
            "Top-left / bottom-right quadrants = model and market disagree — that's where potential edge lives."
        )

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
    fitted_model = pipeline.named_steps["model"]

    if hasattr(fitted_model, "coef_"):
        coef_df = pd.DataFrame({
            "Feature": feature_list,
            "Coefficient": fitted_model.coef_,
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

    elif hasattr(fitted_model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature": feature_list,
            "Importance": fitted_model.feature_importances_,
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
    if prob_results is not None:
        prob_col_label = f"P({prob_results['side1_label']})"
        sample[prob_col_label] = (
            prob_results["model_prob_full"].reset_index(drop=True)[:20].round(3).values
        )
    st.dataframe(sample, use_container_width=True)

    # Download predictions
    pred_df = X_test.copy().reset_index(drop=True)
    pred_df["Actual"] = y_test.values
    pred_df["Predicted"] = np.round(y_pred, 2)
    pred_df["Error"] = pred_df["Actual"] - pred_df["Predicted"]
    if prob_results is not None:
        prob_col_label = f"P({prob_results['side1_label']})"
        pred_df[prob_col_label] = prob_results["model_prob_full"].reset_index(drop=True).round(3).values
    csv_out = pred_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Predictions CSV",
        data=csv_out,
        file_name=f"nfl_predictions_{target}_{model_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Persist the trained model so the prediction section survives reruns
    st.session_state["trained_pipeline"]     = pipeline
    st.session_state["trained_features"]     = feature_list
    st.session_state["trained_target"]       = target
    st.session_state["trained_resid_std"]    = (
        prob_results["train_resid_std"] if prob_results else None
    )
    st.session_state["trained_prob_label"]   = (
        prob_results["side1_label"] if prob_results else None
    )
    st.session_state["trained_baseline_col"] = baseline_col

# ── Predict on New Data ────────────────────────────────────────────────────────
if "trained_pipeline" in st.session_state:
    st.divider()
    st.header("🔮 Predict on New Data")
    st.caption(
        f"Run the trained **{st.session_state.get('trained_target','?')}** model "
        "on fresh game data to generate predictions."
    )

    _pipeline   = st.session_state["trained_pipeline"]
    _features   = st.session_state["trained_features"]
    _target     = st.session_state["trained_target"]
    _resid_std  = st.session_state["trained_resid_std"]
    _prob_label = st.session_state["trained_prob_label"]
    _base_col   = st.session_state["trained_baseline_col"]

    input_method = st.radio(
        "Input method",
        ["Upload CSV", "API Feed URL"],
        horizontal=True,
        help="CSV is the primary option. API feed allows pasting a URL that returns JSON or CSV data.",
    )

    new_df = None

    if input_method == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload a CSV file containing the required feature columns",
            type=["csv"],
            help=(
                "The file must contain all feature columns used during training. "
                "Extra columns are ignored. The target column is optional."
            ),
        )
        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    else:  # API Feed URL
        api_sources = {
            "Custom URL": "",
            "nflfastR play-by-play (sample)": "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2023.csv",
        }
        api_choice = st.selectbox("Select a source or choose Custom URL", list(api_sources.keys()))
        api_url = st.text_input(
            "API / feed URL",
            value=api_sources[api_choice],
            placeholder="https://example.com/data.csv  or  .../data.json",
        )
        fetch_btn = st.button("Fetch Data", type="primary", key="fetch_api")
        if fetch_btn and api_url.strip():
            with st.spinner("Fetching data from URL…"):
                try:
                    resp = requests.get(api_url.strip(), timeout=30)
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "")
                    raw = resp.content
                    # Try CSV first, then JSON
                    try:
                        new_df = pd.read_csv(io.BytesIO(raw))
                    except Exception:
                        try:
                            new_df = pd.read_json(io.BytesIO(raw))
                        except Exception:
                            st.error(
                                "Could not parse the response as CSV or JSON. "
                                "Please check the URL and try again."
                            )
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to fetch URL: {e}")

    # ── Run predictions on the loaded DataFrame ────────────────────────────────
    if new_df is not None:
        st.markdown(f"**Loaded {len(new_df):,} rows × {new_df.shape[1]} columns.**")

        # Validate that required feature columns are present
        missing_cols = [c for c in _features if c not in new_df.columns]
        if missing_cols:
            st.error(
                f"The following feature columns are missing from the uploaded data: "
                f"`{'`, `'.join(missing_cols)}`\n\n"
                "Please make sure your file contains all features that were selected "
                "during training."
            )
        else:
            # Drop rows with nulls in feature columns and warn about it
            pred_input = new_df[_features].copy()
            n_before = len(pred_input)
            pred_input = pred_input.dropna()
            n_dropped = n_before - len(pred_input)
            if n_dropped > 0:
                st.warning(
                    f"{n_dropped:,} row(s) were dropped because they had missing values "
                    "in one or more feature columns."
                )

            if len(pred_input) == 0:
                st.error("No valid rows remain after dropping rows with missing values.")
            else:
                with st.spinner("Generating predictions…"):
                    new_preds = _pipeline.predict(pred_input)

                result_df = pred_input.copy().reset_index(drop=True)

                # Carry over non-feature columns from original upload for context
                meta_cols = [c for c in new_df.columns if c not in _features]
                for mc in meta_cols:
                    result_df.insert(0, mc, new_df.loc[pred_input.index, mc].values)

                result_df["Predicted"] = np.round(new_preds, 2)

                # P(Over/Cover) if the model supports it
                if _resid_std is not None and _base_col in new_df.columns:
                    line_vals = new_df.loc[pred_input.index, _base_col].values.astype(float)
                    eff_line  = line_vals if _target == "total" else -line_vals
                    prob_vals = np.clip(
                        norm.cdf((new_preds - eff_line) / _resid_std),
                        1e-6, 1 - 1e-6,
                    )
                    prob_vals = np.where(np.isnan(line_vals), np.nan, prob_vals)
                    result_df[f"P({_prob_label})"] = np.round(prob_vals, 3)
                elif _resid_std is not None and _base_col not in new_df.columns:
                    st.info(
                        f"Column `{_base_col}` not found in the uploaded data — "
                        f"P({_prob_label}) will not be calculated."
                    )

                # Show the target column if present
                if _target in new_df.columns:
                    result_df["Actual"] = new_df.loc[pred_input.index, _target].values
                    result_df["Error"]  = np.round(
                        result_df["Actual"] - result_df["Predicted"], 2
                    )

                st.success(f"Predictions generated for {len(result_df):,} games.")
                st.dataframe(result_df, use_container_width=True)

                # Download
                dl_csv = result_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download New Predictions CSV",
                    data=dl_csv,
                    file_name=f"new_predictions_{_target}.csv",
                    mime="text/csv",
                    key="dl_new_preds",
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
