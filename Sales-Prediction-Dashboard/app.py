import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stMetric { background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    .stMetric label { color: #6b7280 !important; font-size: 13px !important; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 1.8rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
    h3 { font-size: 1rem !important; font-weight: 600 !important; }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px; padding: 1.5rem; text-align: center; color: white; margin-bottom: 1rem;
    }
    .prediction-box h3 { color: white !important; font-size: 0.85rem !important; opacity: 0.85; margin-bottom: 0.3rem; }
    .prediction-box .value { font-size: 2.2rem; font-weight: 700; }
    .prediction-box-rf {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 14px; padding: 1.5rem; text-align: center; color: white; margin-bottom: 1rem;
    }
    .prediction-box-rf h3 { color: white !important; font-size: 0.85rem !important; opacity: 0.85; margin-bottom: 0.3rem; }
    .prediction-box-rf .value { font-size: 2.2rem; font-weight: 700; }
    .insight-box {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        border-radius: 8px; padding: 1rem 1.2rem; margin-top: 1rem;
        font-size: 0.9rem; color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# ─── Color Palette ───────────────────────────────────────────────────────────
BLUE   = "#4361ee"
GREEN  = "#2ec4b6"
CORAL  = "#e76f51"
AMBER  = "#f4a261"
GRAY   = "#6b7280"
COLORS = [BLUE, GREEN, CORAL]
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

# ─── Load / Generate Data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 200
    tv        = np.random.uniform(0.7, 296.4, n)
    radio     = np.random.uniform(0.0,  49.6, n)
    newspaper = np.random.uniform(0.3, 114.0, n)
    sales     = 4.71 + 0.0545*tv + 0.1009*radio + 0.0043*newspaper + np.random.normal(0, 1.5, n)
    sales     = np.clip(sales, 1.6, 27.0)
    return pd.DataFrame({"TV": tv, "Radio": radio, "Newspaper": newspaper, "Sales": sales})

@st.cache_resource
def train_models(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return lr, rf, X_train, X_test, y_train, y_test

df = load_data()
lr_model, rf_model, X_train, X_test, y_train, y_test = train_models(df)

y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

metrics = {
    "Linear Regression": {
        "R²":   round(r2_score(y_test, y_pred_lr), 4),
        "MAE":  round(mean_absolute_error(y_test, y_pred_lr), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 4),
        "preds": y_pred_lr,
        "color": BLUE,
    },
    "Random Forest": {
        "R²":   round(r2_score(y_test, y_pred_rf), 4),
        "MAE":  round(mean_absolute_error(y_test, y_pred_rf), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 4),
        "preds": y_pred_rf,
        "color": GREEN,
    },
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Sales Predictor")
    st.markdown("Adjust advertising spend to forecast sales.")
    st.markdown("---")

    tv_spend    = st.slider("📺 TV Spend ($K)",        0.0, 300.0, 150.0, 0.5)
    radio_spend = st.slider("📻 Radio Spend ($K)",      0.0,  50.0,  23.0, 0.5)
    news_spend  = st.slider("📰 Newspaper Spend ($K)",  0.0, 115.0,  30.0, 0.5)

    input_df = pd.DataFrame({"TV": [tv_spend], "Radio": [radio_spend], "Newspaper": [news_spend]})
    pred_lr  = lr_model.predict(input_df)[0]
    pred_rf  = rf_model.predict(input_df)[0]

    st.markdown("---")
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Linear Regression</h3>
        <div class="value">{pred_lr:.2f}K</div>
        <div style="font-size:0.8rem;opacity:0.8;">units predicted</div>
    </div>
    <div class="prediction-box-rf">
        <h3>Random Forest</h3>
        <div class="value">{pred_rf:.2f}K</div>
        <div style="font-size:0.8rem;opacity:0.8;">units predicted</div>
    </div>
    """, unsafe_allow_html=True)

    total_spend = tv_spend + radio_spend + news_spend
    roi_lr = (pred_lr * 1000) / total_spend if total_spend > 0 else 0
    st.metric("Total Spend", f"${total_spend:.1f}K")
    st.metric("Est. ROI (RF)", f"{(pred_rf*1000/total_spend if total_spend>0 else 0):.1f}x")

# ─── Main Header ─────────────────────────────────────────────────────────────
st.title("📈 Sales Prediction Dashboard")
st.markdown("Machine learning analysis of advertising spend vs sales performance.")

# ─── Top Metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Dataset Size",  f"{len(df)}")
c2.metric("Features",      "3")
c3.metric("LR  R²",        f"{metrics['Linear Regression']['R²']}")
c4.metric("RF  R²",        f"{metrics['Random Forest']['R²']}")
c5.metric("LR  RMSE",      f"{metrics['Linear Regression']['RMSE']}")
c6.metric("RF  RMSE",      f"{metrics['Random Forest']['RMSE']}")

st.markdown("---")

# ─── Tab Layout ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploratory Analysis", "🤖 Model Performance", "🔍 Feature Insights", "📋 Data Table"])

# ══════════════════════════════════════════════════════
# TAB 1 — Exploratory Analysis
# ══════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    # Correlation Heatmap
    with col1:
        st.markdown("##### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(5, 4))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Matrix", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Distribution of Sales
    with col2:
        st.markdown("##### Sales Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df["Sales"], bins=25, color=BLUE, edgecolor="white", alpha=0.85)
        ax.axvline(df["Sales"].mean(), color=CORAL, linewidth=2, linestyle="--", label=f"Mean: {df['Sales'].mean():.1f}")
        ax.axvline(df["Sales"].median(), color=AMBER, linewidth=2, linestyle="--", label=f"Median: {df['Sales'].median():.1f}")
        ax.set_xlabel("Sales (units K)"); ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Sales", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Scatter plots for each feature vs Sales
    st.markdown("##### Advertising Spend vs Sales")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    features = ["TV", "Radio", "Newspaper"]
    for i, (feat, color) in enumerate(zip(features, COLORS)):
        axes[i].scatter(df[feat], df["Sales"], color=color, alpha=0.5, s=30, edgecolors="none")
        m, b = np.polyfit(df[feat], df["Sales"], 1)
        xs = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(xs, m*xs + b, color="black", linewidth=1.5, linestyle="--")
        axes[i].set_xlabel(f"{feat} Spend ($K)", fontsize=10)
        axes[i].set_ylabel("Sales (K)", fontsize=10)
        axes[i].set_title(f"{feat} vs Sales  (r={df[[feat,'Sales']].corr().iloc[0,1]:.2f})", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Pairplot summary stats
    st.markdown("---")
    st.markdown("##### Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")

    col1, col2 = st.columns(2)

    # Actual vs Predicted — LR
    with col1:
        st.markdown("##### Linear Regression: Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred_lr, color=BLUE, alpha=0.65, s=35, edgecolors="none")
        mn, mx = min(y_test.min(), y_pred_lr.min()), max(y_test.max(), y_pred_lr.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
        ax.set_xlabel("Actual Sales"); ax.set_ylabel("Predicted Sales")
        ax.set_title(f"R² = {metrics['Linear Regression']['R²']}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Actual vs Predicted — RF
    with col2:
        st.markdown("##### Random Forest: Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred_rf, color=GREEN, alpha=0.65, s=35, edgecolors="none")
        mn, mx = min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
        ax.set_xlabel("Actual Sales"); ax.set_ylabel("Predicted Sales")
        ax.set_title(f"R² = {metrics['Random Forest']['R²']}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Residuals — LR
    with col1:
        st.markdown("##### Residuals — Linear Regression")
        residuals_lr = y_test.values - y_pred_lr
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(y_pred_lr, residuals_lr, color=BLUE, alpha=0.6, s=30, edgecolors="none")
        ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Residuals — RF
    with col2:
        st.markdown("##### Residuals — Random Forest")
        residuals_rf = y_test.values - y_pred_rf
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(y_pred_rf, residuals_rf, color=GREEN, alpha=0.6, s=30, edgecolors="none")
        ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Metric Comparison Bar Chart
    st.markdown("##### Metric Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metric_names = ["R²", "MAE", "RMSE"]
    for i, metric in enumerate(metric_names):
        vals   = [metrics["Linear Regression"][metric], metrics["Random Forest"][metric]]
        colors = [BLUE, GREEN]
        bars   = axes[i].bar(["Linear\nRegression", "Random\nForest"], vals, color=colors, width=0.5, edgecolor="none")
        axes[i].set_title(metric, fontsize=12, fontweight="bold")
        axes[i].set_ylim(0, max(vals) * 1.25)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                         f"{val:.3f}", ha="center", fontsize=10, fontweight="500")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary table
    st.markdown("##### Summary Table")
    summary_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R² Score": [metrics["Linear Regression"]["R²"], metrics["Random Forest"]["R²"]],
        "MAE":      [metrics["Linear Regression"]["MAE"], metrics["Random Forest"]["MAE"]],
        "RMSE":     [metrics["Linear Regression"]["RMSE"], metrics["Random Forest"]["RMSE"]],
    })
    st.dataframe(summary_df.set_index("Model"), use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 3 — Feature Insights
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("Feature Insights")

    col1, col2 = st.columns(2)

    # Feature Importance (RF)
    with col1:
        st.markdown("##### Feature Importance (Random Forest)")
        importances = rf_model.feature_importances_
        feat_names  = ["TV", "Radio", "Newspaper"]
        sorted_idx  = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh([feat_names[i] for i in sorted_idx],
                       [importances[i] for i in sorted_idx],
                       color=[COLORS[i] for i in sorted_idx], edgecolor="none", height=0.5)
        for bar, imp in zip(bars, [importances[i] for i in sorted_idx]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{imp*100:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance", fontsize=11)
        ax.set_xlim(0, importances.max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # LR Coefficients
    with col2:
        st.markdown("##### Linear Regression Coefficients")
        coefs = lr_model.coef_
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(feat_names, coefs, color=COLORS, edgecolor="none", width=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, c in zip(bars, coefs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001 if c >= 0 else bar.get_height() - 0.004,
                    f"{c:.4f}", ha="center", fontsize=10)
        ax.set_ylabel("Coefficient Value")
        ax.set_title(f"Intercept: {lr_model.intercept_:.2f}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Spend allocation pie
    st.markdown("##### Current Spend Allocation")
    col1, col2 = st.columns([1, 2])
    with col1:
        total = tv_spend + radio_spend + news_spend
        if total > 0:
            fig, ax = plt.subplots(figsize=(4, 4))
            sizes  = [tv_spend, radio_spend, news_spend]
            labels = [f"TV\n${tv_spend:.0f}K", f"Radio\n${radio_spend:.0f}K", f"News\n${news_spend:.0f}K"]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                colors=COLORS, startangle=90, pctdistance=0.75,
                wedgeprops={"edgecolor": "white", "linewidth": 2}
            )
            for t in autotexts:
                t.set_fontsize(9)
            ax.set_title("Spend Breakdown", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Set spend values in the sidebar.")

    with col2:
        st.markdown("##### Simulated Spend Scenarios")
        scenarios = pd.DataFrame({
            "Scenario":  ["Low Budget", "TV-Heavy", "Balanced", "Radio-Heavy", "Max Budget"],
            "TV ($K)":   [50, 250, 150, 80, 290],
            "Radio ($K)":[10, 15,  25, 45, 48],
            "Newspaper ($K)": [10, 10,  30, 20, 100],
        })
        for col in ["TV ($K)", "Radio ($K)", "Newspaper ($K)"]:
            feat = col.split(" ")[0]
            scenarios[f"{feat} Impact"] = (scenarios[col] * lr_model.coef_[["TV","Radio","Newspaper"].index(feat)]).round(2)

        scenarios["Predicted Sales (LR)"] = (
            lr_model.intercept_
            + scenarios["TV ($K)"] * lr_model.coef_[0]
            + scenarios["Radio ($K)"] * lr_model.coef_[1]
            + scenarios["Newspaper ($K)"] * lr_model.coef_[2]
        ).round(2)
        st.dataframe(
            scenarios[["Scenario", "TV ($K)", "Radio ($K)", "Newspaper ($K)", "Predicted Sales (LR)"]].set_index("Scenario"),
            use_container_width=True
        )

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Key Insight:</strong> TV advertising is the dominant driver (84.5% feature importance).
    Radio offers the best per-dollar return in the linear model (+0.10 units per $1K).
    Newspaper has minimal impact — reallocating that budget to TV or Radio improves ROI.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 4 — Data Table
# ══════════════════════════════════════════════════════
with tab4:
    st.subheader("Dataset")
    st.markdown(f"**{len(df)} records** | Features: TV, Radio, Newspaper spend ($K) | Target: Sales (units K)")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_tv = st.slider("Filter: Max TV Spend", 0.0, 300.0, 300.0)
    with col2:
        filter_radio = st.slider("Filter: Max Radio Spend", 0.0, 50.0, 50.0)
    with col3:
        filter_sales = st.slider("Filter: Min Sales", 0.0, 27.0, 0.0)

    filtered_df = df[
        (df["TV"] <= filter_tv) &
        (df["Radio"] <= filter_radio) &
        (df["Sales"] >= filter_sales)
    ].round(2)

    st.info(f"Showing {len(filtered_df)} of {len(df)} records")
    st.dataframe(filtered_df, use_container_width=True, height=400)

    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered CSV", csv, "filtered_data.csv", "text/csv")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:0.8rem;'>"
    "Sales Prediction Dashboard · Built with Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)                                                                                                                                                                       