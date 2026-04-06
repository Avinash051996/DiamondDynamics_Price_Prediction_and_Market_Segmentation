"""
💎 Diamond Dynamics: Price Prediction & Market Segmentation
=============================================================
Production-ready ML Pipeline
- Data Cleaning & Preprocessing
- EDA (saved as plots)
- Feature Engineering
- Outlier & Skewness Handling
- Regression: LinearRegression, Ridge, DecisionTree, RandomForest, KNN, ANN (NumPy)
- Clustering: K-Means with PCA visualisation
- Saves all models as .pkl files
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

USD_TO_INR = 83.5   # Fixed conversion rate (update as needed)

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load diamonds CSV. Falls back to generating a realistic synthetic
    dataset (~54k rows) if no CSV is found — handy for demo/testing.
    """
    csv_candidates = [
        os.path.join(BASE_DIR, "data", "diamonds.csv"),
        os.path.join(BASE_DIR, "diamonds.csv"),
    ]
    for path in csv_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Loaded dataset from {path}  shape={df.shape}")
            return df

    print("⚠️  No CSV found — generating synthetic dataset (~54k rows) …")
    return _generate_synthetic_data(n=53940)


def _generate_synthetic_data(n: int = 53940, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cuts      = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    colors    = list("DEFGHIJ")
    clarities = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]

    carat = np.round(rng.exponential(0.5, n).clip(0.2, 5.0), 2)
    cut_idx     = rng.integers(0, 5, n)
    color_idx   = rng.integers(0, 7, n)
    clarity_idx = rng.integers(0, 8, n)
    depth = rng.normal(61.7, 1.4, n).clip(54, 70)
    table = rng.normal(57.5, 2.2, n).clip(50, 70)
    x = carat * 4.0 + rng.normal(0, 0.1, n)
    y = x + rng.normal(0, 0.05, n)
    z = x * 0.62 + rng.normal(0, 0.05, n)

    price = (
        3000 * carat**1.5
        + cut_idx * 150
        - color_idx * 80
        + clarity_idx * 100
        + rng.normal(0, 200, n)
    ).clip(200).astype(int)

    return pd.DataFrame({
        "carat":   carat,
        "cut":     [cuts[i] for i in cut_idx],
        "color":   [colors[i] for i in color_idx],
        "clarity": [clarities[i] for i in clarity_idx],
        "depth":   np.round(depth, 1),
        "table":   np.round(table, 1),
        "price":   price,
        "x":       np.round(x, 2),
        "y":       np.round(y, 2),
        "z":       np.round(z, 2),
    })


# ──────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ──────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnamed index columns if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Replace zero/negative x, y, z with NaN → impute with median
    for col in ["x", "y", "z"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].mask(df[col] < 0)
        df[col].fillna(df[col].median(), inplace=True)

    # Drop remaining rows with any NaN
    df.dropna(inplace=True)

    print(f"📋 After preprocessing: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 3. OUTLIER REMOVAL (IQR)
# ──────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame,
                    cols: list = None) -> pd.DataFrame:
    if cols is None:
        cols = ["carat", "price", "x", "y", "z"]
    df = df.copy()
    before = len(df)
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"🔍 Outlier removal: {before} → {len(df)} rows")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 4. SKEWNESS HANDLING
# ──────────────────────────────────────────────
def handle_skewness(df: pd.DataFrame,
                    cols: list = None,
                    threshold: float = 0.75) -> pd.DataFrame:
    if cols is None:
        cols = ["price", "carat", "x", "y", "z"]
    df = df.copy()
    skew_log = {}
    for col in cols:
        sk = df[col].skew()
        if abs(sk) > threshold:
            df[col] = np.log1p(df[col])
            skew_log[col] = round(sk, 3)
    if skew_log:
        print(f"📐 Log-transformed (skewness before): {skew_log}")
    return df


# ──────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # USD → INR
    if "price" in df.columns:
        df["price_inr"] = df["price"] * USD_TO_INR

    # Geometric / ratio features
    df["volume"]          = df["x"] * df["y"] * df["z"]
    df["dimension_ratio"] = (df["x"] + df["y"]) / (2 * df["z"].replace(0, np.nan))
    df["dimension_ratio"].fillna(df["dimension_ratio"].median(), inplace=True)

    # Price-per-carat (before log transform)
    if "price" in df.columns and "carat" in df.columns:
        df["price_per_carat"] = df["price"] / (df["carat"].replace(0, np.nan))
        df["price_per_carat"].fillna(df["price_per_carat"].median(), inplace=True)

    # Carat category
    carat_col = "carat"
    df["carat_category"] = pd.cut(
        df[carat_col],
        bins=[-np.inf, np.log1p(0.5), np.log1p(1.5), np.inf],
        labels=["Light", "Medium", "Heavy"]
    )

    print(f"🔧 Feature engineering complete. Columns now: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────
# 6. EDA PLOTS
# ──────────────────────────────────────────────
def run_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """Save EDA plots to PLOT_DIR using the raw (original scale) data."""
    sns.set_theme(style="whitegrid", palette="muted")

    # 6a. Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_raw["price"], bins=60, color="#6c5ce7", edgecolor="white")
    axes[0].set_title("Price Distribution (USD)")
    axes[0].set_xlabel("Price")
    axes[1].hist(np.log1p(df_raw["price"]), bins=60, color="#00b894", edgecolor="white")
    axes[1].set_title("Log Price Distribution")
    axes[1].set_xlabel("log(1 + Price)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_price_distribution.png"), dpi=120)
    plt.close()

    # 6b. Carat distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_raw["carat"], bins=50, color="#fdcb6e", edgecolor="white")
    ax.set_title("Carat Distribution")
    ax.set_xlabel("Carat")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_carat_distribution.png"), dpi=120)
    plt.close()

    # 6c. Count plots for categoricals
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["cut", "color", "clarity"]):
        order = df_raw[col].value_counts().index
        sns.countplot(data=df_raw, x=col, order=order, ax=ax, palette="Set2")
        ax.set_title(f"Count: {col.capitalize()}")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_categorical_counts.png"), dpi=120)
    plt.close()

    # 6d. Price by cut / color
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order   = list("DEFGHIJ")
    sns.boxplot(data=df_raw, x="cut",   y="price", order=cut_order,   ax=axes[0], palette="Set3")
    sns.boxplot(data=df_raw, x="color", y="price", order=color_order, ax=axes[1], palette="Set1")
    axes[0].set_title("Price by Cut")
    axes[1].set_title("Price by Color")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_price_by_cut_color.png"), dpi=120)
    plt.close()

    # 6e. Correlation heatmap
    num_cols = ["carat", "depth", "table", "price", "x", "y", "z"]
    corr = df_raw[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_correlation_heatmap.png"), dpi=120)
    plt.close()

    # 6f. Carat vs Price scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_raw["carat"], df_raw["price"], alpha=0.2, s=5, color="#e17055")
    ax.set_xlabel("Carat")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Carat vs Price")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_carat_vs_price.png"), dpi=120)
    plt.close()

    # 6g. Average price per clarity
    avg_price_clarity = (
        df_raw.groupby("clarity")["price"].mean()
        .reindex(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    avg_price_clarity.plot(kind="bar", color="#74b9ff", edgecolor="white", ax=ax)
    ax.set_title("Average Price by Clarity")
    ax.set_ylabel("Avg Price (USD)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_avg_price_by_clarity.png"), dpi=120)
    plt.close()

    print(f"📊 EDA plots saved to {PLOT_DIR}/")


# ──────────────────────────────────────────────
# 7. ENCODING & FEATURE SELECTION
# ──────────────────────────────────────────────
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER   = list("DEFGHIJ")
CLARITY_ORDER = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """Ordinal-encode cut, color, clarity and return (df_encoded, encoder)."""
    df = df.copy()
    enc = OrdinalEncoder(
        categories=[CUT_ORDER, COLOR_ORDER, CLARITY_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    df[["cut", "color", "clarity"]] = enc.fit_transform(df[["cut", "color", "clarity"]])
    return df, enc


# ──────────────────────────────────────────────
# 8. NUMPY ANN REGRESSOR
# ──────────────────────────────────────────────
class ANNRegressor:
    """
    Simple feed-forward ANN built from scratch with NumPy.
    Architecture: input → 128 → 64 → 32 → 1 (linear output)
    Uses ReLU activations, Adam optimiser, He weight init.
    """

    def __init__(self, hidden_sizes=(128, 64, 32),
                 learning_rate=1e-3, epochs=60, batch_size=512,
                 seed=42):
        self.hidden_sizes  = hidden_sizes
        self.lr            = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.seed          = seed
        self.weights       = []
        self.biases        = []
        self._adam_m_w     = []
        self._adam_v_w     = []
        self._adam_m_b     = []
        self._adam_v_b     = []
        self.loss_history  = []

    @staticmethod
    def _relu(x):      return np.maximum(0, x)
    @staticmethod
    def _relu_grad(x): return (x > 0).astype(float)

    def _init_weights(self, n_features):
        rng = np.random.default_rng(self.seed)
        layer_sizes = [n_features] + list(self.hidden_sizes) + [1]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            W = rng.standard_normal((fan_in, layer_sizes[i+1])) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
            self._adam_m_w.append(np.zeros_like(W))
            self._adam_v_w.append(np.zeros_like(W))
            self._adam_m_b.append(np.zeros_like(b))
            self._adam_v_b.append(np.zeros_like(b))

    def _forward(self, X):
        activations = [X]
        pre_acts    = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_acts.append(z)
            a = self._relu(z) if i < len(self.weights) - 1 else z
            activations.append(a)
        return activations, pre_acts

    def _backward(self, activations, pre_acts, y, t):
        n = y.shape[0]
        dL_da = (activations[-1] - y) * 2 / n   # MSE gradient
        grads_w, grads_b = [], []
        for i in reversed(range(len(self.weights))):
            dL_dz = dL_da if i == len(self.weights)-1 else dL_da * self._relu_grad(pre_acts[i])
            gW = activations[i].T @ dL_dz / n
            gb = dL_dz.mean(axis=0, keepdims=True)
            grads_w.insert(0, gW)
            grads_b.insert(0, gb)
            dL_da = dL_dz @ self.weights[i].T
        return grads_w, grads_b

    def _adam_update(self, grads_w, grads_b, t,
                     beta1=0.9, beta2=0.999, eps=1e-8):
        for i, (gW, gb) in enumerate(zip(grads_w, grads_b)):
            self._adam_m_w[i] = beta1 * self._adam_m_w[i] + (1-beta1) * gW
            self._adam_v_w[i] = beta2 * self._adam_v_w[i] + (1-beta2) * gW**2
            m_hat_w = self._adam_m_w[i] / (1 - beta1**t)
            v_hat_w = self._adam_v_w[i] / (1 - beta2**t)
            self.weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + eps)

            self._adam_m_b[i] = beta1 * self._adam_m_b[i] + (1-beta1) * gb
            self._adam_v_b[i] = beta2 * self._adam_v_b[i] + (1-beta2) * gb**2
            m_hat_b = self._adam_m_b[i] / (1 - beta1**t)
            v_hat_b = self._adam_v_b[i] / (1 - beta2**t)
            self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        self._init_weights(X.shape[1])
        n = X.shape[0]
        t = 0
        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X[perm], y[perm]
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start:start+self.batch_size]
                yb = y_shuf[start:start+self.batch_size]
                t += 1
                acts, pre = self._forward(Xb)
                loss = np.mean((acts[-1] - yb)**2)
                epoch_loss += loss * len(Xb)
                gw, gb = self._backward(acts, pre, yb, t)
                self._adam_update(gw, gb, t)
            self.loss_history.append(epoch_loss / n)
            if epoch % 10 == 0:
                print(f"   ANN epoch {epoch:3d}/{self.epochs}  MSE={epoch_loss/n:.4f}")
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        acts, _ = self._forward(X)
        return acts[-1].ravel()


# ──────────────────────────────────────────────
# 9. REGRESSION PIPELINE
# ──────────────────────────────────────────────
def build_regression_models(X_train, X_test, y_train, y_test,
                             scaler: StandardScaler):
    """Train 5 ML models + ANN, evaluate, return metrics & best model name."""
    # Scale features (already fitted on train)
    Xtr_sc = scaler.transform(X_train)
    Xts_sc = scaler.transform(X_test)

    models = {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=10.0),
        "Decision Tree":      DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
        "Random Forest":      RandomForestRegressor(n_estimators=150, max_depth=14,
                                                    min_samples_leaf=3, n_jobs=-1, random_state=42),
        "KNN Regressor":      KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1),
    }

    results = {}
    best_r2  = -np.inf
    best_name = ""

    for name, model in models.items():
        print(f"\n  ▶ Training {name} …")
        model.fit(Xtr_sc, y_train)
        preds = model.predict(Xts_sc)
        metrics = _eval_metrics(y_test, preds, name)
        results[name] = {"model": model, "metrics": metrics}
        if metrics["R²"] > best_r2:
            best_r2   = metrics["R²"]
            best_name = name

    # ANN
    print("\n  ▶ Training ANN (NumPy) …")
    ann = ANNRegressor(hidden_sizes=(128, 64, 32), learning_rate=1e-3,
                       epochs=60, batch_size=512)
    ann.fit(Xtr_sc, y_train)
    ann_preds = ann.predict(Xts_sc)
    ann_metrics = _eval_metrics(y_test, ann_preds, "ANN (NumPy)")
    results["ANN (NumPy)"] = {"model": ann, "metrics": ann_metrics}
    if ann_metrics["R²"] > best_r2:
        best_r2   = ann_metrics["R²"]
        best_name = "ANN (NumPy)"

    print(f"\n🏆 Best model: {best_name}  (R²={best_r2:.4f})")
    return results, best_name


def _eval_metrics(y_true, y_pred, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"   {label:25s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}


def save_regression_comparison_plot(results: dict):
    names   = list(results.keys())
    metrics = ["MAE", "RMSE", "R²"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        vals = [results[n]["metrics"][metric] for n in names]
        colors_ = ["#6c5ce7" if v == max(vals) else "#b2bec3" for v in vals] \
                  if metric == "R²" else \
                  ["#6c5ce7" if v == min(vals) else "#b2bec3" for v in vals]
        bars = ax.barh(names, vals, color=colors_)
        ax.set_title(f"Model Comparison — {metric}")
        ax.set_xlabel(metric)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="right", fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_model_comparison.png"), dpi=120)
    plt.close()
    print("📊 Model comparison plot saved.")


# ──────────────────────────────────────────────
# 10. CLUSTERING PIPELINE
# ──────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "Affordable Small Diamonds",
    1: "Mid-range Balanced Diamonds",
    2: "Premium Heavy Diamonds",
}

def build_clustering_model(df_encoded: pd.DataFrame,
                            feature_cols: list,
                            n_clusters_range: range = range(2, 9)):
    """Elbow + silhouette → best K → K-Means → PCA visualisation."""
    X_clus = df_encoded[feature_cols].copy()
    scaler_c = StandardScaler()
    X_sc = scaler_c.fit_transform(X_clus)

    # Elbow & silhouette
    inertias, sil_scores = [], []
    for k in n_clusters_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_sc, km.labels_, sample_size=5000))
    best_k = n_clusters_range.start + np.argmax(sil_scores)
    print(f"\n  ▶ Optimal K = {best_k}  (silhouette={max(sil_scores):.4f})")

    # Elbow plot
    ks = list(n_clusters_range)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, inertias, marker="o", color="#6c5ce7")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, sil_scores, marker="o", color="#00b894")
    axes[1].set_title("Silhouette Scores")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_clustering_elbow_silhouette.png"), dpi=120)
    plt.close()

    # Final K-Means (use best_k, cap at 3 so cluster names align)
    final_k = min(best_k, len(CLUSTER_NAMES))
    km_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_sc)

    # Cluster name mapping (by ascending mean price proxy = mean of first feature after sort)
    cluster_stats = pd.DataFrame({"cluster": labels})
    cluster_stats["carat"] = df_encoded["carat"].values if "carat" in df_encoded.columns else X_sc[:, 0]
    # Sort clusters by mean carat to assign budget/mid/premium names
    mean_carat = cluster_stats.groupby("cluster")["carat"].mean().sort_values()
    name_map = {old: CLUSTER_NAMES[new] for new, old in enumerate(mean_carat.index)}

    # PCA 2D visualisation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = ["#6c5ce7", "#00b894", "#fdcb6e", "#e17055", "#74b9ff"]
    for cl in range(final_k):
        mask = labels == cl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=4, alpha=0.4, label=name_map.get(cl, f"Cluster {cl}"),
                   color=palette[cl % len(palette)])
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title("K-Means Clusters — PCA 2D Projection")
    ax.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "10_cluster_pca_2d.png"), dpi=120)
    plt.close()

    sil_final = silhouette_score(X_sc, labels, sample_size=5000)
    print(f"  ▶ K-Means silhouette (K={final_k}): {sil_final:.4f}")
    print(f"  ▶ Cluster name map: {name_map}")

    return km_final, scaler_c, pca, name_map, feature_cols, final_k, sil_final


# ──────────────────────────────────────────────
# 11. SAVE ARTEFACTS
# ──────────────────────────────────────────────
def save_artifacts(best_model, encoder, reg_scaler,
                   km_model, clus_scaler, pca_model,
                   cluster_name_map, regression_feature_cols,
                   clustering_feature_cols, log_cols):
    """Pickle everything Streamlit needs."""
    artefacts = {
        "best_regression_model.pkl": best_model,
        "ordinal_encoder.pkl":       encoder,
        "regression_scaler.pkl":     reg_scaler,
        "kmeans_model.pkl":          km_model,
        "clustering_scaler.pkl":     clus_scaler,
        "pca_model.pkl":             pca_model,
        "cluster_name_map.pkl":      cluster_name_map,
        "regression_feature_cols.pkl":  regression_feature_cols,
        "clustering_feature_cols.pkl":  clustering_feature_cols,
        "log_transformed_cols.pkl":     log_cols,
    }
    for fname, obj in artefacts.items():
        path = os.path.join(MODEL_DIR, fname)
        joblib.dump(obj, path)
    print(f"\n✅ All artefacts saved to {MODEL_DIR}/")


# ──────────────────────────────────────────────
# 12. MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("💎  DIAMOND DYNAMICS — ML PIPELINE")
    print("=" * 60)

    # Load
    df_raw = load_data()

    # Preprocess
    df = preprocess(df_raw.copy())

    # EDA on raw data
    run_eda(df_raw, df)

    # Outlier removal
    df = remove_outliers(df)

    # Log transform skewed columns BEFORE feature engineering
    LOG_COLS = ["price", "carat", "x", "y", "z"]
    df = handle_skewness(df, cols=LOG_COLS)

    # Feature engineering
    df = feature_engineering(df)

    # Encode categoricals
    df, encoder = encode_features(df)

    # Drop non-numeric / helper columns
    drop_cols = ["price_inr", "price_per_carat", "carat_category"]
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── REGRESSION ──────────────────────────────
    print("\n📈  REGRESSION SECTION")
    REG_FEATURES = [c for c in df_model.columns if c != "price"]
    X = df_model[REG_FEATURES]
    y = df_model["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg_scaler = StandardScaler()
    reg_scaler.fit(X_train)

    reg_results, best_name = build_regression_models(
        X_train, X_test, y_train, y_test, reg_scaler
    )
    save_regression_comparison_plot(reg_results)
    best_model = reg_results[best_name]["model"]

    # Save model comparison table
    rows = []
    for name, res in reg_results.items():
        m = res["metrics"]
        rows.append({"Model": name, "MAE": round(m["MAE"], 4),
                     "MSE": round(m["MSE"], 4), "RMSE": round(m["RMSE"], 4),
                     "R²": round(m["R²"], 4)})
    pd.DataFrame(rows).to_csv(os.path.join(PLOT_DIR, "model_metrics.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # ── CLUSTERING ──────────────────────────────
    print("\n🔵  CLUSTERING SECTION")
    CLUS_FEATURES = [c for c in df_model.columns if c != "price"]
    (km_model, clus_scaler, pca_model,
     cluster_name_map, clus_feat_cols,
     final_k, sil_score) = build_clustering_model(
        df_model, CLUS_FEATURES
    )

    # ── SAVE ────────────────────────────────────
    save_artifacts(
        best_model     = best_model,
        encoder        = encoder,
        reg_scaler     = reg_scaler,
        km_model       = km_model,
        clus_scaler    = clus_scaler,
        pca_model      = pca_model,
        cluster_name_map          = cluster_name_map,
        regression_feature_cols   = REG_FEATURES,
        clustering_feature_cols   = CLUS_FEATURES,
        log_cols                   = LOG_COLS,
    )

    print("\n" + "=" * 60)
    print("🎉  Pipeline complete!")
    print(f"    Best regression model : {best_name}")
    print(f"    Clustering K          : {final_k}  (silhouette={sil_score:.4f})")
    print(f"    Plots saved to        : {PLOT_DIR}/")
    print(f"    Models saved to       : {MODEL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
