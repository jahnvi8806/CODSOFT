"""
Credit Card Fraud Detection
============================
Pipeline: Load → Preprocess → Handle Imbalance → Train → Evaluate
Models: Logistic Regression, Random Forest
Techniques: SMOTE oversampling, undersampling, threshold tuning
"""
from unicodedata import name

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load transaction data.
    - Pass a CSV filepath (e.g. Kaggle's creditcard.csv) OR
    - Leave None to generate a synthetic dataset for demo.

    Expected CSV columns:
        Time, V1..V28 (PCA features), Amount, Class (0=genuine, 1=fraud)
    """
    if filepath:
        df = pd.read_csv(filepath)
        print(f"Loaded real dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    else:
        print("No filepath provided — generating synthetic dataset…")
        X, y = make_classification(
            n_samples=284_807,
            n_features=30,
            n_informative=20,
            n_redundant=5,
            weights=[0.9983, 0.0017],   # mirrors real fraud ratio (~0.17%)
            flip_y=0,
            random_state=42
        )
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame(X, columns=cols)
        df["Class"] = y

    fraud_pct = df["Class"].mean() * 100
    print(f"Fraud rate: {fraud_pct:.4f}%  ({df['Class'].sum():,} / {len(df):,} transactions)\n")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    - Scale 'Time' and 'Amount' (PCA features V1-V28 are already scaled)
    - Split into train/test with stratification
    """
    df = df.copy()

    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"]   = scaler.fit_transform(df[["Time"]])
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")
    print(f"Train fraud: {y_train.sum():,}  |  Test fraud: {y_test.sum():,}\n")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3. CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────

def apply_oversampling(X_train: pd.DataFrame, y_train: pd.Series):
    """Random oversampling of the minority (fraud) class."""
    df_train = pd.concat([X_train, y_train], axis=1)
    majority = df_train[df_train["Class"] == 0]
    minority = df_train[df_train["Class"] == 1]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )
    balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)

    X_bal = balanced.drop("Class", axis=1)
    y_bal = balanced["Class"]
    print(f"[Oversample] Balanced dataset: {len(X_bal):,} rows  "
          f"(fraud={y_bal.sum():,}  genuine={(y_bal==0).sum():,})")
    return X_bal, y_bal


def apply_undersampling(X_train: pd.DataFrame, y_train: pd.Series):
    """Random undersampling of the majority (genuine) class."""
    df_train = pd.concat([X_train, y_train], axis=1)
    majority = df_train[df_train["Class"] == 0]
    minority = df_train[df_train["Class"] == 1]

    majority_downsampled = resample(
        majority,
        replace=False,
        n_samples=len(minority) * 10,   # keep 10x ratio for balance + enough data
        random_state=42
    )
    balanced = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=42)

    X_bal = balanced.drop("Class", axis=1)
    y_bal = balanced["Class"]
    print(f"[Undersample] Balanced dataset: {len(X_bal):,} rows  "
          f"(fraud={y_bal.sum():,}  genuine={(y_bal==0).sum():,})")
    return X_bal, y_bal


def try_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    SMOTE (Synthetic Minority Oversampling Technique) — best option when available.
    Requires: pip install imbalanced-learn
    Falls back to random oversampling if not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"[SMOTE] Resampled: {len(X_res):,} rows  "
              f"(fraud={y_res.sum():,}  genuine={(y_res==0).sum():,})")
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)
    except ImportError:
        print("[SMOTE] imbalanced-learn not installed — falling back to oversampling.")
        return apply_oversampling(X_train, y_train)


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Logistic Regression with class_weight='balanced' as a built-in imbalance handler.
    Also works great on already-balanced data.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Random Forest — handles non-linearity well; great for tabular fraud data.
    n_estimators=100 balances speed vs accuracy.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test, threshold: float = 0.5):
    """
    Full evaluation suite:
      - Classification report (Precision / Recall / F1)
      - Confusion matrix
      - ROC-AUC
      - Optional custom threshold (useful for fraud where recall matters most)
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Classification report
    print(f"\nThreshold used: {threshold}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"],
                                 digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    # Summary metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    print(f"\nROC-AUC : {roc_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Best threshold by F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx  = np.argmax(f1_scores)
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    print(f"Best threshold (max F1={f1_scores[best_idx]:.4f}): {best_thr:.4f}")
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.savefig(f"roc_curve_{name.replace(' ', '_')}.png")
    plt.show()

    return {"model": name, "roc_auc": roc_auc, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def cross_validate_model(name: str, model, X_train, y_train, cv: int = 5):
    """Stratified K-Fold cross-validation on training data."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"\n[CV {cv}-fold] {name}  ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────

def show_feature_importance(model: RandomForestClassifier, feature_names, top_n: int = 10):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    print(f"\nTop {top_n} Features (Random Forest):")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<20s}  {importances[idx]:.4f}")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    # ── Load ──────────────────────────────────
    # To use the real Kaggle dataset:
    #   df = load_data("creditcard.csv")
    df = load_data("creditcard.csv")  # synthetic demo

    # ── Preprocess ────────────────────────────
    X_train, X_test, y_train, y_test = preprocess(df)

    # ── Handle imbalance ──────────────────────
    print("\n--- Imbalance handling ---")
    # Choose ONE of the three strategies below:
    X_bal, y_bal = try_smote(X_train, y_train)          # Option A: SMOTE (recommended)
    # X_bal, y_bal = apply_oversampling(X_train, y_train)  # Option B: random oversample
    # X_bal, y_bal = apply_undersampling(X_train, y_train) # Option C: random undersample
    # X_bal, y_bal = X_train, y_train  # Option D: use class_weight only (no resampling)

    # ── Train ─────────────────────────────────
    print("\n--- Training models ---")
    lr = train_logistic_regression(X_bal, y_bal)
    rf = train_random_forest(X_bal, y_bal)
    joblib.dump(rf, "fraud_model.pkl")
    joblib.dump(rf, "best_model.pkl")
    print("Model saved as fraud_model.pkl")
    print("Logistic Regression trained ")
    print("Random Forest trained       ")

    # ── Cross-validate ────────────────────────
    print("\n--- Cross-validation (on balanced train set) ---")
    # cross_validate_model("Logistic Regression", lr, X_bal, y_bal)
    # cross_validate_model("Random Forest",       rf, X_bal, y_bal)

    # ── Evaluate on held-out test set ─────────
    # Using threshold=0.3 to boost recall (catches more fraud at cost of more FP)
    results_lr = evaluate_model("Logistic Regression", lr, X_test, y_test, threshold=0.3)
    results_rf = evaluate_model("Random Forest",       rf, X_test, y_test, threshold=0.3)

    # ── Feature importance ────────────────────
    show_feature_importance(rf, X_train.columns, top_n=10)

    # ── Summary ───────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    for r in [results_lr, results_rf]:
        print(f"  {r['model']:<25s}  ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
              f"TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")

    winner = results_rf if results_rf["roc_auc"] >= results_lr["roc_auc"] else results_lr
    print(f"\n   Recommended model: {winner['model']} (ROC-AUC {winner['roc_auc']:.4f})")
    print(f"{'='*60}\n")
    print("\n--- Sample Transaction Prediction ---")

    sample_transaction = X_test.iloc[[0]]

    prediction = rf.predict(sample_transaction)[0]

    if prediction == 1:
      print("Prediction: FRAUD")
    else:
      print("Prediction: GENUINE")


if __name__ == "__main__":
    main()