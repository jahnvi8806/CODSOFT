# =============================================================================
# TASK 1: TITANIC SURVIVAL PREDICTION
# =============================================================================
# Predicts whether a passenger survived the Titanic disaster using
# passenger data (age, gender, ticket class, fare, etc.)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)

df = pd.read_csv("Titanic-Dataset.csv")

print(f"\n✔ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n── Column Overview ──")
print(df.dtypes)
print("\n── Missing Values ──")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n── Basic Statistics ──")
print(df.describe())

overall_survival = df["Survived"].mean() * 100
print(f"\n✔ Overall survival rate: {overall_survival:.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic Dataset – Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Survival count
sns.countplot(x="Survived", data=df, palette=["#e74c3c", "#2ecc71"], ax=axes[0, 0])
axes[0, 0].set_title("Survival Count")
axes[0, 0].set_xticklabels(["Did Not Survive (0)", "Survived (1)"])
for bar in axes[0, 0].patches:
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    int(bar.get_height()), ha="center", fontweight="bold")

# Survival by gender
sns.countplot(x="Sex", hue="Survived", data=df,
              palette=["#e74c3c", "#2ecc71"], ax=axes[0, 1])
axes[0, 1].set_title("Survival by Gender")
axes[0, 1].legend(["Did Not Survive", "Survived"])

# Survival by class
sns.countplot(x="Pclass", hue="Survived", data=df,
              palette=["#e74c3c", "#2ecc71"], ax=axes[0, 2])
axes[0, 2].set_title("Survival by Passenger Class")
axes[0, 2].legend(["Did Not Survive", "Survived"])

# Age distribution
df["Age"].dropna().plot(kind="hist", bins=30, color="#3498db",
                         edgecolor="white", ax=axes[1, 0])
axes[1, 0].set_title("Age Distribution")
axes[1, 0].set_xlabel("Age")

# Fare distribution (log scale for clarity)
df["Fare"].plot(kind="hist", bins=40, color="#9b59b6",
                edgecolor="white", ax=axes[1, 1])
axes[1, 1].set_title("Fare Distribution")
axes[1, 1].set_xlabel("Fare")

# Survival rate by embarkation port
survival_by_embark = df.groupby("Embarked")["Survived"].mean().reset_index()
axes[1, 2].bar(survival_by_embark["Embarked"], survival_by_embark["Survived"],
               color=["#e67e22", "#1abc9c", "#e74c3c"])
axes[1, 2].set_title("Survival Rate by Embarkation Port")
axes[1, 2].set_ylabel("Survival Rate")
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔ EDA plots saved → eda_plots.png")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n── Feature Engineering ──")

data = df.copy()

# Fill missing Age with median grouped by Pclass + Sex
data["Age"] = data.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Fill missing Embarked with mode
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Fill missing Fare with median
data["Fare"].fillna(data["Fare"].median(), inplace=True)

# Title extraction from Name
data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.")
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Lady": "Royalty",
    "Countess": "Royalty", "Jonkheer": "Royalty", "Don": "Royalty",
    "Sir": "Royalty", "Capt": "Officer"
}
data["Title"] = data["Title"].map(title_map).fillna("Other")

# Family size
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

# Age bins
data["AgeBand"] = pd.cut(data["Age"], bins=[0, 12, 18, 35, 60, 120],
                          labels=["Child", "Teenager", "YoungAdult", "Adult", "Senior"])

# Fare bins
data["FareBand"] = pd.qcut(data["Fare"], q=4,
                             labels=["Low", "Medium", "High", "VeryHigh"])

# Has Cabin indicator
data["HasCabin"] = data["Cabin"].notna().astype(int)

print("✔ New features created: Title, FamilySize, IsAlone, AgeBand, FareBand, HasCabin")

# ─────────────────────────────────────────────
# 4. ENCODING & FEATURE SELECTION
# ─────────────────────────────────────────────
encode_cols = ["Sex", "Embarked", "Title", "AgeBand", "FareBand"]
le = LabelEncoder()
for col in encode_cols:
    data[col] = le.fit_transform(data[col].astype(str))

features = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "Title", "FamilySize", "IsAlone",
    "AgeBand", "FareBand", "HasCabin"
]

X = data[features]
y = data["Survived"]

print(f"✔ Selected {len(features)} features: {features}")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✔ Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 6. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
print("\n── Model Training & Cross-Validation ──")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring="accuracy")
    model.fit(X_tr, y_train)
    X_te = X_test_scaled if name == "Logistic Regression" else X_test
    y_pred = model.predict(X_te)
    test_acc = accuracy_score(y_test, y_pred)
    roc     = roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])
    results[name] = {
        "model": model, "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(), "test_acc": test_acc, "roc_auc": roc
    }
    print(f"  {name:<25}  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  "
          f"Test: {test_acc:.4f}  AUC: {roc:.4f}")

# ─────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING (Best Model → Random Forest)
# ─────────────────────────────────────────────
print("\n── Hyperparameter Tuning (Random Forest) ──")

param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
best_acc = accuracy_score(y_test, y_pred_best)
best_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print(f"✔ Best parameters: {grid_search.best_params_}")
print(f"✔ Tuned RF  →  Accuracy: {best_acc:.4f}  |  AUC: {best_auc:.4f}")

# ─────────────────────────────────────────────
# 8. EVALUATION & VISUALISATIONS
# ─────────────────────────────────────────────
print("\n── Classification Report (Tuned Random Forest) ──")
print(classification_report(y_test, y_pred_best, target_names=["Did Not Survive", "Survived"]))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation – Tuned Random Forest", fontsize=14, fontweight="bold")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
axes[1].plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {best_auc:.4f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")

# Feature Importance
importances = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=True)
importances.plot(kind="barh", color="#27ae60", edgecolor="white", ax=axes[2])
axes[2].set_title("Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔ Evaluation plots saved → model_evaluation.png")

# ─────────────────────────────────────────────
# 9. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────
model_names  = list(results.keys()) + ["RF (Tuned)"]
test_accs    = [v["test_acc"] for v in results.values()] + [best_acc]
roc_aucs     = [v["roc_auc"] for v in results.values()]  + [best_auc]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width / 2, test_accs, width, label="Test Accuracy",
               color="#3498db", edgecolor="white")
bars2 = ax.bar(x + width / 2, roc_aucs,  width, label="ROC-AUC",
               color="#e67e22", edgecolor="white")

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_title("Model Comparison – Accuracy & AUC", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("Score")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔ Model comparison chart saved → model_comparison.png")

# ─────────────────────────────────────────────
# 10. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Best Model      : Tuned Random Forest")
print(f"  Test Accuracy   : {best_acc * 100:.2f}%")
print(f"  ROC-AUC Score   : {best_auc:.4f}")
print(f"  Best Parameters : {grid_search.best_params_}")
print("=" * 60)
print("\n✔ All outputs saved. Task complete!")