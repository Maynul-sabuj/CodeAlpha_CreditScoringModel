

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_SEED = 42
DATA_PATH = "data/credit_data.csv"
TARGET_COL = "creditworthy"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=["Bad Risk (0)", "Good Risk (1)"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics, y_pred, y_proba


def plot_confusion_matrices(results, X_test, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, pipeline) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_estimator(
            pipeline, X_test, y_test,
            display_labels=["Bad Risk", "Good Risk"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved outputs/confusion_matrices.png")


def plot_roc_curves(results, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipeline in results.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_title("ROC Curves - Credit Scoring Models")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150)
    plt.close()
    print("Saved outputs/roc_curves.png")


def plot_metric_comparison(metrics_df: pd.DataFrame):
    plot_df = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Model Comparison - Credit Scoring")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150)
    plt.close()
    print("Saved outputs/model_comparison.png")


def plot_feature_importance(pipeline, numeric_features, categorical_features):
    rf = pipeline.named_steps["classifier"]
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names

    importances = pd.Series(rf.feature_importances_, index=all_feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, color="#4C72B0")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("Saved outputs/feature_importance.png")


def main():
    df = load_data()
    preprocessor, numeric_features, categorical_features = build_preprocessor(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    all_metrics = []
    fitted_pipelines = {}

    for name, model in get_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        metrics, y_pred, y_proba = evaluate_model(name, pipeline, X_test, y_test)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + "=" * 60)
    print("SUMMARY - All Models")
    print("=" * 60)
    print(metrics_df.set_index("model").round(4))

    metrics_df.to_csv("outputs/model_metrics.csv", index=False)
    print("\nSaved outputs/model_metrics.csv")

    # Plots
    plot_confusion_matrices(fitted_pipelines, X_test, y_test)
    plot_roc_curves(fitted_pipelines, X_test, y_test)
    plot_metric_comparison(metrics_df)
    plot_feature_importance(fitted_pipelines["Random Forest"], numeric_features, categorical_features)

    # Save the best model (by F1-score, a balanced metric for imbalanced classes)
    best_row = metrics_df.loc[metrics_df["f1_score"].idxmax()]
    best_name = best_row["model"]
    best_pipeline = fitted_pipelines[best_name]
    joblib.dump(best_pipeline, "models/best_credit_scoring_model.joblib")

    with open("outputs/best_model_info.json", "w") as f:
        json.dump(best_row.to_dict(), f, indent=2)

    print(f"\nBest model: {best_name} (F1-score = {best_row['f1_score']:.4f})")
    print("Saved models/best_credit_scoring_model.joblib")


if __name__ == "__main__":
    main()
