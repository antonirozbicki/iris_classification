#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris as sk_load_iris
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score,
    GridSearchCV,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


RANDOM_STATE = 42


@dataclass
class Config:
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = RANDOM_STATE
    output_models_dir: Path = Path("models")
    output_reports_dir: Path = Path("reports")
    output_figures_dir: Path = Path("reports/figures")


def set_style() -> None:
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    except Exception:
        pass
    mpl.rcParams.update({
        "figure.titlesize": 16,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_iris_df() -> Tuple[pd.DataFrame, pd.Series]:
    iris = sk_load_iris(as_frame=True)
    df = iris.frame.copy()
    target_map = {i: name for i, name in enumerate(iris.target_names)}
    df["target"] = df["target"].map(target_map)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def eda_histograms(X: pd.DataFrame, out_dir: Path) -> None:
    set_style()
    try:
        import seaborn as sns
        features = list(X.columns)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, col in zip(axes.flat, features):
            sns.histplot(data=X, x=col, bins=20, kde=True, stat="count",
                         edgecolor="white", alpha=0.9, ax=ax)
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        fig.suptitle("Feature Distributions (Iris)", y=1.02)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "hist_features.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Skipped histograms: {e}")


def eda_pairplot(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    set_style()
    try:
        import seaborn as sns
        df_plot = X.copy()
        df_plot["target"] = y.values
        g = sns.pairplot(
            df_plot, hue="target", diag_kind="hist", palette="Set2",
            corner=False, plot_kws={"alpha": 0.8, "edgecolor": "white", "s": 45},
            diag_kws={"edgecolor": "white", "linewidth": 0.5}, height=2.4
        )
        g.fig.suptitle("Pairwise Feature Relationships (Iris)", y=1.02, fontsize=16, fontweight="semibold")
        out_dir.mkdir(parents=True, exist_ok=True)
        g.savefig(out_dir / "pairplot.png", dpi=150)
        plt.close(g.fig)
    except Exception as e:
        print(f"[WARN] Skipped pairplot: {e}")


def build_models() -> Dict[str, Pipeline]:
    scaler = StandardScaler()
    models_scaled = {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "svc": SVC(probability=True, random_state=RANDOM_STATE),
    }
    pipes: Dict[str, Pipeline] = {
        name: Pipeline(steps=[("scaler", scaler), ("clf", model)])
        for name, model in models_scaled.items()
    }
    pipes["rf"] = Pipeline(steps=[("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
    return pipes


def cv_compare_models(
    pipes: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, pipe in pipes.items():
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        rows.append({"model": name, "cv_mean_acc": float(scores.mean()),
                     "cv_std_acc": float(scores.std()), "folds": n_splits})
    return pd.DataFrame(rows).sort_values("cv_mean_acc", ascending=False).reset_index(drop=True)


def grid_search_best(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    base_pipe = Pipeline(steps=[("scaler", StandardScaler()), ("clf", SVC(probability=True))])
    param_grid = [
        {
            "clf": [SVC(probability=True, random_state=RANDOM_STATE)],
            "clf__kernel": ["rbf", "linear"],
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"],
        },
        {
            "clf": [RandomForestClassifier(random_state=RANDOM_STATE)],
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 3, 5],
            "clf__min_samples_split": [2, 4],
        },
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, out_dir: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Confusion Matrix – Test Set", fontsize=16, fontweight="semibold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(gs: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path) -> None:
    set_style()
    try:
        import seaborn as sns
        classes = np.unique(y_test)
        y_bin = label_binarize(y_test, classes=classes)
        if hasattr(gs.best_estimator_, "predict_proba"):
            y_score = gs.best_estimator_.predict_proba(X_test)
        else:
            y_score = gs.best_estimator_.decision_function(X_test)
        if isinstance(y_score, list):
            y_score = np.column_stack(y_score)
        palette = sns.color_palette("Set2", n_colors=len(classes))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2.5, label=f"{cls} (AUC = {roc_auc:.3f})", color=palette[i])
        ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="gray", label="Chance")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest) – Test Set")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "roc_curves.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Skipped ROC curves: {e}")


def plot_learning_curve(gs: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series, out_dir: Path) -> None:
    set_style()
    try:
        import seaborn as sns
        best_model = gs.best_estimator_
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True, random_state=RANDOM_STATE
        )
        train_mean = train_scores.mean(axis=1); train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1); val_std = val_scores.std(axis=1)
        colors = sns.color_palette("Set2", 2)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_mean, marker="o", linewidth=2, markersize=5, label="Training score", color=colors[0])
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=colors[0])
        ax.plot(train_sizes, val_mean, marker="o", linewidth=2, markersize=5, label="Validation score", color=colors[1])
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color=colors[1])
        ax.set_xlabel("Training examples"); ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve – Best Estimator"); ax.legend(frameon=False)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "learning_curve.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Skipped learning curve: {e}")


def plot_permutation_importance(gs: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path) -> None:
    set_style()
    try:
        import seaborn as sns
        result = permutation_importance(
            gs.best_estimator_, X_test, y_test, scoring="accuracy",
            n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
        )
        imp_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values("importance_mean", ascending=True)
        imp_df["importance_std"] = imp_df["importance_std"].fillna(0.0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            imp_df["feature"], imp_df["importance_mean"],
            xerr=imp_df["importance_std"], height=0.6, alpha=0.9,
            edgecolor="white", linewidth=1.0, capsize=4, color=sns.color_palette("Set2")[2]
        )
        ax.set_xlabel("Permutation Importance (mean ± std)")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance (Permutation) – Test Set")
        ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "feature_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Skipped permutation importance: {e}")


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    print("\n=== Test Set Results ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, out_dir)
    return {"accuracy": float(acc), "balanced_accuracy": float(bacc)}


def main():
    parser = argparse.ArgumentParser(description="Iris Classification – baseline + GridSearch (scikit-learn dataset)")
    parser.add_argument("--plots", action="store_true", help="Save EDA plots (histograms + pairplot)")
    parser.add_argument("--with-roc", action="store_true", help="Save ROC curves (OVR)")
    parser.add_argument("--with-learning-curve", action="store_true", help="Save learning curve")
    parser.add_argument("--with-permutation-importance", action="store_true", help="Save permutation importance plot")
    parser.add_argument("--save-model", action="store_true", help="Persist best model to models/")
    args = parser.parse_args()

    cfg = Config()
    cfg.output_models_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_figures_dir.mkdir(parents=True, exist_ok=True)

    set_style()

    X, y = load_iris_df()
    feature_names = list(X.columns)

    if args.plots:
        eda_histograms(X, cfg.output_figures_dir)
        eda_pairplot(X, y, cfg.output_figures_dir)
        print(f"[INFO] EDA figures saved to: {cfg.output_figures_dir}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    pipes = build_models()
    cv_table = cv_compare_models(pipes, X_train, y_train, n_splits=cfg.n_splits, seed=cfg.random_state)
    print("\n=== Cross-Validation Model Comparison (Accuracy) ===")
    print(cv_table.to_string(index=False))

    gs = grid_search_best(X_train, y_train)
    print("\n=== Best Model from GridSearch ===")
    print(f"Estimator: {gs.best_estimator_}")
    print(f"CV accuracy: {gs.best_score_:.4f}")
    print("Best params:")
    for k, v in gs.best_params_.items():
        print(f"  {k}: {v}")

    metrics = evaluate(gs.best_estimator_, X_test, y_test, cfg.output_reports_dir)

    if args.with_roc:
        plot_roc_curves(gs, X_test, y_test, cfg.output_figures_dir)
    if args.with_learning_curve:
        plot_learning_curve(gs, X_train, y_train, cfg.output_figures_dir)
    if args.with_permutation_importance:
        plot_permutation_importance(gs, X_test, y_test, cfg.output_figures_dir)

    if args.save_model:
        model_path = cfg.output_models_dir / "best_model.joblib"
        joblib.dump(
            {
                "model": gs.best_estimator_,
                "feature_names": feature_names,
                "metrics_test": metrics,
                "best_cv_accuracy": float(gs.best_score_),
                "best_params": gs.best_params_,
                "data_source": "scikit-learn built-in iris dataset",
            },
            model_path,
        )
        print(f"[INFO] Model saved to: {model_path}")


if __name__ == "__main__":
    main()
