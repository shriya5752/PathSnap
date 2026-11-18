
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = Path("results/person2")
PLOTS = RESULTS / "plots"


def save_bar(data, x, y, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 5))

    data = data.sort_values(y, ascending=False)

    ax.bar(
        data[x],
        data[y],
        color="#3568a8"
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", rotation=25)

    for i, value in enumerate(data[y]):
        ax.text(
            i, value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()
    fig.savefig(PLOTS / filename, dpi=180)
    plt.close(fig)


def cv_comparison():
    df = pd.read_csv(
        RESULTS / "sampled_80trip_cv_results.csv"
    )

    summary = df.groupby("model", as_index=False)[
        "macro_f1"
    ].mean()

    save_bar(
        summary,
        "model",
        "macro_f1",
        "Five-Model Cross-Validation Comparison",
        "Mean Macro F1",
        "cv_macro_f1.png"
    )


def student_test_comparison():
    leading = pd.read_csv(
        RESULTS / "student_test_leading_models.csv"
    )

    remaining = pd.read_csv(
        RESULTS / "student_test_remaining_models.csv"
    )

    knn = pd.read_csv(
        RESULTS / "student_test_knn.csv"
    )

    df = pd.concat(
        [leading, remaining, knn],
        ignore_index=True
    )

    save_bar(
        df,
        "model",
        "top1_accuracy",
        "Student-Test Top-1 Accuracy",
        "Top-1 Accuracy",
        "student_top1.png"
    )

    save_bar(
        df,
        "model",
        "log_loss",
        "Student-Test Log Loss",
        "Log Loss",
        "student_log_loss.png"
    )


def bayesian_plot():
    df = pd.read_csv(
        RESULTS / "bayesian_trials.csv"
    )

    if "value" not in df.columns:
        raise ValueError("Optuna trial values not found")

    values = df["value"]
    best = values.cummax()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        range(1, len(values) + 1),
        values,
        marker="o",
        label="Trial Macro F1"
    )

    ax.plot(
        range(1, len(best) + 1),
        best,
        linewidth=2,
        label="Best So Far"
    )

    ax.set_title("Bayesian Hyperparameter Optimisation")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cross-Validation Macro F1")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PLOTS / "bayesian_optimisation.png",
        dpi=180
    )

    plt.close(fig)


def tuned_comparison():
    df = pd.read_csv(
        RESULTS / "tuned_vs_untuned_student_test.csv"
    )

    save_bar(
        df,
        "configuration",
        "macro_f1",
        "Tuned vs Untuned HistGradientBoosting",
        "Student-Test Macro F1",
        "tuned_vs_untuned.png"
    )


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)

    cv_comparison()
    student_test_comparison()
    bayesian_plot()
    tuned_comparison()

    print("ALL GRAPHS GENERATED")
    print("Saved in:", PLOTS)


if __name__ == "__main__":
    main()
