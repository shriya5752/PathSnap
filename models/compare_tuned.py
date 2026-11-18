
from pathlib import Path
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from preprocessing.create_features import FEATURE_COLUMNS
from models.train_cv import load_data, evaluate
from models.evaluate_student import load_student_test
from models.evaluate_student import evaluate as evaluate_student


OUTPUT_PATH = Path("results/person2")
PARAMS_PATH = OUTPUT_PATH / "bayesian_best_params.csv"
SEED = 42


def get_models():
    params = pd.read_csv(PARAMS_PATH).iloc[0]

    baseline = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=31,
        random_state=SEED
    )

    tuned = HistGradientBoostingClassifier(
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        learning_rate=float(params["learning_rate"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        random_state=SEED
    )

    return {
        "Untuned": baseline,
        "Tuned": tuned
    }


def cross_validate_models(data):
    x = data[FEATURE_COLUMNS]
    y = data["label"].astype(int)
    groups = data["trip_id"]

    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(x, y, groups))

    results = []

    for name, model in get_models().items():
        print(f"\n{name} model", flush=True)

        for fold, (train_idx, valid_idx) in enumerate(
            splits, start=1
        ):
            train = data.iloc[train_idx]
            validation = data.iloc[valid_idx]

            if not set(train["trip_id"]).isdisjoint(
                set(validation["trip_id"])
            ):
                raise ValueError("Trip leakage detected")

            metrics = evaluate(
                model, train, validation
            )

            results.append({
                "configuration": name,
                "fold": fold,
                **metrics
            })

            print(
                f"Fold {fold}: "
                f"F1={metrics['macro_f1']:.4f}, "
                f"Top-1={metrics['top1_accuracy']:.4f}",
                flush=True
            )

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_PATH / "tuned_vs_untuned_cv.csv",
        index=False
    )

    print("\nCROSS-VALIDATION COMPARISON")
    print(
        output.groupby("configuration")[
            ["macro_f1", "top1_accuracy", "log_loss"]
        ].mean().round(5)
    )


def student_test_comparison(train, test):
    results = []

    for name, model in get_models().items():
        metrics = evaluate_student(
            name, model, train, test
        )

        results.append({
            "configuration": name,
            **metrics
        })

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_PATH / "tuned_vs_untuned_student_test.csv",
        index=False
    )

    print("\nSTUDENT-TEST COMPARISON")
    print(output.to_string(index=False))


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train = load_data()
    test = load_student_test()

    print("Training trips:", train["trip_id"].nunique())
    print("Student-test trips:", test["trip_id"].nunique())

    cross_validate_models(train)
    student_test_comparison(train, test)

    print("\nCOMPARISON COMPLETE")


if __name__ == "__main__":
    main()

