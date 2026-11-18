
from pathlib import Path
import time
import pandas as pd
import optuna

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from preprocessing.create_features import FEATURE_COLUMNS
from models.train_cv import load_data


OUTPUT_PATH = Path("results/person2")
SEED = 42
N_TRIALS = 20


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    data = load_data()

    x = data[FEATURE_COLUMNS]
    y = data["label"].astype(int)
    groups = data["trip_id"]

    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(x, y, groups))

    def objective(trial):
        params = {
            "max_iter": trial.suggest_int(
                "max_iter", 50, 200
            ),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 15, 63
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 10, 60
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-5, 1.0, log=True
            )
        }

        scores = []

        for train_idx, valid_idx in splits:
            model = HistGradientBoostingClassifier(
                **params,
                random_state=SEED
            )

            model.fit(
                x.iloc[train_idx],
                y.iloc[train_idx]
            )

            predictions = model.predict(
                x.iloc[valid_idx]
            )

            score = f1_score(
                y.iloc[valid_idx],
                predictions,
                average="macro",
                zero_division=0
            )

            scores.append(score)

        return sum(scores) / len(scores)

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_hgb_sampled",
        storage=f"sqlite:///{OUTPUT_PATH / 'optuna_trials.db'}",
        load_if_exists=True
    )

    remaining = max(0, N_TRIALS - len(study.trials))

    print("Existing trials:", len(study.trials))
    print("Remaining trials:", remaining)

    start = time.perf_counter()

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True
        )

    elapsed = time.perf_counter() - start

    trials = study.trials_dataframe()
    trials.to_csv(
        OUTPUT_PATH / "bayesian_trials.csv",
        index=False
    )

    best = pd.DataFrame([{
        "best_cv_macro_f1": study.best_value,
        "total_trials": len(study.trials),
        "run_seconds": elapsed,
        **study.best_params
    }])

    best.to_csv(
        OUTPUT_PATH / "bayesian_best_params.csv",
        index=False
    )

    print("\nOPTIMISATION COMPLETE")
    print("Best CV Macro F1:", study.best_value)
    print("Best parameters:", study.best_params)
    print("Trials recorded:", len(study.trials))
    print("This run took:", round(elapsed, 2), "seconds")


if __name__ == "__main__":
    main()

