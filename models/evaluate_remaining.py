
from pathlib import Path
import pandas as pd

from models.train_cv import load_data, get_models
from models.evaluate_student import load_student_test, evaluate


OUTPUT_PATH = Path("results/person2")


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train = load_data()
    test = load_student_test()

    selected = [
        "Logistic Regression",
        "Decision Tree"
    ]

    models = get_models()
    results = []

    for name in selected:
        metrics = evaluate(
            name,
            models[name],
            train,
            test
        )

        results.append(metrics)

        pd.DataFrame(results).to_csv(
            OUTPUT_PATH / "student_test_remaining_models.csv",
            index=False
        )

    print("\nEVALUATION COMPLETE")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
