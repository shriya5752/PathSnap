
from pathlib import Path
import pandas as pd

from models.train_cv import load_data, get_models
from models.evaluate_student import load_student_test, evaluate


OUTPUT_PATH = Path("results/person2")


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train = load_data()
    test = load_student_test()

    model = get_models()["KNN"]

    results = evaluate(
        "KNN",
        model,
        train,
        test
    )

    output = pd.DataFrame([results])

    output.to_csv(
        OUTPUT_PATH / "student_test_knn.csv",
        index=False
    )

    print("\nKNN EVALUATION COMPLETE")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
