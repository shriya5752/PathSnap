# PathSnap

GPS map matching project. This initial contribution implements Person 1 (dataset exploration, features, weak labels and maps). Models, decoder, API and UI are separate team contributions.

## Quick start

Extract the supplied Student folder ZIP outside the repository or into an ignored directory. It must contain `train/` and `student_test/`, with numbered trip directories.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-person1.txt
python -m preprocessing.explore_data --data "../Student folder" --output results/person1
python -m visualizations.create_maps "../Student folder/train/00000001" results/person1/train_map.html
python -m unittest discover -s tests
```

Open the generated HTML in a browser. Basemap tiles require network access. No raw data or row-level derived GPS files are committed publicly. Team members regenerate locally from the same supplied dataset.

See [Person 1 handoff](preprocessing/HANDOFF.md) for the interface, label assumptions and evaluation rules.
