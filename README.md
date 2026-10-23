# nesy-nids-bench

Benchmark suite comparing a Logic Tensor Network (LTN) detector with matched
machine learning and deep learning baselines for network intrusion detection.

Status: under construction.

## Layout
- `configs/`  one YAML file per experiment run
- `kb/`       versioned knowledge bases, one per dataset
- `data/`     `raw/` original datasets, `interim/` cleaned copies of large
              datasets (both not committed). Splits are not stored: they
              are rebuilt from the seed.
- `src/nesybench/`  data loading, models, logic, utilities
- `results/`  one JSON file per run (not committed)
- `analysis/` statistical tests, tables and figures
- `scripts/`  environment check and run helpers
- `thesis/`   LaTeX source of the thesis

## Principle
One run = one configuration + one seed = one result file.
Every table is regenerated from `results/`.

## Data
Datasets go in `data/raw/<name>/` (e.g. `data/raw/nsl-kdd/KDDTrain+.txt`).
Another location can be set with the `NESYBENCH_DATA` environment variable.
A dataset is added by writing its loader and `DatasetSpec` in
`src/nesybench/data/` and registering it in `src/nesybench/data/__init__.py`.

## Running
    python -m nesybench.run --config configs/example.yaml
    python -m nesybench.run --config configs/example.yaml --seeds 0 1

Each seed writes `results/<run_id>_seed<seed>.json` (metrics per part, timings,
training information, configuration, commit). Existing files are not replaced
unless `--overwrite` is given. The test part is evaluated only if the
configuration has `evaluate_test: true` or `--evaluate-test` is given.
`--resume` skips the seeds that already have a result file (same configuration
required); `--save-model-dir DIR` writes the model and the preprocessing of each
run under `DIR/<run_id>_seed<seed>/`.

A model is added by writing its module in `src/nesybench/models/` and
registering it in `src/nesybench/models/__init__.py`.

## Tests
    python -m pytest tests -v

Tests that need a dataset are skipped when it is not in `data/raw/`.

## Setup
Linux or WSL2 (tested on Debian), Python 3.11.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e . --no-deps
    python scripts/check_env.py
    pip freeze > requirements.lock

## GPU on WSL2
TensorFlow may not find libcusolver. Fix (once, after creating the venv):

    NV=$(python -c "import nvidia,os;print(chr(58).join(os.path.join(p,d,chr(108)+chr(105)+chr(98)) for p in nvidia.__path__ for d in os.listdir(p) if os.path.isdir(os.path.join(p,d,chr(108)+chr(105)+chr(98)))))")
    echo "export LD_LIBRARY_PATH=$NV:\$LD_LIBRARY_PATH" >> <venv>/bin/activate

