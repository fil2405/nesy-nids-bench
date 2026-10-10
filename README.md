# nesy-nids-bench

Benchmark suite comparing a Logic Tensor Network (LTN) detector with matched
machine learning and deep learning baselines for network intrusion detection.

Status: under construction.

## Layout
- `configs/`  one YAML file per experiment run
- `kb/`       versioned knowledge bases (axiom sets), one per dataset
- `data/`     `raw/` original datasets (not committed), `processed/` splits
- `src/nesybench/`  data loading, models, logic, utilities
- `results/`  one JSON file per run (not committed)
- `analysis/` statistical tests, tables and figures
- `scripts/`  environment check and run helpers

## Principle
One run = one configuration + one seed = one result file.
Every table is regenerated from `results/`.

## Setup
Linux or WSL2 (Ubuntu), Python 3.10-3.11.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python scripts/check_env.py
    pip freeze > requirements.lock

## GPU on WSL2
TensorFlow may not find libcusolver. Fix (once, after creating the venv):

    NV=$(python -c "import nvidia,os;print(chr(58).join(os.path.join(p,d,chr(108)+chr(105)+chr(98)) for p in nvidia.__path__ for d in os.listdir(p) if os.path.isdir(os.path.join(p,d,chr(108)+chr(105)+chr(98)))))")
    echo "export LD_LIBRARY_PATH=$NV:\$LD_LIBRARY_PATH" >> <venv>/bin/activate

