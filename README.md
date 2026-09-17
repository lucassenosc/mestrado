# Binarization as an Attack Surface in Weightless Neural Networks

Reproducibility repository for the intrusion-detection experiments associated with **“Binarization as an Attack Surface in Weightless Neural Networks.”**

The repository contains the experimental pipelines for **Standard WiSARD** and **Bloom WiSARD** on the **Bot-IoT**, **CICIDS2017**, and **UNSW-NB15** intrusion-detection datasets, together with preprocessing, report aggregation, curve-generation utilities, local WiSARD/Bloom implementations, and environment specifications.

## Scope

The paper's IDS experiments evaluate Standard WiSARD and Bloom WiSARD under binary and multiclass classification. The paper-reported IDS perturbations are:

- **FGSM** (`L∞`, `ε = 0.3`), generated on a differentiable surrogate and transferred to the WNN;
- **Random L∞** (`ε = 0.3`);
- **Random L2** (`ε = 3.0`).

The Random L∞ and Random L2 methods are model-agnostic sensitivity probes.

The repository also contains auxiliary Decision Tree and Random Forest notebooks. These notebooks are retained as exploratory baselines, but they are **not required to reproduce the IDS results reported in the current paper**.

A locally adapted `carlini_wagner_l2.py` implementation is also retained for compatibility with previous C&W experiments. The current paper does **not** use the unverified C&W results in its reported IDS analyses, so C&W is not required to reproduce the paper's IDS tables and figures.

## Repository Structure

```text
.
├── analysis/
│   ├── aggregate_reports.ipynb
│   └── curva_roc.ipynb
│
├── data/                         # Not versioned; downloaded separately
│   ├── botiot/
│   ├── cicids2017/
│   └── unsw_nb15/
│
├── docs/
│   ├── ambiente.md
│   ├── dados.md
│   └── execucao.md
│
├── experiments/
│   ├── baselines/
│   │   ├── decision_tree.ipynb
│   │   └── random_forest.ipynb
│   │
│   └── wisard/
│       ├── bloom_wisard.ipynb
│       └── standart_wisard.ipynb
│
├── libs/
│   ├── bloom_filter.py
│   ├── carlini_wagner_l2.py
│   └── wisard.py
│
├── preprocessing/
│   ├── prepare_botiot.ipynb
│   └── prepare_cicids2017.ipynb
│
├── results/
│   ├── aggregated_reports/
│   ├── curves_data/
│   └── reports/
│
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

`__pycache__/` and virtual-environment directories are local artifacts and are intentionally excluded from version control.

## Main Components

### WiSARD experiments

- `experiments/wisard/standart_wisard.ipynb`  
  Standard WiSARD experimental pipeline.

- `experiments/wisard/bloom_wisard.ipynb`  
  Bloom WiSARD experimental pipeline.

Both notebooks handle dataset loading, preprocessing, perturbation generation, binarization, model evaluation, metrics, and result export.

### Data preprocessing

- `preprocessing/prepare_botiot.ipynb`  
  Builds the Bot-IoT training and test files from the reduced Bot-IoT CSV files.

- `preprocessing/prepare_cicids2017.ipynb`  
  Builds the CICIDS2017 training and test files from the cleaned CICIDS2017 dataset.

The UNSW-NB15 train/test files are expected to already be present in `data/unsw_nb15/`.

### Analysis

- `analysis/aggregate_reports.ipynb`  
  Reads individual CSV files from `results/reports/` and writes grouped outputs to `results/aggregated_reports/`.

- `analysis/curva_roc.ipynb`  
  Reads the saved curve data from `results/curves_data/` and generates comparison curves.

### Local libraries

- `libs/wisard.py`  
  Bloom WiSARD implementation used by the Bloom WiSARD notebook.

- `libs/bloom_filter.py`  
  Bloom-filter implementation used by `libs/wisard.py`.

- `libs/carlini_wagner_l2.py`  
  Locally adapted CleverHans C&W L2 implementation retained for compatibility with previous experiments. It is not required for the paper-reported IDS results.

## Reference Environment

The validated reference environment uses:

```text
Python 3.13.12
```

Two dependency files are provided:

- `requirements.txt`: direct project dependencies;
- `requirements-lock.txt`: complete dependency snapshot generated from a clean, validated reproduction environment.

For the closest reproduction of the validated environment, use `requirements-lock.txt`.

## 1. Clone the Repository

```bash
git clone https://github.com/lucassenosc/mestrado.git
cd mestrado
```

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

The expected Python version is:

```text
Python 3.13.12
```

### Linux / macOS

If Python 3.13 is installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python --version
```

The lock file was validated in the Windows reference environment. On another operating system, `requirements.txt` may be preferable if a platform-specific locked dependency cannot be resolved.

## 3. Install Dependencies

Upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

For the validated locked environment:

```bash
python -m pip install -r requirements-lock.txt
```

Alternatively, install only the declared direct dependencies:

```bash
python -m pip install -r requirements.txt
```

Check the resulting environment:

```bash
python -m pip check
```

A healthy environment should report:

```text
No broken requirements found.
```

## 4. Download the Datasets

Large datasets are intentionally excluded from Git because of their size.

Download the dataset files from:

[Download the datasets from Google Drive](https://drive.google.com/drive/folders/1xaqv2979bKpBT411Ifh70MKPlZ1Dz0id?usp=drive_link)

Place the files under:

```text
data/
├── botiot/
├── cicids2017/
└── unsw_nb15/
```

The experimental notebooks expect the prepared splits to follow this organization:

```text
data/
├── botiot/
│   ├── BotIoT_training-set.csv
│   └── BotIoT_testing-set.csv
│
├── cicids2017/
│   ├── CICIDS_training-set.csv
│   └── CICIDS_testing-set.csv
│
└── unsw_nb15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

If the prepared Bot-IoT or CICIDS2017 splits are not available, generate them with the preprocessing notebooks described below.

## 5. Prepare the Data if Necessary

### Bot-IoT

Place the reduced source files in:

```text
data/botiot/
```

Then run:

```text
preprocessing/prepare_botiot.ipynb
```

The notebook generates:

```text
data/botiot/BotIoT_training-set.csv
data/botiot/BotIoT_testing-set.csv
```

### CICIDS2017

Place:

```text
data/cicids2017/cicids2017_cleaned.csv
```

and run:

```text
preprocessing/prepare_cicids2017.ipynb
```

The notebook generates:

```text
data/cicids2017/CICIDS_training-set.csv
data/cicids2017/CICIDS_testing-set.csv
```

## 6. Run the WiSARD Experiments

Open either notebook directly in Jupyter or VS Code:

```text
experiments/wisard/standart_wisard.ipynb
experiments/wisard/bloom_wisard.ipynb
```

Select the project's `.venv` as the notebook kernel and use **Run All**.

The notebooks include project-root discovery logic, so they can be opened from their current subdirectories while still resolving `data/`, `libs/`, and `results/` from the repository root.

Before running a full experiment, review the experiment control panel near the beginning of each notebook. It controls items such as:

```text
DATASETS_TO_RUN
RUN_BINARY
RUN_MULTICLASS
ENCODING_TYPES
ATTACKS_TO_RUN
```

For paper-aligned IDS reproduction, the reported perturbation methods are:

```text
FGSM
RANDOM_LINF
RANDOM_L2
```

The notebooks may contain additional optional/legacy experimental code, but such configurations are not required to reproduce the reported IDS results.

## 7. Results

Individual experiment reports are written to:

```text
results/reports/
```

Curve data used by the analysis notebook are written to:

```text
results/curves_data/
```

Aggregated reports are written to:

```text
results/aggregated_reports/
```

The result directories are intentionally **not ignored** by `.gitignore`, allowing experimental outputs selected for the reproducibility artifact to be versioned.

## 8. Aggregate Reports

After generating individual experiment CSVs, run:

```text
analysis/aggregate_reports.ipynb
```

The aggregation flow is:

```text
results/reports/
        ↓
aggregate_reports.ipynb
        ↓
results/aggregated_reports/
```

The aggregator groups experiment files according to the filename convention used by the experimental notebooks.

## 9. Generate ROC / PR Comparisons

Run:

```text
analysis/curva_roc.ipynb
```

It reads the `.npz` files stored in:

```text
results/curves_data/
```

and generates comparison plots from the stored clean and perturbed predictions/scores.

## 10. Auxiliary Baselines

The repository currently retains:

```text
experiments/baselines/decision_tree.ipynb
experiments/baselines/random_forest.ipynb
```

These are auxiliary exploratory baselines. They are not required for reproduction of the IDS results reported in the current paper.

## Reproducibility Checklist

A reviewer reproducing the IDS experiments should:

1. Clone the repository.
2. Use Python 3.13.12.
3. Create and activate a fresh virtual environment.
4. Install `requirements-lock.txt` for the validated environment.
5. Run `python -m pip check`.
6. Download the three IDS datasets and place them under `data/`.
7. Generate the Bot-IoT/CICIDS2017 prepared splits if necessary.
8. Select the new virtual environment as the Jupyter kernel.
9. Run the Standard WiSARD and/or Bloom WiSARD notebook.
10. Aggregate generated CSV reports with `analysis/aggregate_reports.ipynb`.
11. Generate comparison curves with `analysis/curva_roc.ipynb` when needed.

## Important Scope Note

The repository structure documented above corresponds to the **IDS experimental pipeline** currently present in this repository.

The paper also contains separate MNIST experiments, including threshold-crossing/certificate diagnostics, Simulated Annealing, and surrogate-transfer experiments. Those experiments require their corresponding implementation files to be included in the reproducibility artifact if the repository is intended to reproduce the **entire paper**, rather than only its IDS component.

## License / Citation

If this repository is used in academic work, please cite the associated paper:

> *Binarization as an Attack Surface in Weightless Neural Networks.*

Citation metadata can be added here once the final publication information is available.
