# Code, Data and Results for "Limits of machine learning for automatic vulnerability detection"

## Structure

Below is an annotated map of the directory structure of this repository.

```
.
├── scripts................................ Scripts to exactly reproduce all experiments presented in our paper.
│   └── <dataset>.......................... One directory for each dataset (CodeXGLUE, VulDeePecker).
│       └── <technique>.................... One directory for each ML technique (VulBERTa, CoTexT, PLBart)
│           ├── run.py..................... Compute experimental results for selected ML technique and dataset.
│           └── run_at.py.................. Compute adversarial training results for selected ML technique and dataset. Only for CodeXGLUE.
│
├── datasets............................... All datasets that we use in the experiments for the paper + our own created dataset VulnPatchPairs.
│   └── README.md.......................... Instructions for downloading all datasets used in this repository.
│
├── models................................. All pretrained models that are not downloaded in the training scripts.
│   └── README.md.......................... Instructions for downloading all models used in this repository.
│
├── plots.................................. Generates all plots shown in the paper.
│   ├── generate_plots.py.................. Script that generates all plots from the experimental results.
│   └── plots.............................. The produced plots and tables that can be found in the paper.
│
├── additional_experiments................. Additional experiments presented in the paper.
│   └── naturalness........................ Additional experiments on naturalness of transformations.
│       └── run.py......................... Run additional experiment on naturalness.
│
├── install_requirements.sh................ Script to install Python environment and required packages.
├── requirements.txt....................... All Python packages that you need to run the experiments.
│
├── run_experiments.sh..................... Script to reproduce all experiments presented in our paper.
│
└── README.md
```

## Setup

### Step 1: Install Anaconda

Anaconda is an open-source package and environment management tool for Python. Instructions for Installation can be found [here](https://www.anaconda.com/products/distribution).

### Step 2: Install Requirements

We assume that you have Anaconda installed.

Running the following script from the root directory of this repository creates a virtual environment in Anaconda, and installs the required Python packages.

```
bash install_requirements.sh
```

Activate the environment with the following command.

```
conda activate LimitsOfMl4Vuln
```

### Step 3: Download the required datasets

Go to [datasets/README.md](https://github.com/niklasrisse/LimitsOfML4Vuln/tree/main/datasets/README.md) and follow the instructions to download all datasets needed to run our experiments.

### Step 4: Download the required models

Go to [models/README.md](https://github.com/niklasrisse/LimitsOfML4Vuln/tree/main/models/README.md) and follow the instructions to download all models needed to run our experiments.

### Step 5: Ready to go

Run

```
bash run_experiments.sh
```

to reproduce all experimental results presented in our paper. The script also serves as an entry point into the scripts for the different experiments.

## References
