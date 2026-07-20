# FLPoison Project Reproduction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Platform](https://img.shields.io/badge/Platform-Ubuntu%2022.04-success)
![License](https://img.shields.io/badge/License-GPL%20v2-blue)

## Overview

This repository contains a reproduced implementation of the **FLPoison** framework, a benchmarking platform for evaluating poisoning attacks and defense mechanisms in Federated Learning (FL).

The project was reproduced as part of a **Mitacs Research Project** to verify the reproducibility of the original framework and provide a documented execution environment that can be easily followed by other researchers.

The original FLPoison framework implements multiple Federated Learning algorithms, poisoning attacks, and defense mechanisms within a unified PyTorch-based environment, enabling reproducible experiments across different datasets and learning settings.

This repository preserves the original implementation while providing an updated execution guide, validated environment configuration, and simplified testing procedure.

---
## Original Project

This repository reproduces the FLPoison framework originally developed by Zhang et al. for benchmarking poisoning attacks and defense mechanisms in Federated Learning.

- **Original Repository:** https://github.com/vio1etus/FLPoison
- **Original Paper:** *SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning* (2025)

The purpose of this repository is to reproduce, validate, and document the original implementation as part of a Mitacs research project.

---
# Project Objectives

The objectives of this reproduction are:

- Successfully reproduce the FLPoison framework.
- Validate the installation on a clean Ubuntu 22.04 environment.
- Verify that all required dependencies can be installed using Conda.
- Execute a complete Federated Learning experiment successfully.
- Provide clear documentation for future users and researchers.

---

# Main Features

The framework provides implementations of:

## Federated Learning Algorithms

- FedSGD
- FedAvg
- FedOpt

---

## Supported Datasets

- MNIST
- FashionMNIST
- EMNIST
- CIFAR-10
- CIFAR-100
- CINIC-10
- CHMNIST
- TinyImageNet

---

## Supported Models

- Logistic Regression
- SimpleCNN
- LeNet-5
- ResNet Family
- VGG Family

---

## Implemented Poisoning Attacks

### Data Poisoning Attacks

- Neurotoxin
- Edge-Case Backdoor
- Model Replacement Attack
- Alternating Minimization
- DBA
- BadNets
- Label Flipping

### Model Poisoning Attacks

- HIDRA
- Mimic Attack
- Min-Max
- Min-Sum
- Fang Attack
- IPM Attack
- ALIE Attack
- Sign Flipping
- Gaussian Attack

---

## Implemented Defenses

### Against Data Poisoning

- FLAME
- DeepSight
- CRFL
- Norm Clipping
- FoolsGold
- Auror

### Against Model Poisoning

- LASA
- FedSIGN
- FLDetector
- SignGuard
- Bucketing
- Divide-and-Conquer (DnC)
- Centered Clipping
- FLTrust
- RFA
- Bulyan
- Coordinate-wise Median
- Trimmed Mean
- Multi-Krum
- Krum
- SimpleClustering

---

# Repository Structure

```

FLPoison-project/

├── aggregators/ # Defense aggregation methods

├── attackers/ # Poisoning attack implementations

├── configs/ # Experiment configuration files

├── data/ # Dataset directory

├── datapreprocessor/ # Dataset preprocessing utilities

├── docs/ # Project documentation

├── fl/ # Federated Learning framework

├── logs/ # Generated experiment outputs

├── environment.yaml # Conda environment

├── batchrun.py # Batch experiment runner

├── main.py # Main execution file

└── README.md

```

---

# Tested Environment

The reproduction was successfully validated using the following environment.

| Component | Version |
|------------|----------|
| Operating System | Ubuntu 22.04 LTS |
| Python | 3.10.16 |
| Conda | Miniconda |
| PyTorch | 2.6.0 |
| CUDA | Supported (optional) |

The framework can also be executed on CPU-only systems, although GPU acceleration is recommended for larger experiments.

---
# Quick Start Guide

This guide provides a complete walkthrough for cloning, installing, and running the FLPoison framework on a clean Ubuntu system.

---

## Step 1 – Clone the Repository

Open a terminal and clone the repository:

```bash
git clone https://github.com/lunaabboud/FLPoison-project.git
```

Move into the project directory:

```bash
cd FLPoison-project
```

---

## Step 2 – Create the Conda Environment

The repository provides an `environment.yaml` file containing all required dependencies.

Create the environment by running:

```bash
conda env create -f environment.yaml
```

This command installs all required Python packages and creates a Conda environment named:

```text
torchenv
```

The installation may take several minutes depending on your internet connection.

---

## Step 3 – Activate the Environment

Activate the Conda environment before running any experiments.

```bash
conda activate torchenv
```

---

## Step 4 – Verify the Installation

Verify that Python is correctly installed.

```bash
python --version
```

Expected output:

```text
Python 3.10.16
```

---

Verify that PyTorch is available.

```bash
python -c "import torch; print(torch.__version__)"
```

Example output:

```text
2.6.0
```

---

(Optional) Verify GPU support.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Possible outputs:

```text
True
```

or

```text
False
```

If the output is `False`, the framework will automatically execute using the CPU.

---

## Step 5 – Run a Quick Validation

To quickly verify that the framework is functioning correctly, execute:

```bash
python main.py -config configs/FedSGD_MNIST_test.yaml
```

The `FedSGD_MNIST_test.yaml` configuration was created during the reproduction process to reduce the number of training epochs, allowing the framework to be validated much faster than the standard experiment.

This is the recommended configuration for first-time execution.

---

## Step 6 – Run the Standard Experiment

To execute the original MNIST experiment, run:

```bash
python main.py -config configs/FedSGD_MNIST_config.yaml
```

Alternatively, the number of training epochs can be overridden directly from the command line without modifying the configuration file.

Example:

```bash
python main.py -config configs/FedSGD_MNIST_config.yaml -e 3
```

---

## Step 7 – Verify Successful Execution

The framework has been installed successfully if you observe the following:

- The configuration file loads successfully.
- The dataset is downloaded automatically (if not already available).
- The Federated Learning server and clients are initialized.
- Training rounds begin without errors.
- Training and evaluation metrics are displayed in the terminal.
- The experiment completes successfully.
- Output files are generated inside the `logs/` directory.

Typical terminal output includes messages similar to:

```text
Loading configuration...
Loading dataset...
Initializing clients...
Starting Federated Learning...
Training...
Evaluating...
Experiment completed successfully.
```

---

## Step 8 – Generated Outputs

After successful execution, experiment outputs are stored inside the `logs/` directory.

Depending on the selected experiment, generated files may include:

- Training logs
- Evaluation metrics
- Accuracy and loss values
- Figures and plots
- Experiment summaries

---

## Running Other Experiments

To execute experiments using different datasets or algorithms, simply specify another configuration file.

Examples:

FedSGD on CIFAR-10

```bash
python main.py -config configs/FedSGD_CIFAR10_config.yaml
```

FedSGD on CIFAR-100

```bash
python main.py -config configs/FedSGD_CIFAR100_config.yaml
```

FedOpt on MNIST

```bash
python main.py -config configs/FedOpt_MNIST_config.yaml
```

Additional experiment configurations are available in the `configs/` directory.

---

## Troubleshooting

### Conda environment already exists

Remove the existing environment:

```bash
conda env remove -n torchenv
```

Then recreate it:

```bash
conda env create -f environment.yaml
```

---

### CUDA is not detected

Run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The project can still be executed using the CPU if CUDA is unavailable.

---

### Package installation failed

If the Conda environment was not created successfully, remove it and reinstall:

```bash
conda env remove -n torchenv
conda env create -f environment.yaml
```

---

## Installation Checklist

Before running experiments, verify that:

- Repository cloned successfully.
- Conda environment created successfully.
- Environment activated.
- Python version is 3.10.16.
- PyTorch imports successfully.
- The quick validation experiment executes without errors.
- Output files are generated in the `logs/` directory.

If all of the above conditions are satisfied, the FLPoison framework has been successfully reproduced and is ready for further experimentation.

# Running the Project

After completing the installation and activating the Conda environment, the framework is ready to execute Federated Learning experiments.

The project supports multiple datasets and algorithms through YAML configuration files located in the `configs/` directory.

---

# Available Configuration Files

Some of the available experiment configurations include:

| Configuration File | Description |
|--------------------|-------------|
| `FedSGD_MNIST_config.yaml` | Standard FedSGD experiment on MNIST |
| `FedSGD_MNIST_test.yaml` | Lightweight MNIST configuration for quick validation |
| `FedSGD_CIFAR10_config.yaml` | FedSGD on CIFAR-10 |
| `FedSGD_CIFAR100_config.yaml` | FedSGD on CIFAR-100 |
| `FedSGD_CINIC10_config.yaml` | FedSGD on CINIC-10 |
| `FedSGD_CHMNIST_config.yaml` | FedSGD on CHMNIST |
| `FedSGD_TinyImageNet_config.yaml` | FedSGD on TinyImageNet |
| `FedOpt_MNIST_config.yaml` | FedOpt on MNIST |
| `FedOpt_CIFAR10_config.yaml` | FedOpt on CIFAR-10 |

Additional configuration files can be found inside the `configs/` directory.

---

# Quick Validation (Recommended)

To quickly verify that the project has been installed correctly, execute:

```bash
python main.py -config configs/FedSGD_MNIST_test.yaml
```

The `FedSGD_MNIST_test.yaml` configuration was created specifically for the reproduction process.

It is based on the standard MNIST configuration but uses a reduced number of training epochs, allowing the framework to be validated much faster while preserving the same execution workflow.

This configuration is recommended for installation verification, debugging, and quick functionality testing.

---

# Running the Standard MNIST Experiment

To execute the original MNIST experiment, run:

```bash
python main.py -config configs/FedSGD_MNIST_config.yaml
```

This command executes the experiment using the parameters defined in the original configuration file.

---

# Overriding the Number of Epochs

Instead of modifying the YAML configuration file, the number of training epochs can be overridden directly from the command line.

For example, to execute the standard MNIST experiment using only three epochs:

```bash
python main.py -config configs/FedSGD_MNIST_config.yaml -e 3
```

The `-e` argument overrides the epoch value specified inside the configuration file.

This is particularly useful for:

- Quick validation
- Debugging
- Testing new modifications
- Reducing execution time

without permanently modifying the original experiment configuration.

---

# Running Other Experiments

To execute experiments using different datasets, simply specify another configuration file.

Examples:

CIFAR-10

```bash
python main.py -config configs/FedSGD_CIFAR10_config.yaml
```

CIFAR-100

```bash
python main.py -config configs/FedSGD_CIFAR100_config.yaml
```

FedOpt on MNIST

```bash
python main.py -config configs/FedOpt_MNIST_config.yaml
```

The same procedure applies to all configuration files included in the repository.

---

# Expected Output

When the experiment starts successfully, the terminal will display information similar to:

- Loading configuration...
- Loading dataset...
- Initializing clients...
- Building the neural network...
- Starting federated learning rounds...
- Training progress
- Evaluation metrics
- Experiment completed successfully

Depending on the selected configuration, execution time may vary from a few minutes to significantly longer for larger datasets.

---

# Generated Output

After the experiment completes successfully, the framework automatically generates experiment outputs.

Typical outputs include:

- Training logs
- Evaluation metrics
- Accuracy and loss values
- Generated figures (when applicable)
- Experiment summaries

These outputs are stored inside the `logs/` directory.

---

# Verification Checklist

The reproduction can be considered successful if:

- The project starts without import errors.
- The selected configuration loads successfully.
- The training process begins normally.
- Federated learning rounds execute correctly.
- The experiment completes without runtime errors.
- Output files are generated inside the `logs/` directory.

Successful completion of these steps confirms that the FLPoison framework has been reproduced correctly.
