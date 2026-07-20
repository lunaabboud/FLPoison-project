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
# Installation

This section describes the steps required to install and execute the FLPoison framework on a clean Ubuntu system.

---

## Prerequisites

Before installing the project, ensure the following software is available on your machine:

- Ubuntu 22.04 LTS (recommended)
- Git
- Miniconda or Anaconda
- Python 3.10
- Internet connection (required for dependency and dataset downloads)

---

## Clone the Repository

Clone the reproduced repository from GitHub:

```bash
git clone https://github.com/lunaabboud/FLPoison-project.git
```

Navigate to the project directory:

```bash
cd FLPoison-project
```

---

## Create the Conda Environment

All required dependencies are provided in the `environment.yaml` file.

Create the environment using:

```bash
conda env create -f environment.yaml
```

Depending on your internet connection, the installation may take several minutes.

---

## Activate the Environment

After the installation completes successfully, activate the environment:

```bash
conda activate torchenv
```

---

## Verify the Installation

Verify that Python has been installed correctly:

```bash
python --version
```

Expected output:

```text
Python 3.10.16
```

---

Verify the installed PyTorch version:

```bash
python -c "import torch; print(torch.__version__)"
```

Expected output:

```text
2.6.0
```

---

## Verify CUDA (Optional)

If an NVIDIA GPU is available, verify CUDA support:

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

No additional configuration is required.

---

## Verify the Repository

The project directory should contain files similar to the following:

```text
aggregators/
attackers/
configs/
data/
datapreprocessor/
docs/
fl/
logs/
environment.yaml
main.py
batchrun.py
README.md
```

---

## First-Time Execution

During the first execution, the framework may automatically create additional directories and download the required datasets if they are not already available.

This process only occurs once.

Subsequent executions will reuse the downloaded datasets.

---

## Installation Checklist

Before running any experiments, verify that:

- Repository cloned successfully.
- Conda environment created successfully.
- Environment activated (`torchenv`).
- Python version is 3.10.16.
- PyTorch imports without errors.
- CUDA is detected if a compatible GPU is available.

If all the above steps complete successfully, the project is ready to execute experiments.

---
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
