#!/bin/bash

#SBATCH --account=def-morshed
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=8000M
#SBATCH --time=20:00:00
#SBATCH --output=/home/morshed/FLProjects/FLPoison-project/logs/job_output.log
#SBATCH --error=/home/morshed/FLProjects/FLPoison-project/logs/job_error.log

# Exit script immediately if any command fails
set -e

# Load necessary modules
module load python/3.10.13

# Change to the project directory
cd /home/morshed/FLProjects/FLPoison-project

# Get attack and defense from command-line parameters
ATTACK="$1"
DEFENSE="$2"

# Check that both parameters are provided
if [ -z "$ATTACK" ] || [ -z "$DEFENSE" ]; then
    echo "Usage: sbatch run_tumor.sh <attack> <defense>"
    echo "Example: sbatch run_tumor.sh Neurotoxin Mean"
    exit 1
fi

echo "Running attack: $ATTACK"
echo "Running defense: $DEFENSE"

# Run the federated learning experiment
python main.py \
    -config configs/FedSGD_TUMOR4_config.yaml \
    --attack "$ATTACK" \
    --defense "$DEFENSE"

echo "End of job"