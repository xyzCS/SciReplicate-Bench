#!/bin/bash

# SciReproducer Runner Script
# Usage: bash run_sci_reproducer.sh <root_path> [model]
# Example: bash run_sci_reproducer.sh /path/to/SciReplicate-Bench gpt-4o-mini

set -e  # Exit on any error

# Check if root_path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <root_path> [model]"
    echo "Example: $0 /path/to/SciReplicate-Bench gpt-4o-mini"
    echo ""
    echo "Parameters:"
    echo "  root_path: Path to the root directory containing all project files"
    echo "  model:     Language model to use (default: gpt-4o-mini)"
    echo "             Options: gpt-4o-mini, gpt-4o, claude-3, etc."
    exit 1
fi

# Get parameters
ROOT_PATH="$1"
MODEL="${2:-gpt-4o-mini}"  # Default to gpt-4o-mini if not specified

# Validate root_path exists
if [ ! -d "$ROOT_PATH" ]; then
    echo "Error: Root path '$ROOT_PATH' does not exist"
    exit 1
fi

# Construct other paths based on root_path
BENCHMARK_PATH="${ROOT_PATH}/BENCHMARK"
OUTPUT_DIR="${ROOT_PATH}/Result"
CONDA_ENV_PATH="${ROOT_PATH}/envs_sci"

# Validate required directories exist
if [ ! -d "$BENCHMARK_PATH" ]; then
    echo "Error: Benchmark directory '$BENCHMARK_PATH' does not exist"
    echo "Please ensure the BENCHMARK directory is present in the root path"
    exit 1
fi

if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Error: Conda environments directory '$CONDA_ENV_PATH' does not exist"
    echo "Please run the environment setup script first: bash ./env.sh"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Check if SciReproducer.py exists
SCRIPT_PATH="${ROOT_PATH}/SciReproducer.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: SciReproducer.py not found at '$SCRIPT_PATH'"
    exit 1
fi

echo "=========================================="
echo "SciReproducer Configuration"
echo "=========================================="
echo "Root Path:        $ROOT_PATH"
echo "Benchmark Path:   $BENCHMARK_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Conda Env Path:   $CONDA_ENV_PATH"
echo "Model:            $MODEL"
echo "=========================================="

# Ask for confirmation
read -p "Do you want to proceed with these settings? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo "Starting SciReproducer..."
echo "=========================================="

# Change to root directory and run SciReproducer
cd "$ROOT_PATH"

python SciReproducer.py \
    --model "$MODEL" \
    --root_path "$ROOT_PATH" \
    --Benchmark_Path "$BENCHMARK_PATH" \
    --OutputDir "$OUTPUT_DIR" \
    --conda_env_path "$CONDA_ENV_PATH"

echo "=========================================="
echo "SciReproducer execution completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="
