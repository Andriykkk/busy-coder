#!/bin/bash

# This script downloads and prepares the data required for the evaluation benchmarks.

# Create the data directory
mkdir -p ../data/eval

# --- MMLU ---
echo "Setting up MMLU dataset..."
mkdir -p ../data/eval/mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P ../data/eval/mmlu
tar -xvf ../data/eval/mmlu/data.tar -C ../data/eval/mmlu
rm ../data/eval/mmlu/data.tar
mv ../data/eval/mmlu/data/* ../data/eval/mmlu
rm -r ../data/eval/mmlu/data
echo "MMLU setup complete."

# --- MATH ---
echo "Setting up MATH dataset..."
mkdir -p ../data/eval/math
wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar -P ../data/eval/math
tar -xvf ../data/eval/math/MATH.tar -C ../data/eval/math
rm ../data/eval/math/MATH.tar
mv ../data/eval/math/MATH/* ../data/eval/math
rm -r ../data/eval/math/MATH
echo "MATH setup complete."

# --- GSM8K ---
echo "Setting up GSM8K dataset..."
echo "This requires the 'datasets' library. Please install it using 'pip install datasets'."
mkdir -p ../data/eval/gsm8k
python3 -c "import datasets; dataset = datasets.load_dataset('gsm8k', 'main'); dataset.save_to_disk('../data/eval/gsm8k')"
echo "GSM8K setup complete."

echo "All datasets have been set up in the 'developer2/evaluation/data' directory."
