#!/bin/bash

# run_method.sh
# Usage: ./run_method.sh <method_name>
# This runs all datasets and privacy settings for a specific method.

METHOD=$1
DATASETS=("sst2" "qnli" "qqp" "mnli")
EPSILONS=("8" "-1") # 8 is Private, -1 is Non-Private

if [ -z "$METHOD" ]; then
    echo "Error: Please specify a method."
    echo "Available methods:"
    echo "  soft-prompt"
    echo "  prefix"
    echo "  lora"
    echo "  full-finetuning"
    echo "  last-layer-finetuning"
    echo "  soft-prompt+lora"
    echo "  prefix+lora"
    echo "  ia3   <-- Use this for (IA)^3"
    echo ""
    echo "Example: ./run_method.sh ia3"
    exit 1
fi

echo "========================================================"
echo "Batch Running Method: $METHOD"
echo "========================================================"

for eps in "${EPSILONS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo ">>> Launching: Dataset=$dataset | Epsilon=$eps"
        ./run_param.sh $METHOD $dataset $eps
        echo "--------------------------------------------------------"
    done
done

echo ">>> Batch for $METHOD completed."