#!/bin/bash

# run_experiments.sh
# Iterates over all combinations and saves to CSV

METHODS=("soft-prompt" "prefix" "lora" "full-finetuning" "last-layer-finetuning" "soft-prompt+lora" "prefix+lora")
DATASETS=("sst2" "qnli" "qqp" "mnli")
PRIVACY_SETTINGS=("8" "-1") # 8 for DP, -1 for Non-Private
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"

OUTPUT_FILE="final_results.csv"

# Write Header
echo "Privacy(Epsilon),Dataset,Method,Score,NumParams" > $OUTPUT_FILE

echo ">>> Starting Experiments on Blackwell GPU..."
echo ">>> Results will be saved to $OUTPUT_FILE"

for epsilon in "${PRIVACY_SETTINGS[@]}"; do
    eps_label="Infinity"
    if [ "$epsilon" == "8" ]; then
        eps_label="8"
    fi

    for dataset in "${DATASETS[@]}"; do
        for method in "${METHODS[@]}"; do
            echo "----------------------------------------------------------------"
            echo "Running: Dataset=$dataset | Method=$method | Epsilon=$eps_label"
            
            # Run Python Script
            # We capture the exit code to ensure robustness
            "$PYTHON_BIN" "$SCRIPT_DIR/main.py" --dataset "$dataset" --method "$method" --epsilon "$epsilon"
            
            if [ $? -eq 0 ]; then
                # Read result from temp file created by python script
                content=$(cat temp_result.txt)
                score=$(echo "$content" | cut -d',' -f1)
                params=$(echo "$content" | cut -d',' -f2)
                echo "$eps_label,$dataset,$method,$score,$params" >> $OUTPUT_FILE
                echo ">>> Success. Score: $score | Params: $params"
            else
                echo ">>> Failure."
                echo "$eps_label,$dataset,$method,ERROR,ERROR" >> $OUTPUT_FILE
            fi
            
            echo "----------------------------------------------------------------"
        done
    done
done

echo ">>> All experiments completed."
cat $OUTPUT_FILE