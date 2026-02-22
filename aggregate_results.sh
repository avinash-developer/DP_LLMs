#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# aggregate_results.sh
OUTPUT_FILE="final_aggregated_results.csv"

# Updated Header to include Params
echo "Privacy(Epsilon),Dataset,Method,Score,NumParams" > $OUTPUT_FILE

if ls results/*.csv 1> /dev/null 2>&1; then
    cat results/*.csv | sort >> $OUTPUT_FILE
    echo ">>> Successfully aggregated results to $OUTPUT_FILE"
    cat $OUTPUT_FILE
else
    echo ">>> No result files found in results/."
fi