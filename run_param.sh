#!/bin/bash

# run_param.sh
# Usage: ./run_param.sh <method> <dataset> <epsilon>

METHOD=$1
DATASET=$2
EPSILON=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"

if [ -z "$METHOD" ] || [ -z "$DATASET" ] || [ -z "$EPSILON" ]; then
    echo "Error: Missing arguments."
    echo "Usage: ./run_param.sh <method> <dataset> <epsilon>"
    exit 1
fi

RUN_ID="${METHOD}_${DATASET}_EPS${EPSILON}"
LOG_FILE="logs/${RUN_ID}.log"
RESULT_FILE="results/${RUN_ID}.csv"

echo ">>> Starting Run: $RUN_ID"


# Run Python Script using venv python
"$PYTHON_BIN" "$SCRIPT_DIR/main.py" --dataset "$DATASET" --method "$METHOD" --epsilon "$EPSILON" > >(tee "$LOG_FILE") 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    if [ -f "temp_result.txt" ]; then
        # Read comma-separated score and params
        CONTENT=$(cat temp_result.txt)
        SCORE=$(echo $CONTENT | cut -d',' -f1)
        PARAMS=$(echo $CONTENT | cut -d',' -f2)
        
        # Format: Epsilon,Dataset,Method,Score,NumParams
        echo "${EPSILON},${DATASET},${METHOD},${SCORE},${PARAMS}" > "$RESULT_FILE"
        echo ">>> Finished $RUN_ID. Score: $SCORE | Params: $PARAMS"
    else
        echo ">>> Error: temp_result.txt not found for $RUN_ID"
    fi
else
    echo ">>> Failed $RUN_ID. Check $LOG_FILE"
fi