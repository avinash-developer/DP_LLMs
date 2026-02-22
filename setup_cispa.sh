#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ">>> Setting up CISPA environment..."

if [ ! -d "venv" ]; then
    echo ">>> Creating virtual environment (venv)..."
    python3 -m venv venv
fi

source venv/bin/activate

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing dependencies from requirements.txt..."
pip install -r requirements.txt

mkdir -p logs results

echo ">>> Verifying core imports..."
python - <<'PY'
import transformers
import datasets
import torch
print("✅ Environment ready")
print(f"transformers={transformers.__version__}")
print(f"torch={torch.__version__}")
PY

echo ">>> Setup complete."
