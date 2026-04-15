#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

set -a
source .env
set +a

export PYTHONUNBUFFERED=1

rm -rf outputs/01_machiavelli
mkdir -p outputs/01_machiavelli

# Run each model in parallel (standard + good + maxscore policies)
models=(claude-haiku-4-5 claude-sonnet-4-6 claude-opus-4-6)
policies=(standard good maxscore)
pids=()

for model in "${models[@]}"; do
    for policy in "${policies[@]}"; do
        python3 experiments/01_machiavelli/run.py \
            --model "$model" --policy "$policy" --skip-plot \
            --episodes 1 --concurrency 100 --verbose &
        pids+=($!)
    done
done

# Wait for all and propagate failures
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done

if [ "$failed" -ne 0 ]; then
    echo "One or more runs failed." >&2
    exit 1
fi

# Generate plots now that all results are in
python3 experiments/01_machiavelli/run.py --plot-only

echo "All evals complete."
