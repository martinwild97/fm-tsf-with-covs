#!/bin/bash

# Define sleep time (in hours) after successful evaluation to allow downloading evaluation results before Kubernetes job is finished.
SLEEP_TIME_HOURS=96
export PYTHONPATH="$PYTHONPATH:/workspaces/progai-forecast-evaluation/src"
echo "Starting ProgAI forecast evaluation job"

python3 src/main.py

echo "Going to sleep (for $SLEEP_TIME_HOURS hours) before shutdown."
sleep $((SLEEP_TIME_HOURS * 3600))
echo "Finished ProgAI forecast evaluation job"
