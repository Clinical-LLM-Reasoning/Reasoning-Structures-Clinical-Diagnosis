#!/bin/bash

LOG_FILE="progress_log.txt"
SCRIPT="main.py"

METHODS=("bfs" "dtree" "cot" "bot" "pure_llm")
TEXT_OPTIONS=("" "--use_text")

# Optional: Clear the last log.
# echo "" > $LOG_FILE

for method in "${METHODS[@]}"; do
    for txt_opt in "${TEXT_OPTIONS[@]}"; do
        
        # 1. Determine the current status of the text parameter (for writing logs).
        if [ -z "$txt_opt" ]; then
            STATUS="False"
        else
            STATUS="True"
        fi

        # 2. Get timestamp
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        MSG="[$TIMESTAMP] Start Task: Method = $method | Use Text = $STATUS"

        # 3. Print to screen and write to file
        echo "------------------------------------------------"
        echo "$MSG"
        echo "------------------------------------------------"
        echo "$MSG" >> $LOG_FILE

    # 4. === Using uv run ===

    # Note: You don't need to explicitly write python here; uv run will handle it automatically.

    # If your script doesn't have execute permissions, you can also write: uv run python $SCRIPT ...
        uv run $SCRIPT --method "$method" $txt_opt

    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks finished." >> $LOG_FILE
