#!/usr/bin/env bash
set -e          # stop if a command fails
set -x          # echo commands

# Launch FastAPI on port 8000 (background)
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Launch Streamlit on port 8501 (foreground)
streamlit run ./script/test_UI.py \
         --server.port 8501 \
         --server.address 0.0.0.0
