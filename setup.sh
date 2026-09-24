#!/usr/bin/env bash
# One-time setup on macOS / Linux
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
echo "Setup complete. Start the dashboard with:  source .venv/bin/activate && streamlit run dashboard/app.py"
