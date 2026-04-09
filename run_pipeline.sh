#!/bin/bash

echo "Updating NREL data"
python3.11 -u update_data.py

echo "Running gradient boosting forecast"
python3.11 time_series_model.py

echo "Pipeline complete"

