@echo off
cd /d %~dp0
call conda activate edge_ai
python scripts/run_all_pipeline.py --image images/cat.jpg
pause
