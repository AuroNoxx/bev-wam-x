@echo off
:: ==============================================================================
:: TensorBoard Windows 一键启动脚本 (Double-click to Launch)
:: ==============================================================================
cd /d "%~dp0"
echo 正在启动 TensorBoard 看板...
start "" "http://127.0.0.1:6006"
if exist .venv\Scripts\tensorboard.exe (
    .venv\Scripts\tensorboard.exe --logdir artifacts --port 6006
) else (
    tensorboard --logdir artifacts --port 6006
)
pause
