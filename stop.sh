#!/usr/bin/env bash
# bev-wam-x 停止服务脚本
pkill -f "marimo run app.py" || true
pkill -f "marimo edit app.py" || true
which notify-send >/dev/null 2>&1 && notify-send "bev-wam-x" "服务已停止" || true
echo "[✓] bev-wam-x 服务已成功停止。"
