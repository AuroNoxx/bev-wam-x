#!/usr/bin/env bash
# Marimo 一键启动：打开 bev-wam-x 实验控制台/看板
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MARIMO="$SCRIPT_DIR/.venv/bin/marimo"
if [ ! -x "$MARIMO" ]; then
    which notify-send >/dev/null 2>&1 && notify-send -u critical "marimo" "未找到 .venv/bin/marimo"
    echo "错误: 未找到 $MARIMO" >&2
    exit 1
fi

HOST=127.0.0.1
PORT=2718
URL="http://$HOST:$PORT"
NOTEBOOK="$SCRIPT_DIR/scripts/dashboard_marimo.py"

if ss -tuln | grep -q ":$PORT "; then
    which notify-send >/dev/null 2>&1 && notify-send "marimo" "服务已在运行，打开浏览器..."
    which xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
    exit 0
fi

which notify-send >/dev/null 2>&1 && notify-send "marimo" "正在启动实验控制台..."
( sleep 3; which xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true ) &
exec "$MARIMO" edit "$NOTEBOOK" --host "$HOST" --port "$PORT" --no-token
