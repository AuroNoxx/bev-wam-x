#!/usr/bin/env bash
# ==============================================================================
# bev-wam-x 一键启动脚本 (Click & Launch Script)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/auronox/.local/share/ai-rocm/shared-env.sh
VENV_MARIMO="$AI_ROCM_VENV/bin/marimo"
if [ ! -f "$VENV_MARIMO" ]; then
    echo "错误: 未找到虚拟环境 $VENV_MARIMO"
    which notify-send >/dev/null 2>&1 && notify-send -u critical "bev-wam-x" "未找到虚拟环境: $VENV_MARIMO"
    exit 1
fi

PORT=2718
URL="http://127.0.0.1:$PORT"

MODE="run"
if [ "$1" == "--edit" ] || [ "$1" == "-e" ]; then
    MODE="edit"
fi

# 检查服务是否已在运行
if ss -tuln | grep -q ":$PORT "; then
    which notify-send >/dev/null 2>&1 && notify-send "bev-wam-x" "服务已在运行，正在打开网页..."
    which xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
    exit 0
fi

which notify-send >/dev/null 2>&1 && notify-send "bev-wam-x" "正在启动工作台并打开浏览器..."

# 后台计时器兜底唤醒系统浏览器
(
    sleep 2
    which xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
) &

if [ "$MODE" == "edit" ]; then
    exec "$VENV_MARIMO" edit app.py --port "$PORT" --host 127.0.0.1
else
    exec "$VENV_MARIMO" run app.py --port "$PORT" --host 127.0.0.1 --watch
fi
