#!/usr/bin/env bash
# ==============================================================================
# TensorBoard 一键启动脚本 (Click & Launch Script)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_TENSORBOARD="$SCRIPT_DIR/.venv/bin/tensorboard"
if [ ! -f "$VENV_TENSORBOARD" ]; then
    echo "错误: 未找到 TensorBoard 可执行文件: $VENV_TENSORBOARD" >&2
    which notify-send >/dev/null 2>&1 && notify-send -u critical "TensorBoard" "未找到虚拟环境中的 TensorBoard"
    exit 1
fi

PORT=6006
URL="http://127.0.0.1:$PORT"

# 检查服务是否已在运行
if ss -tuln | grep -q ":$PORT "; then
    which notify-send >/dev/null 2>&1 && notify-send "TensorBoard" "服务已在后台运行，正在打开网页看板..."
    which xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
    exit 0
fi

which notify-send >/dev/null 2>&1 && notify-send "TensorBoard" "正在启动训练监控看板并打开浏览器..."

# 后台计时器唤醒系统浏览器
(
    sleep 2
    which xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
) &

exec "$VENV_TENSORBOARD" --logdir "$SCRIPT_DIR/artifacts" --port "$PORT" --bind_all
