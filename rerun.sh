#!/usr/bin/env bash
# Rerun 一键启动：优先打开最新的 BEV WA-JEPA .rrd 时间线
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RERUN="$SCRIPT_DIR/.venv/bin/rerun"
if [ ! -x "$RERUN" ]; then
    which notify-send >/dev/null 2>&1 && notify-send -u critical "Rerun" "未找到 .venv/bin/rerun"
    echo "错误: 未找到 $RERUN" >&2
    exit 1
fi

LATEST="$(find "$SCRIPT_DIR/artifacts" -name '*.rrd' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
if [ -n "${LATEST:-}" ]; then
    which notify-send >/dev/null 2>&1 && notify-send "Rerun" "打开最新时间线: ${LATEST#$SCRIPT_DIR/}"
    exec "$RERUN" "$LATEST"
else
    which notify-send >/dev/null 2>&1 && notify-send "Rerun" "没有 .rrd；打开空白 Rerun 查看器"
    exec "$RERUN"
fi
