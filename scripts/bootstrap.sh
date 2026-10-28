#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/auronox/.local/share/ai-rocm/shared-env.sh
python="$AI_ROCM_VENV/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Shared ROCm environment is missing: $AI_ROCM_VENV" >&2
    exit 1
fi
if [[ -e .venv || -L .venv ]]; then
    if [[ ! -L .venv || "$(readlink -f .venv)" != "$AI_ROCM_VENV" ]]; then
        echo 'Migrate the existing .venv before bootstrap; no second PyTorch environment will be created.' >&2
        exit 1
    fi
else
    ln -s "$AI_ROCM_VENV" .venv
fi
"$python" - <<'PY'
import torch
assert torch.version.hip and torch.version.cuda is None, 'Shared ROCm PyTorch required'
PY
if [[ ! -d .deps/metadrive/.git ]]; then
    git clone --single-branch https://github.com/metadriverse/metadrive.git .deps/metadrive
fi
required=fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1
if [[ "$(git -C .deps/metadrive rev-parse HEAD)" != "$required" ]]; then
    git -C .deps/metadrive checkout "$required"
fi
"$python" -m pip install -c "$PIP_CONSTRAINT" -r requirements-runtime.txt -r requirements-training.txt -r requirements-visualization.txt
"$python" -m pip install -c "$PIP_CONSTRAINT" -e .deps/metadrive
# Pull assets only if absent; training data/models/reports are never touched.
if [[ ! -d .deps/metadrive/metadrive/assets ]]; then
    "$python" -m metadrive.pull_asset
fi
"$python" -m pip check
echo "Ready. Shared interpreter: $python"
