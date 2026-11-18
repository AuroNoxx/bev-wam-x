"""Project-local runtime and precision contexts; no global environment changes."""
from contextlib import contextmanager
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure():
    for name, folder in {
        "HF_HOME": "huggingface", "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton", "TMP": "tmp", "TEMP": "tmp",
    }.items():
        path = ROOT / ".cache/v5-qwen-drive-world" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.path[:0] = [str(ROOT / "runtime/v5-qwen-drive-deps"), str(ROOT / "runtime/qwen-drive-deps"), str(ROOT / "runtime/qwen-drive-official/src")]
    site = ROOT / '.venv-train/Lib/site-packages'
    hip = site / '_rocm_sdk_core'
    compiler = site / 'triton/runtime/tcc/tcc.exe'
    if hip.is_dir() and compiler.is_file():
        # Existing bundled compiler and SDK, selected only for this process.
        # clang-cl needs a separate MSVC CRT; the shipped TinyCC includes its CRT.
        os.environ['HIP_PATH'] = str(hip)
        os.environ['CC'] = str(compiler)
    os.environ["PATH"] = os.pathsep.join((str(ROOT / '.venv-train/Scripts'),
                                         str(ROOT / "runtime/v5-qwen-drive-deps/bin"), os.environ.get("PATH", "")))


@contextmanager
def math_attention():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        yield
