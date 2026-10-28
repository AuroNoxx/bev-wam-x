"""SDPA/operator contract regression tests (ROCm backend consistency).

The stage-3/4 main paths must enter mini_wa_jepa.operators.sdpa_context so the
recorded backend is the backend that actually runs. These tests pin that contract.
"""
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from mini_wa_jepa.operators import (
    configure_operator_optimizations,
    resolve_sdpa_backends,
    sdpa_context,
    verify_attention_fidelity,
)


def test_math_backend_is_pure_math():
    assert resolve_sdpa_backends("MATH", "AMD") == [SDPBackend.MATH]


def test_auto_amd_orders_vendor_backends_first_and_math_last():
    backends = resolve_sdpa_backends("AUTO", "AMD")
    assert backends[0] == SDPBackend.FLASH_ATTENTION
    assert backends[-1] == SDPBackend.MATH
    assert set(backends) == {
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    }


def test_unknown_backend_falls_back_to_math():
    assert resolve_sdpa_backends("NOT_A_BACKEND", "AMD") == [SDPBackend.MATH]


def test_stage3_config_backend_is_a_recorded_contract_value():
    from configs.stage3 import Stage3Config

    assert Stage3Config().sdpa_backend.upper() in {"AUTO", "MATH", "FLASH", "EFFICIENT", "CUDNN"}


def test_configure_operator_optimizations_applies_and_is_recorded():
    before = {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    try:
        applied = configure_operator_optimizations(
            vendor="AMD", allow_tf32=False, cudnn_benchmark=False, deterministic=True
        )
        assert applied["cudnn_benchmark"] is False
        assert applied["cudnn_deterministic"] is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = before["matmul_allow_tf32"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SDPA runtime check requires GPU")
@pytest.mark.parametrize("backend", ["MATH", "AUTO"])
def test_sdpa_context_runs_and_is_finite(backend):
    q = torch.randn(8, 4, 32, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with sdpa_context(backend, vendor="AMD"):
        out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SDPA fidelity check requires GPU")
def test_auto_backend_fidelity_is_recorded_against_math():
    result = verify_attention_fidelity(
        torch.randn(8, 4, 32, 64, device="cuda", dtype=torch.bfloat16),
        torch.randn(8, 4, 32, 64, device="cuda", dtype=torch.bfloat16),
        torch.randn(8, 4, 32, 64, device="cuda", dtype=torch.bfloat16),
        candidate_backend="AUTO",
        vendor="AMD",
    )
    assert result["candidate_backend"] == "AUTO"
    assert result["max_abs_error"] >= 0.0
    assert result["relative_l2_error"] >= 0.0
    assert result["max_abs_error"] < 0.5  # bf16 AOTriton vs math sanity bound
