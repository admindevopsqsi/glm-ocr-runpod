import pytest

pytest.importorskip("flask")
pytest.importorskip("glmocr")

import service


def test_resolve_runtime_profile_for_16gb_gpu(monkeypatch):
    monkeypatch.delenv("GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)
    monkeypatch.delenv("MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("ENABLE_MTP", raising=False)
    monkeypatch.setattr(service, "detect_gpu_info", lambda: ("RTX A4000", 16.0))

    profile = service.resolve_runtime_profile()

    assert profile["gpu_name"] == "RTX A4000"
    assert profile["gpu_memory_utilization"] == "0.88"
    assert profile["max_model_len"] == "4096"
    assert profile["max_num_seqs"] == "1"
    assert profile["enable_mtp"] is False


def test_resolve_runtime_profile_for_24gb_gpu(monkeypatch):
    monkeypatch.delenv("GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)
    monkeypatch.delenv("MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("ENABLE_MTP", raising=False)
    monkeypatch.setattr(service, "detect_gpu_info", lambda: ("RTX 4090", 24.0))

    profile = service.resolve_runtime_profile()

    assert profile["gpu_name"] == "RTX 4090"
    assert profile["gpu_memory_utilization"] == "0.9"
    assert profile["max_model_len"] == "8192"
    assert profile["max_num_seqs"] == "2"
    assert profile["enable_mtp"] is True


def test_ping_returns_200_while_initializing(monkeypatch):
    service.state.ready = False
    service.state.stage = "starting_vllm"

    with service.app.test_client() as client:
        response = client.get("/ping")

    assert response.status_code == 200
