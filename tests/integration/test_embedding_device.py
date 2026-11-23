"""The encoder exactly as the server builds it, checked against a CPU fp32 reference.

Background: on Apple Silicon, torch's MPS scaled-dot-product attention returns
wrong BGE-M3 vectors for single inputs of about 8 tokens or fewer (torch 2.8.0:
cosine to the CPU vector 0.25-0.57), i.e. for every tag and most short queries.
src/core/encoder.load_bge_m3 loads the model with eager attention on MPS (or
falls back to CPU). These tests use the real weights on the real device.
"""
import numpy as np
import pytest
import torch

ON_MPS = torch.backends.mps.is_available()

SHORT = ["lisbon", "music", "project-kestrel", "cat at the vet", "наши коты"]  # <= 8 tokens each
LONG = ("Took the cat to the vet clinic for her annual vaccination; she weighed 4.8 kg, got a "
        "booster shot and sulked in the carrier all the way home.")


@pytest.fixture(scope="module")
def server_storage(encoder):
    import src.storage.postgres_enhanced as pe

    storage = pe.EnhancedPostgresStorage()
    storage.encoder = encoder  # the session encoder IS the server-built one (load_bge_m3)
    return storage


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def reference(model, text):
    return model.encode([text], return_dense=True, return_sparse=False,
                        return_colbert_vecs=False)["dense_vecs"][0]


def test_long_text_matches_the_cpu_reference(server_storage, cpu_reference, run):
    assert cosine(run(server_storage.encode_text(LONG)), reference(cpu_reference, LONG)) > 0.999


@pytest.mark.skipif(not ON_MPS, reason="MPS not available: the short-text attention bug is MPS-only")
def test_on_mps_the_server_encoder_uses_eager_attention(encoder):
    assert encoder.target_devices == ["mps:0"], "load_bge_m3 fell back to CPU on an MPS machine"
    assert encoder.model.model.config._attn_implementation == "eager"


@pytest.mark.skipif(not ON_MPS, reason="MPS not available: the short-text attention bug is MPS-only")
@pytest.mark.parametrize("text", SHORT)
def test_on_mps_short_texts_match_the_cpu_reference(server_storage, cpu_reference, run, text):
    assert cosine(run(server_storage.encode_text(text)), reference(cpu_reference, text)) > 0.999
