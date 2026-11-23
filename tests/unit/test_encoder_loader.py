"""src/core/encoder.py: load_bge_m3() without loading any weights.

FlagEmbedding and AutoModel are replaced by fakes, so these tests only check the
loader's own logic: on MPS it must make FlagEmbedding's M3 runner pass
attn_implementation='eager', restore the runner afterwards, and fall back to CPU
when it cannot confirm eager attention. The real model on a real MPS device is
checked in tests/integration/test_embedding_device.py.
"""
import inspect
from types import SimpleNamespace

import pytest

import src.core.encoder as enc
from FlagEmbedding.finetune.embedder.encoder_only.m3 import runner as m3_runner


class FakeAutoModel:
    calls = []

    @staticmethod
    def from_pretrained(name, **kwargs):
        FakeAutoModel.calls.append(kwargs)
        attn = kwargs.get("attn_implementation", "sdpa")
        return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn))


class FakeFlagModel:
    """Builds its backbone through the M3 runner's AutoModel, as FlagEmbedding does."""

    def __init__(self, name, use_fp16=True, devices=None):
        self.name, self.use_fp16, self.devices = name, use_fp16, devices
        self.model = SimpleNamespace(model=m3_runner.AutoModel.from_pretrained(name))


@pytest.fixture
def fakes(monkeypatch):
    FakeAutoModel.calls = []
    monkeypatch.setattr(m3_runner, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(enc, "BGEM3FlagModel", FakeFlagModel)
    return monkeypatch


def test_the_patched_seam_exists_in_the_installed_flagembedding():
    # load_bge_m3 relies on FlagEmbedding's M3 runner loading the backbone through
    # its module-level AutoModel. If that changes, the loader falls back to CPU.
    assert hasattr(m3_runner, "AutoModel")
    assert "AutoModel.from_pretrained" in inspect.getsource(m3_runner)


def test_on_mps_the_backbone_is_loaded_with_eager_attention(fakes):
    fakes.setattr(enc, "_mps_available", lambda: True)

    model = enc.load_bge_m3("BAAI/bge-m3")

    assert FakeAutoModel.calls == [{"attn_implementation": "eager"}]
    assert model.model.model.config._attn_implementation == "eager"
    assert model.devices is None and model.use_fp16 is True  # stays on MPS, fp16
    assert m3_runner.AutoModel is FakeAutoModel  # the temporary wrapper was removed


def test_without_mps_nothing_is_patched(fakes):
    fakes.setattr(enc, "_mps_available", lambda: False)

    model = enc.load_bge_m3("BAAI/bge-m3", use_fp16=False)

    assert FakeAutoModel.calls == [{}]
    assert model.use_fp16 is False and model.devices is None


def test_falls_back_to_cpu_when_eager_cannot_be_confirmed(fakes, capsys):
    class IgnoresTheRunner(FakeFlagModel):
        def __init__(self, name, use_fp16=True, devices=None):
            self.name, self.use_fp16, self.devices = name, use_fp16, devices
            backbone = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
            self.model = SimpleNamespace(model=backbone)

    fakes.setattr(enc, "_mps_available", lambda: True)
    fakes.setattr(enc, "BGEM3FlagModel", IgnoresTheRunner)

    model = enc.load_bge_m3("BAAI/bge-m3")

    assert (model.devices, model.use_fp16) == ("cpu", False)
    assert "using CPU" in capsys.readouterr().err


def test_falls_back_to_cpu_when_flagembedding_internals_moved(fakes):
    class NoBackbone(FakeFlagModel):
        def __init__(self, name, use_fp16=True, devices=None):
            self.name, self.use_fp16, self.devices = name, use_fp16, devices
            self.model = object()  # .model.model raises AttributeError

    fakes.setattr(enc, "_mps_available", lambda: True)
    fakes.setattr(enc, "BGEM3FlagModel", NoBackbone)

    assert enc.load_bge_m3("BAAI/bge-m3").devices == "cpu"
    assert m3_runner.AutoModel is FakeAutoModel


def test_runner_is_restored_even_if_loading_raises(fakes):
    class Explodes:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights corrupt")

    fakes.setattr(enc, "_mps_available", lambda: True)
    fakes.setattr(enc, "BGEM3FlagModel", Explodes)

    with pytest.raises(RuntimeError):
        enc.load_bge_m3("BAAI/bge-m3")
    assert m3_runner.AutoModel is FakeAutoModel
