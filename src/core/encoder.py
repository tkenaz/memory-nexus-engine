"""The one place where the embedding model is loaded.

Why this exists: on Apple Silicon (MPS), PyTorch's scaled-dot-product attention
returns wrong BGE-M3 dense vectors for texts of about 8 tokens or fewer when they
are encoded on their own (measured with torch 2.8 / FlagEmbedding 1.3.5: cosine to
the CPU vector 0.14-0.40 for "lisbon", short Russian queries, tag names; 1.0 for
longer texts). Inside a batch a short text is padded to the longest one and comes
out right, which hides the problem. Short queries and tags are exactly the texts
that matter for retrieval.

Loading the model with eager attention gives vectors identical to the CPU
reference (cosine 1.0000) and stays about 6x faster than CPU. FlagEmbedding does
not expose an attention option, so the loader is wrapped for the duration of the
load. If FlagEmbedding's internals change, the code falls back to running on CPU
rather than to silently wrong vectors.
"""
import contextlib
import sys
from typing import Iterator

from FlagEmbedding import BGEM3FlagModel


def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


@contextlib.contextmanager
def _eager_attention() -> Iterator[None]:
    """Make FlagEmbedding's M3 loader pass attn_implementation='eager'."""
    from FlagEmbedding.finetune.embedder.encoder_only.m3 import runner as m3_runner

    real_auto_model = m3_runner.AutoModel

    class _EagerAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("attn_implementation", "eager")
            return real_auto_model.from_pretrained(*args, **kwargs)

    m3_runner.AutoModel = _EagerAutoModel
    try:
        yield
    finally:
        m3_runner.AutoModel = real_auto_model


def load_bge_m3(model_name: str, use_fp16: bool = True) -> BGEM3FlagModel:
    """Load BGE-M3 with dense vectors that are correct on every device."""
    if not _mps_available():
        return BGEM3FlagModel(model_name, use_fp16=use_fp16)

    try:
        with _eager_attention():
            model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        backbone = model.model.model
        if getattr(backbone.config, "_attn_implementation", None) == "eager":
            return model
    except (ImportError, AttributeError):
        pass

    print(
        "memory-nexus: could not load BGE-M3 with eager attention on MPS; "
        "using CPU to avoid wrong vectors for short texts",
        file=sys.stderr,
    )
    return BGEM3FlagModel(model_name, use_fp16=False, devices="cpu")
