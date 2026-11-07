import os
import platform
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("LOBOTOMY_MODEL", ROOT / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"))
HF_DIR = ROOT / "models" / "qwen3.5-0.8b-hf"  # optional official checkpoint, for reference tests

needs_model = pytest.mark.skipif(not MODEL.exists(), reason=f"model file not found: {MODEL}")


def backends_to_test() -> list[str]:
    names = os.environ.get("LOBOTOMY_TEST_BACKENDS")
    if names:
        return names.split(",")
    return ["mlx"] if platform.system() == "Darwin" and platform.machine() == "arm64" else ["cuda"]


@pytest.fixture(scope="session")
def gguf_model():
    if not MODEL.exists():
        pytest.skip(f"model file not found: {MODEL}")
    from lobotomy.gguf_loader import open_gguf

    return open_gguf(MODEL)


@pytest.fixture(scope="session")
def config(gguf_model):
    return gguf_model.config


@pytest.fixture(scope="session")
def tokenizer(gguf_model):
    from lobotomy.tokenizer import ChatTokenizer

    return ChatTokenizer(gguf_model.tokenizer)


@pytest.fixture(scope="session", params=backends_to_test())
def engine(request):
    if not MODEL.exists():
        pytest.skip(f"model file not found: {MODEL}")
    from lobotomy.engine import Engine

    try:
        eng = Engine(MODEL, backend=request.param)
    except NotImplementedError as e:
        pytest.skip(str(e))
    return eng


@pytest.fixture
def fresh(engine):
    """The shared engine, with no edits and no chat, restored afterwards."""
    from lobotomy.sampling import SamplingParams

    engine.reset_edits()
    engine.reset_chat()
    engine.sampling = SamplingParams(temperature=0.0, max_tokens=24)
    yield engine
    engine.reset_edits()
    engine.reset_chat()


@pytest.fixture
def prompt_ids(tokenizer):
    return tokenizer.encode_chat([{"role": "user", "content": "Name three colours of the rainbow."}])


def assert_close(a, b, atol=1e-6):
    np.testing.assert_allclose(np.asarray(a, np.float32), np.asarray(b, np.float32), atol=atol, rtol=0)
