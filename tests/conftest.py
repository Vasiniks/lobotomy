import json
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
    return ["mlx"] if platform.system() == "Darwin" and platform.machine() == "arm64" else ["torch"]


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
def official(config):
    """The official checkpoint's folder, for the reference tests. Skips when it isn't
    downloaded, or when it's a different size from the model under test (the
    folder holds the 0.8B; LOBOTOMY_MODEL may point at a 2B)."""
    if not list(HF_DIR.glob("*.safetensors")):
        pytest.skip("official checkpoint not downloaded")
    c = json.loads((HF_DIR / "config.json").read_text())
    c = c.get("text_config", c)
    ref = (c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"])
    if ref != (config.n_layers, config.hidden_size, config.intermediate_size):
        pytest.skip(f"the official checkpoint in {HF_DIR.name} is a different size from {MODEL.name}")
    return HF_DIR


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
