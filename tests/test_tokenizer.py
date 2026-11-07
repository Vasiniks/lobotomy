import pytest

from conftest import HF_DIR, needs_model

TEXTS = [
    "Hello, world!",
    "  leading spaces and\ttabs\n\nnewlines",
    "Numbers 12345 and 3.14159, symbols #$%^&*()",
    "Ünïcödé, emoji 🧠🔪, 中文测试, русский, العربية",
    "def f(x):\n    return x ** 2  # code",
    "<|im_start|>user\nhi<|im_end|>\n<think>\n\n</think>",
]


@needs_model
def test_roundtrip(tokenizer):
    for t in TEXTS:
        assert tokenizer.decode(tokenizer.encode(t)) == t


@needs_model
def test_special_tokens_are_single_ids(tokenizer):
    for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
        assert len(tokenizer.encode(s)) == 1


@needs_model
@pytest.mark.skipif(not (HF_DIR / "tokenizer.json").exists(), reason="official tokenizer.json not downloaded")
def test_matches_official_tokenizer(tokenizer):
    from tokenizers import Tokenizer

    ref = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    for t in TEXTS:
        assert tokenizer.encode(t) == ref.encode(t, add_special_tokens=False).ids, t


@needs_model
def test_chat_template_generation_prompt(tokenizer):
    text = tokenizer.render_chat([{"role": "user", "content": "hi"}])
    assert text.startswith("<|im_start|>user\nhi<|im_end|>\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    thinking = tokenizer.render_chat([{"role": "user", "content": "hi"}], enable_thinking=True)
    assert thinking.endswith("<|im_start|>assistant\n<think>\n")
