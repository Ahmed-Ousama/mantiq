import pytest

from mantiq.cag import NaiiveCAG, naiive_cag
from helpers import make_llama, make_tokenizer


@pytest.fixture
def make_cag(monkeypatch):
    """Build a NaiiveCAG around a tiny local model instead of downloading one."""

    def _make(tokenizer=None, model=None, model_name="tiny-llama", **kwargs):
        tokenizer = tokenizer or make_tokenizer()
        model = model or make_llama(tokenizer)
        monkeypatch.setattr(naiive_cag, "load_model", lambda *args, **kw: (model, tokenizer))
        return NaiiveCAG(model_name, hf_token=None, **kwargs)

    return _make
