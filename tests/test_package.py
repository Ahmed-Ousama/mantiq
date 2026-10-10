import subprocess
import sys
from pathlib import Path

from mantiq.cag import utils

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cag_import_does_not_load_the_loaders():
    code = "import sys, mantiq.cag; print('langchain_community' in sys.modules)"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_load_model_does_not_trust_remote_code_by_default(monkeypatch):
    calls = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(utils, "AutoTokenizer", FakeAuto)
    monkeypatch.setattr(utils, "AutoModelForCausalLM", FakeAuto)

    utils.load_model("some/model", hf_token=None, quantized=False)

    assert calls["trust_remote_code"] is False
