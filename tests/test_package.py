import subprocess
import sys
from pathlib import Path

import torch

import mantiq
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


def test_quantizes_only_when_a_gpu_can(monkeypatch):
    calls = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(utils, "AutoTokenizer", FakeAuto)
    monkeypatch.setattr(utils, "AutoModelForCausalLM", FakeAuto)

    monkeypatch.setattr(utils, "can_quantize", lambda: False)
    utils.load_model("some/model")
    assert calls["quantization_config"] is None

    monkeypatch.setattr(utils, "can_quantize", lambda: True)
    utils.load_model("some/model")
    assert calls["quantization_config"].load_in_4bit


def test_old_class_name_still_works():
    from mantiq import NaiiveCAG, NaiveCAG

    assert NaiiveCAG is NaiveCAG


def test_version_matches_the_installed_package():
    from importlib.metadata import version

    assert mantiq.__version__ == version("mantiq")


def test_loads_in_the_checkpoint_dtype_unless_told_otherwise(monkeypatch):
    calls = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(utils, "AutoTokenizer", FakeAuto)
    monkeypatch.setattr(utils, "AutoModelForCausalLM", FakeAuto)

    utils.load_model("some/model")
    assert calls["dtype"] == "auto"

    utils.load_model("some/model", model_kwargs={"dtype": torch.float32, "revision": "v1"})
    assert calls["dtype"] is torch.float32
    assert calls["revision"] == "v1"


def test_the_tokenizer_follows_the_model_revision_and_dtype_settings_are_kept(monkeypatch):
    calls = {}

    def recorder(name):
        class FakeAuto:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls[name] = kwargs
        return FakeAuto

    monkeypatch.setattr(utils, "AutoTokenizer", recorder("tokenizer"))
    monkeypatch.setattr(utils, "AutoModelForCausalLM", recorder("model"))

    utils.load_model("some/model", model_kwargs={"revision": "v1", "torch_dtype": torch.float32})

    assert calls["tokenizer"]["revision"] == "v1"
    assert calls["model"]["torch_dtype"] is torch.float32
    assert "dtype" not in calls["model"]  # it would override torch_dtype
