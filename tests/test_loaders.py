import http.server
import importlib.util
import threading
from functools import partial

import pytest

pytest.importorskip("langchain_community")

import mantiq
from mantiq.loaders import LoaderFactory


def test_loaders_are_reachable_from_the_package():
    assert mantiq.loaders.LoaderFactory is LoaderFactory


def test_json_accepts_a_custom_jq_schema(tmp_path):
    pytest.importorskip("jq")
    path = tmp_path / "data.json"
    path.write_text('{"items": [{"text": "hello"}, {"text": "world"}]}')

    text = LoaderFactory.get_loader("JSON", str(path), jq_schema=".items[].text")

    assert "hello" in text and "world" in text


def test_json_keeps_its_default_schema(tmp_path):
    pytest.importorskip("jq")
    path = tmp_path / "data.json"
    path.write_text('{"data": [{"attributes": {"message": "hi there"}}]}')

    assert "hi there" in LoaderFactory.get_loader("JSON", str(path))


def test_csv(tmp_path):
    path = tmp_path / "products.csv"
    path.write_text("name,price\nTV,500\nFridge,800\n")

    text = LoaderFactory.get_loader("CSV", str(path))

    assert "TV" in text and "800" in text


def test_loader_types_are_case_insensitive(tmp_path):
    path = tmp_path / "products.csv"
    path.write_text("name,price\nTV,500\n")

    assert LoaderFactory.get_loader("csv", str(path)) == LoaderFactory.get_loader("CSV", str(path))


def test_unsupported_loader_lists_the_supported_types():
    with pytest.raises(ValueError, match="Supported types: .*PDF"):
        LoaderFactory.get_loader("Excel", "data.xlsx")


def test_text(tmp_path):
    path = tmp_path / "faq.txt"
    path.write_text("الشحن مجاني للطلبات فوق 500 جنيه.", encoding="utf-8")

    assert "الشحن مجاني" in LoaderFactory.get_loader("Text", str(path))


def test_markdown_and_html(tmp_path):
    pytest.importorskip("unstructured")
    md = tmp_path / "notes.md"
    md.write_text("# Products\n\nThe TV has 4K resolution.\n")
    html = tmp_path / "page.html"
    html.write_text("<html><body><p>The fridge has an A++ rating.</p></body></html>")

    assert "4K resolution" in LoaderFactory.get_loader("Markdown", str(md))
    assert "A++ rating" in LoaderFactory.get_loader("HTML", str(html))


def test_webpage(tmp_path):
    pytest.importorskip("bs4")
    (tmp_path / "index.html").write_text("<html><body><p>Washing machine with AI cycles.</p></body></html>")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        text = LoaderFactory.get_loader("WebPage", f"http://127.0.0.1:{server.server_port}/index.html")
    finally:
        server.shutdown()

    assert "AI cycles" in text


def test_microsoft_file_is_registered():
    if importlib.util.find_spec("azure") and importlib.util.find_spec("azure.ai.documentintelligence"):
        pytest.skip("would call the Azure API")

    # Reaching the Azure import (instead of "not supported") proves the factory knows the loader.
    with pytest.raises(ImportError):
        LoaderFactory.get_loader("MicrosoftFile", "https://example.invalid", "key", "file.pdf")
