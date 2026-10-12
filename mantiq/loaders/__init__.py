try:
    import langchain_community  # noqa: F401
except ImportError as e:
    raise ImportError('mantiq.loaders needs the optional loader dependencies: pip install "mantiq[loaders]"') from e

from .csv_loader import CSV
from .directory_loader import Directory
from .html_loader import HTML
from .json_loader import JSON
from .markdown_loader import Markdown
from .microsoftfile_loader import MicrosoftFile
from .pdf_loader import PDF
from .text_loader import Text
from .webpage_loader import WebPage
from .factory import LoaderFactory

__all__ = ["CSV", "Directory", "HTML", "JSON", "Markdown", "MicrosoftFile", "PDF", "Text", "WebPage", "LoaderFactory"]