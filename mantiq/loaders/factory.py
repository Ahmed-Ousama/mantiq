from .csv_loader import CSV
from .markdown_loader import Markdown
from .json_loader import JSON
from .html_loader import HTML
from .directory_loader import Directory
from .webpage_loader import WebPage
from .pdf_loader import PDF
from .microsoftfile_loader import MicrosoftFile
from .text_loader import Text

class LoaderFactory:
    loaders = {
        "CSV": CSV,
        "Markdown": Markdown,
        "JSON": JSON,
        "HTML": HTML,
        "Directory": Directory,
        "WebPage": WebPage,
        "PDF": PDF,
        "MicrosoftFile": MicrosoftFile,
        "Text": Text,
    }

    @staticmethod
    def get_loader(loader_type, *args, **kwargs):
        # Match loader types case-insensitively, so "pdf" works as well as "PDF"
        loaders = {name.lower(): loader for name, loader in LoaderFactory.loaders.items()}
        if loader_type.lower() in loaders:
            # Pass **kwargs to the loader's load method
            return loaders[loader_type.lower()]().load(*args, **kwargs)
        supported = ", ".join(LoaderFactory.loaders)
        raise ValueError(f"Loader type '{loader_type}' is not supported. Supported types: {supported}")
