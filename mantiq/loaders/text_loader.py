from langchain_community.document_loaders import TextLoader
from .abstract_loader import AbstractLoader

class Text(AbstractLoader):
    def create_loader(self, file_path, encoding="utf-8", **kwargs):
        # Pass **kwargs to TextLoader
        return TextLoader(file_path, encoding=encoding, **kwargs)
