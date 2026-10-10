from langchain_community.document_loaders import WebBaseLoader
from .abstract_loader import AbstractLoader

class WebPage(AbstractLoader):
    def create_loader(self, web_paths, **kwargs):
        # A single URL string would otherwise be read as a sequence of characters
        if isinstance(web_paths, str):
            web_paths = [web_paths]
        # Pass **kwargs to WebBaseLoader
        return WebBaseLoader(web_paths=web_paths, **kwargs)
