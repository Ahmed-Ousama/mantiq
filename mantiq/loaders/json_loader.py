from langchain_community.document_loaders import JSONLoader
from .abstract_loader import AbstractLoader

class JSON(AbstractLoader):
    def __init__(self, jq_schema=".data[].attributes.message"):
        self.jq_schema = jq_schema

    def create_loader(self, file_path, **kwargs):
        # kwargs may override the default jq_schema and text_content
        kwargs.setdefault("jq_schema", self.jq_schema)
        kwargs.setdefault("text_content", False)
        return JSONLoader(file_path, **kwargs)
