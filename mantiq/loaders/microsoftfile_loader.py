from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from .abstract_loader import AbstractLoader

class MicrosoftFile(AbstractLoader):
    def create_loader(self, endpoint, key, file_path, api_model="prebuilt-layout", **kwargs):
        # Pass **kwargs to AzureAIDocumentIntelligenceLoader
        return AzureAIDocumentIntelligenceLoader(
            api_endpoint=endpoint, api_key=key, file_path=file_path, api_model=api_model, **kwargs
        )
