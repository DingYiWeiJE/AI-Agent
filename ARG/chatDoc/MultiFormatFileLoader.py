from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_unstructured import UnstructuredLoader
import os


class MultiFormatFileLoader:
    loaders = {
        '.docx': Docx2txtLoader,
        '.pdf': PyPDFLoader,
        '.unstructured': UnstructuredLoader
    }

    def __init__(self, file_path):
        self.file_path = file_path

    def getLoader(self):
        file_extension = os.path.splitext(self.file_path)[1].lower()
        loader = self.loaders.get(file_extension)
        if loader:
            return loader
        elif file_extension in ['.docx', '.pdf', '.txt', '.html', '.xlsx']:
            return self.loaders.get('unstructured')
        else:
            return None

    def load(self):
        file_extension = os.path.splitext(self.file_path)[1].lower()
        loader_class = self.loaders.get(file_extension)
        if loader_class:
            loader = loader_class(self.file_path)
            return loader.load()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")


multi_loader = MultiFormatFileLoader("../files/load_test.pdf")
print(multi_loader.load())
