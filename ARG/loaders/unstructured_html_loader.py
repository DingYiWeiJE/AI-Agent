from langchain.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
loader = UnstructuredHTMLLoader("../files/load_test.html")
docs = loader.load()

print(docs)