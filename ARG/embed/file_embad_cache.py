from langchain.embeddings import CacheBackedEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

fs = LocalFileStore("./embed_cache")

e_model = OllamaEmbeddings(
    base_url="http://10.1.4.136:11434/",
    model="deepseek-r1:32b"
)

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    e_model,
    fs,
    namespace=e_model.model.replace(":", "_")
)

raw_documents = TextLoader("../files/load_test.txt").load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

db=FAISS.from_documents(documents, cached_embeddings)

result = db.similarity_search("他曾作为联军主帅平定了苏峻之乱，为稳定东晋政权，立下赫赫战功；他治下的荆州，史称“路不拾遗”")
print(len(result))
