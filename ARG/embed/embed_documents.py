from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma

e_model = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)

splitText = [
    "你快乐吗",
    "我很快乐",
    "第一步就是向后退一步",
    "快乐吗，没有什么道理"
]
raw_documents = TextLoader("../files/load_test.txt").load()


embeddings = e_model.embed_documents(splitText)
print(len(embeddings))
for i in range(len(embeddings)):
    print(f"Embedding {i}: {embeddings[i]}")

db = Chroma.from_documents(raw_documents, embeddings)

