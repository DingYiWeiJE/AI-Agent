from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

fs = LocalFileStore("./embed_cache")  # 新增实例化文件存储对象

e_model = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    e_model,
    fs,
    namespace=e_model.model.replace(":", "_")
)

test_texts = ["万一奥特曼打不过小怪兽"]
embeddings = cached_embeddings.embed_documents(test_texts)

# 查看缓存文件
print(list(fs.yield_keys()))  # 输出生成的键名