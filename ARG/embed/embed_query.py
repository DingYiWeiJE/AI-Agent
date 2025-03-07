from langchain_ollama import OllamaEmbeddings
e_model = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)
embeddings = e_model.embed_query(
    "What is the capital of France?"
)

for i in range(len(embeddings)):
    print(f"Embedding {i}: {embeddings[i]}")
print(f"总数量：{len(embeddings)}")
print(embeddings[:5])
