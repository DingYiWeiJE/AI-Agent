from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)

# 加载向量数据库
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 初始化 LLM
llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# 创建 RetrievalQA 实例
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 4}))

# 提问
result = qa.invoke('张帅是谁')
print(result)
