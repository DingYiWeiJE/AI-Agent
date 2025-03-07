from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# 加载文档
def load_documents(file_path):
    loader = TextLoader(file_path)
    return loader.load()


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def create_embeddings(base_url="http://localhost:11434", model="deepseek-r1:1.5b"):
    return OllamaEmbeddings(base_url=base_url, model=model)


def load_or_create_vector_db(embeddings, texts, index_path="faiss_index"):
    if os.path.exists(index_path):
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(texts)
    else:
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(index_path)
    return db


# 主函数
    documents = load_documents('./state_of_the_union2.txt')

    texts = split_documents(documents)

    embeddings = create_embeddings()

    db = load_or_create_vector_db(embeddings, texts)

    db.save_local("faiss_index")
