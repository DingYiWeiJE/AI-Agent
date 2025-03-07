from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)


db = FAISS.from_documents(texts, embeddings)

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
result = qa.invoke('Evay有女朋友吗？')
print(result)
