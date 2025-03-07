from langchain_community.document_loaders import TextLoader

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader('./state_of_the_union2.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings(api_key=api_key)


db = FAISS.from_documents(texts, embeddings)

# db.save_local('出来吧神龙')


qa = RetrievalQA.from_chain_type(llm=OpenAI(api_key=api_key), chain_type="stuff", retriever=db.as_retriever())
result2 = qa.invoke('流量1005G要花多少钱的答案')
print(result2)
