from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10,
#     length_function=len,
#     add_start_index=True
# )
text_splitter = CharacterTextSplitter(chunk_size=105, chunk_overlap=5)


documents = TextLoader("../files/state_of_the_union.txt").load()
docs = text_splitter.split_text(documents[0].page_content)

# db = Chroma.from_texts(docs, embeddings)
db = Chroma.from_documents(docs, embeddings)
query = "根铁杵磨成一个绣花针"
# result = db.similarity_search_with_score(query)
retriever = db.as_retriever()
result = retriever.invoke(query)

print(result)
print(len(result))
