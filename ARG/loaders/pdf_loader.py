from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../files/load_test.pdf")
docs = loader.load()
print(docs[0])

