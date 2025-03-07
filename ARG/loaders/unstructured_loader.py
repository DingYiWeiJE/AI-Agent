from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader("../files/load_test.xlsx", mode="elements")
docs = loader.load()
print(docs)
