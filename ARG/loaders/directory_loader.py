from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("../files/", glob="*.xlsx")
docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)