from langchain_community.document_loaders import TextLoader

loader = TextLoader("../files/load_test.md")
# loader = TextLoader("./files/load_test.csv")
docs = loader.load()
print(docs[0].page_content)  # 输出文件主体文本
print(docs[0].metadata)

