from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="../files/load_test.json",
    jq_schema=".test",
)
docs = loader.load()

print(docs[0].page_content)