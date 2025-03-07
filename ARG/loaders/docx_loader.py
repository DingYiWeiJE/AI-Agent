from langchain_community.document_loaders import Docx2txtLoader

load_file=Docx2txtLoader('../files/load_test.docx')
doc=load_file.load()

print(doc)