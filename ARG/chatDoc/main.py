from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_unstructured import UnstructuredLoader


class ChatFileDoc():
    @staticmethod
    def getFile():
        load_file = Docx2txtLoader('../files/load_test.docx')
        return load_file.load()


print(ChatFileDoc.getFile())


class ChatFilePdf():
    @staticmethod
    def getFile():
        try:
            load_file = PyPDFLoader('../files/load_test.pdf')
            return load_file.load()
        except Exception as e:
            print(f"Error Loading pdf: {e}")


print(ChatFilePdf.getFile())


class ChatFileExcel():
    @staticmethod
    def getFile():
        try:
            loader = UnstructuredLoader("../files/load_test.xlsx", mode="elements")
            return loader.load()
        except Exception as e:
            print(f"Error Loading xlsx: {e}")
            return e


print(ChatFileExcel.getFile())
