from fastapi import FastAPI, File, UploadFile, HTTPException, status
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from dotenv import load_dotenv
import filelock

load_dotenv()

app = FastAPI(title="Document Processing API")


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_embeddings():
    return OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
    )


def load_or_create_vector_db(embeddings, texts, index_path="jiaoer"):
    with filelock.FileLock(f"{index_path}.lock"):  # 防止并发写入
        if os.path.exists(index_path):
            db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(texts)
            db.save_local(index_path)
        else:
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(index_path)
        return db


@app.post("/process-document/",
          status_code=status.HTTP_201_CREATED,
          summary="Process uploaded TXT document",
          response_description="Document processing result")
async def process_document(file: UploadFile = File(..., description="TXT file to process")):
    tmp_file_path = None
    try:
        # 验证文件类型
        if not file.filename.endswith(".txt"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only TXT files are allowed"
            )

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # 加载文档
        try:
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file encoding (must be UTF-8)"
            )

        # 处理流程
        texts = split_documents(documents)
        embeddings = create_embeddings()

        try:
            db = load_or_create_vector_db(embeddings, texts)
        except ConnectionError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama embedding service unavailable"
            )

        return {"status": "success", "message": f"Processed {len(texts)} text chunks"}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9527)