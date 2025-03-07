from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("./files/load_test.txt") as f:
    zuizhonghuanxiang = f.read()
    print(zuizhonghuanxiang)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,  # 每个chunk的大小
    chunk_overlap=10,  # 每个chunk的重叠大小
    length_function=len,  # 计算每个chunk长度的函数
    add_start_index=True  # 是否在每个chunk的metadata中添加起始索引
)

doc = text_splitter.create_documents([zuizhonghuanxiang])

for document in doc:
    print(document, '\n')