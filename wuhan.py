from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="deepseek-r1:32b",
    temperature=0.7,
    base_url="http://10.1.4.136:11434"
)

response = llm.invoke([
    HumanMessage(content="给我讲个刘伯温的具有教育意义的故事")
])
print(response.content)