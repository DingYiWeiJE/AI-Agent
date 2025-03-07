from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

response = llm.invoke([
    HumanMessage(content="望梅止渴是什么意思")
])
print(response.content)