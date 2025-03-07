from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatDeepSeek(
	model="deepseek-chat",
	api_key=os.getenv("DEEPSEEK_API_KEY"),
)

response = llm.invoke("你是谁")
print(response.content)