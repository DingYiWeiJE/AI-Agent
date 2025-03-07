from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=api_key)

response = llm.invoke([
    HumanMessage(content="用三个表情来形容中彩票的样子")
])
print(response)