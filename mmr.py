from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# 假设已经有这么多的提示词示例组
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "rainy"},
    {"input": "windy", "output": "calm"},
    {"input": "hot", "output": "cold"},
    {"input": "fast", "output": "slow"},
    {"input": "big", "output": "small"},
    {"input": "bright", "output": "dark"},
    {"input": "strong", "output": "weak"},
    {"input": "clean", "output": "dirty"},
    {"input": "heavy", "output": "light"},
    {"input": "happy", "output": "angry"},
    {"input": "高兴", "output": "悲伤"},
    {"input": "愤怒", "output": "冷静"},
    {"input": "晴天", "output": "雨天"},
    {"input": "有风", "output": "平静"},
    {"input": "轻松", "output": "紧张"},
    {"input": "快", "output": "慢"},
    {"input": "亲切", "output": "冷漠"},
    {"input": "明亮", "output": "暗"},
    {"input": "强", "output": "弱"}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n反义：{output}"
)

# 初始化OpenAIEmbeddings时显式传递密钥
# embeddings = OpenAIEmbeddings(openai_api_key=api_key)
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    embeddings,  # 使用已初始化的embeddings对象
    FAISS,
    k=2
)

mmr_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="请根据以下示例生成反义词：",
    suffix="原词是：{adjective}\n反义词：",
    input_variables=["adjective"]
)

print(mmr_prompt.format(adjective="worried"))