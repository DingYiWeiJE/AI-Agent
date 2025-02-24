from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

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
    {"input": "high", "output": "low"},
    {"input": "rich", "output": "poor"},
    {"input": "beautiful", "output": "ugly"},
    {"input": "full", "output": "empty"},
    {"input": "young", "output": "old"},
    {"input": "loud", "output": "quiet"},
    {"input": "soft", "output": "hard"},
    {"input": "strong", "output": "fragile"},
    {"input": "sweet", "output": "sour"},
    {"input": "clean", "output": "messy"},
    {"input": "open", "output": "closed"},
    {"input": "warm", "output": "cool"}
]

# 构造提示词模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n 反义：{output}"
)

# 调用长度示例选择器
example_selector = LengthBasedExampleSelector(
    examples=examples,  # 传入示例提示词组
    example_prompt=example_prompt,  # 传入的提示词模板
    max_length=20  # 格式化后，的提示词的最大长度
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义:",
    input_variables=["adjective"]
)

print(dynamic_prompt.format(adjective="forward"))
