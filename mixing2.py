from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

# 定义最终模板框架
full_template = """{introduction}\n{example}\n{start}"""
full_prompt = PromptTemplate.from_template(full_template)

# 定义子模板
introduction_template = """角色设定：你正在模仿{person}的对话风格。"""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """示例对话：\nQ：{example_q}\nA：{example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

start_template = """当前对话：\nQ：{input}\nA："""
start_prompt = PromptTemplate.from_template(start_template)

# 构建管道
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt)
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts=input_prompts
)

output = pipeline_prompt.format(
    person="科学家",
    example_q="如何验证假设？",
    example_a="通过设计对照实验",
    input="量子纠缠的原理是什么？"
)

print(output)