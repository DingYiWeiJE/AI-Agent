from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

full_template = """{Character}
{Behavior}
{Prohibit}"""
full_prompt = PromptTemplate.from_template(full_template)

Character_template = "你是{person}, 你有着{attribute}"
Character_prompt = PromptTemplate.from_template(
    Character_template
)

behavior_template = "你会遵从以下行为：\n{behavior_list}"
behavior_prompt = PromptTemplate.from_template(behavior_template)

prohibit_template = "你不能遵从以下行为：\n{prohibit_list}"
prohibit_prompt = PromptTemplate.from_template(prohibit_template)

input_prompts = [
    ("Character", Character_prompt),
    ("Behavior", behavior_prompt),
    ("Prohibit", prohibit_prompt),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts=input_prompts
)

print(pipeline_prompt.format(
    person="科学家",
    attribute="严谨",
    behavior_list="1. 确保实验结果准确\n2. 记录实验过程\n3. 分析实验数据",
    prohibit_list="1. 不做实验\n2. 不记录实验过程\n3. 不分析实验数据"
))