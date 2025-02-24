from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_ollama import ChatOllama
from onnxruntime.transformers.models.bart import export

example = [
    {
        "question": "谁的寿命更长，悟空还是如来",
        "answer":
            """
            这里需要跟进问题吗： 是的。
            跟进： 悟空在记载中，最后能够追溯到他有多大年龄？
            中间答案： 五千岁
            跟进： 如来呢？
            中间答案： 佛祖在记载中，最后能够追溯到他多大年龄是一万岁。
            所以最终答案是： 悟空五千年，如来一万岁，所以如来寿命更长。
            """
    },
    {
        "question": "高中重要还是大学重要",
        "answer":
            """
            这里需要跟进问题吗： 是的。
            跟进：高中的重要性在于什么？
            中间答案： 高中的重要性是考上一个好的大学，考不上好的大学大概率就完蛋了。
            跟进： 大学的重要性是什么？
            中间答案： 大学的重要性是学会技能，适应社会，哪怕学校不好，也能很小几率靠自己打拼出来。
            所以最终答案是： 高中更重要。
            """
    },
]


example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题：{question}\n{answer}")

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,  # 这里需要是PromptTemplate模板
    examples=example,  # 这里需要是列表
    suffix="问题：{input}",
    input_variables=["input"]
)

print(prompt.format(input="谁更厉害，孙悟空还是如来"))
