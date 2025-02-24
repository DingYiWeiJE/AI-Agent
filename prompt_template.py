from langchain_ollama import ChatOllama
from langchain_core.prompts import StringPromptTemplate


def hello_word(address):
    print("Hello, world!" + address)
    return f"Hello, {address}!"


PROMPT = """\
    You are a helpful assistant that answers questions based on the provided context.
    function name: {function_name}
    source code:
    {source_code}
    explain:
"""

import inspect  # 这个包可以根据函数名，获取到函数源代码


def get_source_code(function_name):
    # 获取源代码
    return inspect.getsource(function_name)


class CustmPrompt(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        source_code = get_source_code(kwargs["function_name"])

        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__, source_code=source_code
        )
        return prompt


a = CustmPrompt(input_variables=["function_name"])
pm = a.format(function_name=hello_word)

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

result = llm.invoke(pm)
print(result.content)
