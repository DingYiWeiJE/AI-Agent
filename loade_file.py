from langchain_core.prompts import load_prompt

prompt = load_prompt("./prompts/simple_prompt.json")
print(prompt.format(name="科学家", what="热爱搞发明"))