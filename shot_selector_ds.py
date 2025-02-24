from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

examples = [
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
        "question": "什么是快乐星球",
        "answer": """
            这里需要跟进问题吗： 是的。
            跟进： 快乐星球是什么？
            中间答案： 快乐星球是电影《快乐星球》中的星球。
            跟进： 电影《快乐星球》是什么？
            中间答案： 电影《快乐星球》是一部关于快乐星球的科幻电影。
            所以最终答案是： 快乐星球是电影《快乐星球》中的星球。
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

class DeepSeekEmbeddings:
    def __init__(self, model_name="deepseek-local"):
        self.model_name = model_name
        self.embedding_dim = 1024  # 必须与实际模型维度一致

    def embed_documents(self, texts):
        return [
            [float(i%100)/100 for i in range(self.embedding_dim)]
            for _ in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    DeepSeekEmbeddings(),
    Chroma,
    k=1
)

question = "快乐星球在哪里"
selected_examples = example_selector.select_examples({"question": question})

print(f"最相似的示例:{question}")
for example in selected_examples:
    print("\n".join([f"{k}: {v}" for k, v in example.items()]))