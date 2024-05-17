# 利用GPT-2进行通用的文本补全任务

我们将从头到尾探索一个使用GPT-2通用模型的示例。我们运行的这个示例的目标是确定GPT模型可以达到的抽象推理水平。

## 定义文本补全函数

我们定义一个函数，输入模型、分词器以及提示词（prompt），输出补全在prompt之后的内容：

```{.python .input}
def complete_text(model, tokenizer, prompt):
    device = model.device
    # 将输入构建为tensor batch
    input_batch = tokenizer(prompt, return_tensors="pt")
    input_batch = {k: v.to(device) for k, v in input_batch.items()}
    # prompt序列的长度
    prompt_length = input_batch["input_ids"].size(1)
    # 调用模型的generate方法，根据prompt生成文本，最大生成长度为256
    output_ids = model.generate(**input_batch, max_new_tokens=256)
    # 将输出序列解码得到字符串
    answer = tokenizer.batch_decode(output_ids[:, prompt_length:])[0]
    # 只取答案的第一段
    answer = answer.strip().split("\n\n")[0]
    return answer
```

## 加载预训练的GPT-2模型

我们使用Hugging Face Transformer加载GPT-2的预训练模型

```{.python .input}
from transformers import GPT2LMHeadModel, AutoTokenizer


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

## 文本补全

现在我们尝试让GPT-2针对伊曼努尔·康德的话进行补全：

```{.python .input}
prompt = "Human reason, in one sphere of its cognition, is called upon to consider questions, which it cannot decline, as they are presented by its own nature, but which it cannot answer, as they transcend every faculty of the mind."
answer = complete_text(model, tokenizer, prompt)
print(answer)
```

从上述代码的输出可以看到GPT-2对伊曼努尔·康德的话进行了补全。

由于随机性，补全的内容可能每次都不同。但是我们基本可以得出结论，GPT-2可以给出看似合理的文本补全，但是不太能经得起推敲。
