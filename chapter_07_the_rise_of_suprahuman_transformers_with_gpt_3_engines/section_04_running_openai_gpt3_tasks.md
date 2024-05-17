# 运行GPT-3的任务

在本节中，我们会以两种不同的方式来运行GPT-3任务：

- 不使用代码，在线运行GPT-3任务
- 使用代码运行GPT-3任务

## 在线运行NLP任务

首先，让我们定义一组提示（prompt）和响应（response）的标准结构:

- `N`= NLP任务的名称（输入）
- `E`= 对GPT-3引擎的说明。E位于T之前（输入）
- `T`= 我们希望GPT-3处理的文本或内容（输入）
- `S`= 向GPT-3说明我们的期望。必要时添加在T之后（输入）
- `R`= GPT-3的响应（输出）

上述提示的结构只是一个指导性的框架。但GPT-3非常灵活，可能会有许多变化。实际应用中,我们可以根据具体需求灵活调整提示的结构和内容。

下面我们为一些任务构建prompt.

### 知识问答任务

```{.python .input}
N = ""
E = "Q: "
T = "Who was the president of the United States in 1965?"
S = "\nA:"
prompt = N + E + T + S
print(prompt)
```

这里`E = "Q: "`向GPT-3说明了接下来是一个问题，`S = "\nA:"`向GPT-3说明了我们期望它给出一个回答。

另外，这里`R`就是GPt-3针对`T`所给出的回答。

### Movie to Emoji

Movie to Emoji是一个使用emoji表情符号来表示电影名称的任务。

```{.python .input}
N = ""
E = "Back to Future: 👨👴🚗⌚\nBatman: 🤵🦇\n"
T = ""
S = "Transformers: "
prompt = N + E + T + S
print(prompt)
```

这里的`E`给了GPT-3两个示例（2-shot），`S`给出电影标题加冒号和空格，表示期望GPT-3能像示例中一样对新的电影标题生成emoji.

这里GPT-3给出的`R`可以是`🚗🤖`.

### 给小学生解释问题

```{.python .input}
N = ""
E = "My second grader asked me what this passage means:\n\n"
T = """The initial conclusions can be divided into two categories: facts and
fiction. The facts are that OpenAI has one of the most powerful NLP services
in the world. The main facts are: OpenAI engines are powerful zero-shot that
require no hunting for all kinds of transformer models, no pre-training,
and no fine-tuning. The supercomputers used to train the models are unique.
If the prompt is well-designed, we obtain surprisingly accurate responses.
Implementing the NLP tasks in this section required a copy and paste action
that any software beginner can perform. Fiction begins with dystopian and
hype assertions AI will replace data scientists and AI specialists. Is that
true? Before answering that question, first ask yourself the following
questions about the example we just ran: How do we know the sentence was
incorrect in the first place? How do we know the answer is correct without
us humans reading and confirming this? How did the engine know it was a
grammar correction task? If the response is incorrect, how can we understand
what happened to help improve the prompt or revert to manual mode in a welldesigned human interface? The truth is that humans will need to intervene
to answers these questions manually, with rule-bases, quality control
automated pipelines, and many other tools. The facts are convincing. It is
true that running an NLP task requires little development. The fiction is
not convincing. Humans are still required. OpenAI engines are not there to
replace humans but to help them perform more high-level gratifying tasks.
You can now fly a jet without having to build it!\n\n"""
S = "\n\nI rephrase it for him, in plain language a second grader can understand: "
prompt = N + E + T + S
print(prompt)
```

这里`E`告诉GPT-3，有一个二年级学生来询问文章的意思。`T`告诉GPT-3文章的正文。`S`的目的是期望GPT-3能在其后补全一段二年级学生可以听懂的文章释义。

读者可以在OpenAI的官网或镜像网站尝试向GPT-3或其他模型发送以上prompt，看看GPT-3给出的回答怎么样。

## GPT-3引擎入门

OpenAI拥有世界上最强大的Transformer引擎之一。一个GPT-3模型可以执行数百种任务。GPT-3甚至可以做一些它没有被训练过的任务。

这一部分介绍如何利用API来使用GPT-3引擎。

### 第1步：安装

```{.python .input}
!pip install openai
import openai
```

### 第2步：输入API key

首先需要去OpenAI官网或代理机构（x宝，x夕夕）购买一个API key，然后通过代码配置api_key:

```{.python }
openai.api_key = "[REPLACE THIS WITH YOUR API KEY]"
```

### 第3步：运行NLP任务

```{.python }
response = openai.Completion.create(
    engine="davinci",
    prompt="Original: She no went to the market.\nStandard American English:",
    temperature=0,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
)
print(response)
```

上面代码中的prompt描述了一个任务：将“She no went to the market.”这个句子转换为标准英语。

## 小结

可以看到，通过构建GPT-3可以理解的prompt，未经微调的GPT-3也可以很好地完成prompt中指定的任务。因此，开发者可以利用GPT-3的这一特性自行封装许多功能性的模块并用于具体的业务场景中。
