# 用T5做文本摘要

自然语言处理摘要任务提取文本的简洁部分。这一部分将首先介绍我们在本章中将使用的Hugging Face资源。然后我们将初始化一个T5-large变换器模型。最后,我们将看到如何使用T5来总结任何文档,包括法律和公司文档。

## Hugging Face资源

Hugging Face设计了一个在更高级别实现Transformer的框架。在:ref:`chapter-3`中，我们使用Hugging Face对BERT模型进行了微调，在:ref:`chapter-4`中，我们使用Hugging Face预训练了RoBERTa模型。在:ref:`chapter-7`中，我们使用Hugging Face体验了与GPT-2交互的过程。

Hugging Face在其框架内提供了三个主要资源:模型（models）、数据集（datasets）和指标（metrics）。

访问[huggingface.co](https://huggingface.co/)的[模型页面](https://huggingface.co/models)，搜索“google-t5”，可以看到，出现了“google-t5/t5-small”、“google-t5/t5-base”、“google-t5/t5-large”、“google-t5/t5-3b”和“google-t5/t5-7b”等不同大小的模型：

- base是基础模型，类似于bert-base,拥有12层和约220M参数。
- small是小模型，有6层和60M参数。
- large是大模型，类似于bert-large,有12层和770M参数。
- 3b和11b模型都使用24层的编码器和解码器,分别拥有约2.8B和11B参数。

## 初始化T5模型

首先安装必要的包：

```{.python .input}
# Hide outputs
!pip install transformers sentencepiece
```

导入接下来会用到的包

```{.python .input}
import torch
import json

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
```

初始化`google-t5/t5-large`预训练模型和分词器：

```{.python .input}
model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')
tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-large')
```

初始化一个预训练的分词器只需要一行代码。但是,这并不能证明分词词典包含了我们所需的全部词汇。我们将在:ref:`chapter-9`中探究分词器和数据集之间的关系。

在这里我们不使用GPU，CPU足够我们的demo运行了：

```{.python .input}
device = torch.device("cpu")
```

接下来我们探索T5模型的架构。

首先我们来看模型的配置：

```{.python .input}
print(model.config)
```

可以看到，“t5-large”模型有16个注意力头和24层。

在`"task_specific_params"`中，我们还可以看到T5的文本到文本实现,它在输入句子前添加前缀来触发要执行的任务。这个前缀使得在不修改模型参数的情况下,就可以用文本到文本的格式表示各种各样的任务。在我们的例子中,前缀是"summarization:".

我们可以看到T5:

- 实现了束搜索算法,它将扩展四个最重要的文本完成预测。
- 在每个批次中完成num_beam个句子时应用提前停止。
- 确保不重复等于no_repeat_ngram_size的n元语法。
- 使用min_length和max_length来控制样本的长度。
- 应用长度惩罚。

另一个比较有意思的超参数是`"vocab_size": 32128`，即词表大小为32128. 

词表本身就是一个值得探讨的话题。词表过大会导致表征的稀疏性，词表过小会导致许多词无法被表示。我们将在:ref:`chapter-9`中进一步探讨这个问题。

通过打印整个模型我们可以查看模型更多的细节：

```{.python .input}
print(model)
```

也可以只打印你希望的模块，以编码器的第13层为例：

```{.python .input}
print(model.encoder.block[12])
```

可以看到，自注意力子层以长度为1024的向量作为输入和输出，前馈子层的隐藏层产生的特征是长度为4096的向量。

## 使用T5-large对文档做摘要

前面我们已经初始化了T5模型`model`和T5分词器`tokenizer`，现在我们来对文档做摘要。

### 定义摘要函数

```{.python .input}
def summarize(text, ml):
    # 格式规范化
    preprocess_text = text.strip().replace("\n", "")
    # 加上"summarize: "前缀，告诉T5要做摘要任务
    t5_prepared_text = "summarize: " + preprocess_text
    print ("Preprocessed and prepared text: \n", t5_prepared_text)
    # 将文本转换为Tensor
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)
    # 利用前面定义的T5模型生成摘要
    summary_ids = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=ml,
        early_stopping=True
    )
    # 解码输出
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output
```

### 运行摘要任务

常见话题的摘要：

```{.python .input}
text = """
The United States Declaration of Independence was the first Etext released by Project Gutenberg, early in 1971. The title was stored in an emailed instruction set which required a tape or diskpack be hand mounted for retrieval. The diskpack was the size of a large cake in a cake carrier, cost $1500, and contained 5 megabytes, of which this file took 1-2%. Two tape backups were kept plus one on paper tape. The 10,000 files we hope to have online by the end of 2001 should take about 1-2% of a comparably priced drive in 2001.
"""
print("Number of characters:", len(text))
summary = summarize(text, 50)
print("\n\nSummarized text: \n" + summary)
```

权利法案的摘要：

```{.python .input}
text = """
No person shall be held to answer for a capital, or otherwise infamous crime, unless on a presentment or indictment of a Grand Jury, except in cases arising in the land or naval forces, or in the Militia, when in actual service in time of War or public danger; nor shall any person be subject for the same offense to be twice put in jeopardy of life or limb; nor shall be compelled in any criminal case to be a witness against himself, nor be deprived of life, liberty, or property, without due process of law; nor shall private property be taken for public use without just compensation.
"""
print("Number of characters:", len(text))
summary = summarize(text, 50)
print("\n\nSummarized text: \n" + summary)
```

企业法摘要：

```{.python .input}
text = """
The law regarding corporations prescribes that a corporation can be incorporated in the state of Montana to serve any lawful purpose. In the state of Montana, a corporation has all the powers of a natural person for carrying out its business activities. The corporation can sue and be sued in its corporate name. It has perpetual succession. The corporation can buy, sell or otherwise acquire an interest in a real or personal property. It can conduct business, carry on operations, and have offices and exercise the powers in a state, territory or district in possession of the U.S., or in a foreign country. It can appoint officers and agents of the corporation for various duties and fix their compensation. The name of a corporation must contain the word "corporation" or its abbreviation "corp." The name of a corporation should not be deceptively similar to the name of another corporation incorporated in the same state. It should not be deceptively identical to the fictitious name adopted by a foreign corporation having business transactions in the state. The corporation is formed by one or more natural persons by executing and filing articles of incorporation to the secretary of state of filing. The qualifications for directors are fixed either by articles of incorporation or bylaws. The names and addresses of the initial directors and purpose of incorporation should be set forth in the articles of incorporation. The articles of incorporation should contain the corporate name, the number of shares authorized to issue, a brief statement of the character of business carried out by the corporation, the names and addresses of the directors until successors are elected, and name and addresses of incorporators. The shareholders have the power to change the size of board of directors.
"""
print("Number of characters:", len(text))
summary = summarize(text, 50)
print("\n\nSummarized text: \n" + summary)
```
