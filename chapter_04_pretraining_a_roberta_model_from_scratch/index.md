# 从头开始预训练RoBERTa模型
:label:`chapter-4`

在本章中，我们将从头开始构建一个RoBERTa模型。该模型将使用构建BERT模型所需的transformer构建块。此外，我们不会使用预训练的分词器或模型。RoBERTa模型将按照本章描述的十五个步骤进行构建。

我们将利用前几章中学到的transformer知识，逐步构建一个可以进行掩码语言建模的模型。在第二章:ref:`chapter-2`中，我们介绍了原始Transformer的构建块。在第三章:ref:`chapter-3`中，我们微调了一个预训练的BERT模型。本章将重点介绍如何使用基于Hugging Face的无缝模块在Jupyter Notebook中从头开始构建一个预训练的transformer模型。该模型名为KantaiBERT。

我们首先介绍如何获取KantaiBERT预训练所需要的语料库——伊曼努尔·康德（Immanuel Kant）的图书汇编，并使用Jupyter Notebook创建自己的数据集，

KantaiBERT会从头开始训练自己的分词器。分词器将构建合并文件（Merge File）和词表文件（Vocabulary File），并在预训练过程中使用它们。

然后我们处理数据集，初始化训练器并训练KantaiBERT模型。

最后，KantaiBERT使用训练好的模型执行一个实验性的下游语言建模任务，并使用伊曼努尔·康德的逻辑填充一个掩码。

本章包含以下内容：

- RoBERTa和DistilBERT类的模型
- 如何从头开始训练分词器
- 字节级别的字节对编码（BPE）
- 将训练好的分词器保存为文件
- 为预训练过程重新创建分词器
- 从头初始化一个RoBERTa模型
- 探索模型的配置
- 探索模型的8000万个参数
- 为训练器创建数据集
- 初始化训练器
- 预训练模型
- 保存模型
- 将模型应用到下游的MLM任务

在本章中，我们将使用Hugging Face为类似BERT的模型提供的构建块来训练一个名为KantaiBERT的transformer模型。我们在第三章:ref:`chapter-3`中介绍了我们将使用的模型的构建块的理论知识。

KantaiBERT是一种基于BERT架构的，类似于RoBERTa（Robustly Optimized BERT Pretraining Approach）的模型。

原始的BERT模型为原始的transformer模型带来了创新的特性，而RoBERTa通过改进预训练过程的机制，进一步提高了transformer在下游任务中的性能。

例如，RoBERTa不使用WordPiece分词，而是采用字节级别的字节对编码（Byte-Pair Encoding，BPE）。这种方法为各种BERT和类BERT模型铺平了道路。

在本章中，KantaiBERT和BERT一样，将使用Masked Language Modeling（MLM）进行训练。MLM是一种语言建模技术，它会在序列中掩盖一个单词，然后transformer模型必须训练以预测被掩盖的单词。

KantaiBERT将作为一个小型模型进行训练，它由6层、12个注意力头和84,095,008个参数组成。这个参数数量似乎很多，但是这些参数分布在12个注意力头上，使得它成为一个相对较小的模型。小型模型将使预训练过程更加平滑，每个步骤都可以实时查看结果，而无需等待几个小时。

KantaiBERT是类似DistilBERT的模型，因为它具有相同的6层和12个注意力头的架构。DistilBERT意思是进行知识蒸馏（Knowledge Distillation）后的BERT模型，其参数量比原始BERT模型少。类似地，KantaiBERT也是蒸馏后的RoBERTa模型，其参数数量比原始RoBERTa模型少。因此，它的运行速度更快，但结果略微比RoBERTa模型的精确度较低。

## 第1步：加载数据集

通过使用预先准备好的数据集，我们可以客观地训练和比较transformer模型。

这里选择使用伊曼努尔·康德（1724-1804）的作品，他是启蒙时代的典范，也是一位德国哲学家。这个想法是为了在下游推理任务中引入类人的逻辑和预训练的推理能力。

[Gutenberg](https://www.gutenberg.org)中提供了众多免费的电子书，可以下载为文本格式。你也可以使用其他书作为自定义的数据集。

这里收集了伊曼努尔·康德的以下三本书中的语料作为训练数据，并将其命名为`kant.txt`:

- 纯粹理性批判（The Critique of Pure Reason）
- 实践理性批判（The Critique of Practical Reason）
- 道德形而上学基础（Fundamental Principles of the Metaphysic of Morals）

`kant.txt`提供了一个小型训练数据集，用于训练本章的transformer模型。所得到的结果仍然是实验性的。对于一个真实需要落地的项目，可能需要添加更多的语料。

可以使用以下命令下载`kant.txt`:

```{.python .input}
!curl -L https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-2nd-Edition/master/Chapter04/kant.txt --output "kant.txt"
```

## 第2步：安装Hugging Face transformers

```{.python .input}
!pip install transformers
```

## 第3步：训练分词器（tokenizer）

在本节中，我们不使用预训练的分词器（如GPT-2的预训练分词器），而是从头开始训练一个分词器。

我们将使用`kant.txt`来训练Hugging Face的`ByteLevelBPETokenizer`. 一个BPE（Byte-Pair Encoding）分词器会将一个字符串或单词分解为子字符串或子词。除了其他许多优点外，这种方法有两个主要优势：

- 分词器可以将单词分解为最小的组成部分，然后将这些小组件合并为统计上有趣的组合。例如，“smaller”和“smallest”可以变为“small”，“er”和“est”。分词器还可以进一步拆分，例如，我们可以得到“sm”和“all”。无论如何，单词都被拆分为子词标记和更小的子词部分，例如“sm”和“all”，而不是简单的“small”。
- 使用WordPiece级别的编码，被分类为未知（unk_token）的字符串块将基本上消失。

在这个模型中，我们将使用以下参数来训练分词器：

- `files=/path/to/dataset`是数据集的路径
- `vocab_size=52000`是词表大小
- `min_frequency=2`是最小频率的阈值
- `special_tokens=[]`是特殊token的列表

这里，特殊token包括：

- `<s>`: 开始token
- `<pad>`: 填充token
- `</s>`: 结束token
- `<unk>`: 未知token
- `<mask>`: 用于MLM的掩码token

我们来看某个句子中的两个单词：

```
...the tokenizer...
```

分词器的第一步是将字符串变为tokens:

```
'Ġthe', 'Ġtoken', 'izer',
```

其中`Ġ`表示空格信息。

下一步是使用token的索引来代替token:

|'Ġthe'|'Ġtoken'|'izer'|
|:--:|:--:|:--:|
|150|5430|4712|

整个流程代码如下：

```{.python .input}
%%time
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
paths = [str(x) for x in Path(".").glob("**/*.txt")]
# 初始化一个分词器
tokenizer = ByteLevelBPETokenizer()
# 自定义训练
tokenizer.train(
    files=paths,
    vocab_size=52000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)
```

可以看到，上述代码输出了训练运行的时间。现在，tokenizer被训练好了，我们准备把tokenizer保存下来。

## 第4步：将分词器保存到磁盘

分词器被训练后会产生两个文件：

- `merges.txt`: 包含了合并的标记化子字符串（tokenized substrings）。
- `vocab.json`: 包含了标记化子串的索引

以下代码将上述代码所训练的tokenizer对应的两个文件保存到`KantaiBERT`目录下：

```{.python .input}
import os
os.makedirs("KantaiBERT", exist_ok=True)
tokenizer.save_model("KantaiBERT")
```

在这个例子中，文件的大小较小。您可以双击打开这些文件来查看它们的内容。可以看到`merges.txt`包含了标记化的子字符串：

```
#version: 0.2 - Trained by 'huggingface/tokenizers'
Ġ t
h e
Ġ a
o n
i n
Ġ o
Ġt he
r e
i t
Ġo f
```

`vocab.json`包含了token的索引：

```
[…,"Ġthink":955,"preme":956,"ĠE":957,"Ġout":958,"Ġdut":959, "aly":960,"Ġexp":961,…]
```

## 第5步：加载训练的分词器

我们本来可以直接加载Hugging Face上预训练的分词器，但是现在我们可以使用我们自己训练的分词器了：

```{.python .input}
from tokenizers.implementations import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer(
"./KantaiBERT/vocab.json",
"./KantaiBERT/merges.txt",
)
```

tokenizer可以对一个序列进行编码：

```{.python .input}
encoding = tokenizer.encode("The Critique of Pure Reason.")
```

可以看到，`"The Critique of Pure Reason."`这句话变成了如下token序列：

```{.python .input}
encoding
```

为了将序列用于BERT模型及其变种的训练，我们需要让tokenizer对传入的序列进行后处理，加上开始token和结束token:

```{.python .input}
from tokenizers.processors import BertProcessing
# 后处理
tokenizer._tokenizer.post_process = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
# 截断处理
tokenizer.enable_truncation(max_length=512)
```

这样一来，再次调用tokenizer进行分词时，会得到包含特殊token的token序列：

```{.python .input}
tokenizer.encode("The Critique of Pure Reason.").tokens
```

## 第6步：检查资源：GPU和CUDA

使用以下命令检查GPU:

```
!nvidia-smi
```

以上命令会输出GPU的型号和显存等信息。

使用以下python代码检查cuda是否可用：

```{.python .input}
import torch
torch.cuda.is_available()
```

输出为`True`说明cuda可用。

这里建议使用[Google Colab](https://colab.research.google.com/)或[Kaggle Notebook](https://www.kaggle.com/code)，这样可以确保GPU和cuda是可用的。

如果你使用自己的笔记本或者自己搭建的服务器上的GPU，请先安装CUDA和cuDNN（Google/问GPT）.

## 第7步：定义模型的配置

我们将使用与DistilBERT transformer相同数量的层数和注意力头数，对一个RoBERTa类型的transformer模型进行预训练。该模型的词汇表大小将设置为52,000，具有12个注意力头和6个层：

```{.python .input}
from transformers import RobertaConfig
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
```

我们会在后续步骤中详细探索这些配置。

## 第8步：在transformers中重新加载tokenizer

我们现在准备好加载我们训练好的分词器，即使用RobertaTokenizer.from_pretained()加载我们的预训练分词器：

```{.python .input}
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("./KantaiBERT", max_length=512)
```

## 第9步：从头初始化模型

首先导入用于掩码语言建模（MLM）的RoBERTa模型：

```{.python .input}
from transformers import RobertaForMaskedLM
```

使用上面创建的配置来初始化模型：

```{.python .input}
model = RobertaForMaskedLM(config=config)
```

打印模型可以看到，它是一个有6层和12个注意力头的BERT模型：

```{.python .input}
print(model)
```

在继续之前，请花些时间仔细研究上述模型输出的细节。这样您将能够从内部了解模型。

### 探索模型参数

上面的模型算规模较小的，共有84,095,008个参数，可以用代码查看其参数量：

```{.python .input}
print(model.num_parameters())
```

代码打印的参数量只是一个大致的数，不同transformers版本之间可能有细微的差别。

现在让我们来看看参数。我们首先将参数存储在`LP`中，并计算参数列表的长度：

```{.python .input}
LP=list(model.parameters())
lp=len(LP)
print(lp)
```

输出显示，模型的参数中大约有108个矩阵和向量。这一数字在不同transformers版本之间也可能有不同。

可以打印每个矩阵和向量来查看参数的详情：

```{.python }
for p in range(0, lp):
    print(LP[p])
```

模型总参数量的计算方法是，把里面每一个向量和矩阵所包含的元素的数量加起来。

## 第10步：构建数据集

现在我们将逐行加载数据集，以生成用于批量训练的样本：

```{.python .input}
%%time
from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
tokenizer=tokenizer,
file_path="./kant.txt",
block_size=128,
)
```

其中`block_size = 128`限制了一条数据的长度。输出显示，Hugging Face在优化数据处理时间方面投入了相当多的资源。墙时（wall time），即处理器实际活动的时间，已经得到了优化。

## 第11步：定义数据整合器

在初始化训练器之前，我们需要运行一个数据整合器（data collator）。数据整合器会从数据集中获取样本并将它们整合成批次。其结果是类似字典的对象。

通过设置数据整合器的参数`mlm=True`，我们可以将数据处理为MLM训练所需的形式，整合器将会在文本中加入掩码token `<mask>`.

我们还设置了训练MLM的掩码标记数量，即`mlm_probability=0.15`，表示每个token有15%的概率被掩蔽掉。

现在我们使用我们的分词器初始化数据整合器，激活MLM，并将掩码标记的比例设置为0.15：

```{.python .input}
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
```

## 第12步：初始化训练器

前面的步骤已经准备好了初始化训练器所需的信息：数据集、分词器、模型以及数据整合器。

现在我们可以初始化训练器了。由于只是为了演示，我们将对模型进行快速训练，训练的轮数被限制为一轮。由于我们可以共享批次并进行多进程训练任务，所以GPU非常有用：

```{.python .input}
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
```

现在模型准备好进行训练了。

## 第13步：预训练模型

```{.python }
%%time
trainer.train()
```

输出实时显示训练过程，包括损失（loss）、学习率（learning rate）、轮数（epoch）和训练步数（steps）：

## 第14步：将最终的模型（包括分词器和配置）保存到磁盘上

```{.python }
trainer.save("./KantaiBERT")
```

可以在文件系统中看到，`KantaiBERT`目录下多出来了`config.json`, `pytorch_model.bin`和`training_args.bin`文件，以及与分词器相关的`merges.txt`和`vocab.json`文件。

至此，我们就从头开始构建了一个预训练模型，接下来我们构建一个流水线（pipeline）来使用这个模型。

## 第15步：使用`FillMaskPipeline`进行语言建模

现在我们将进行填充掩码（fill-mask）任务，我们将使用我们训练过的模型和训练过的分词器来执行MLM：

```{.python }
from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model="./KantaiBERT",
    tokenizer="./KantaiBERT"
)
```

现在我们可以让模型像伊曼努尔·康德一样思考：

```{.python }
fill_mask("Human thinking involves human <mask>.")
```

模型会根据从数据集中学到的知识来确定`<mask>`应该是什么词.

## 后续步骤

你已经从头开始训练了一个Transformer模型。花些时间想象一下，在个人或企业环境中你可以做什么。你可以为特定任务创建一个数据集，并从头开始训练它。利用你的兴趣领域或公司项目，来探索这个迷人的Transformer构建工具的世界！

一旦你训练出一个你喜欢的模型，你可以与Hugging Face社区分享它。你的模型将出现在[Hugging Face模型页面](https://huggingface.co/models)

你可以按照[这个页面上的说明](https://huggingface.co/transformers/model_sharing.html)，通过几个步骤上传你的模型。

你也可以下载Hugging Face社区分享的模型，以获取关于你个人和专业项目的新想法。
