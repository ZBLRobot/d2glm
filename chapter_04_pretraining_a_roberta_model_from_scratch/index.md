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

```toc
:maxdepth: 2

section_01_pretraining_from_scratch
section_02_exercise
```
