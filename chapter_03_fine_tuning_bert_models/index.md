# 微调BERT模型
:label:`chapter-3`

在基础Transformer提出之后，基于Transformer架构的多种变体模型接踵而至。

本章探索其中最著名的变体模型之一，BERT（Bidirectional Encoder Representations from Transformers）. BERT的创新之处在于只用了Transformer的编码器部分，没有使用解码器部分。

然后，我们将对预训练的BERT模型进行微调（Fine-tuning）。我们将微调的BERT模型是由第三方训练并上传到Hugging Face的。Transformer及其变体模型可以进行预训练（Pre-training），然后可以对预训练的模型（Pre-trained Models）进行微调，以适用于多个自然语言处理任务。这种训练范式被称为“预训练+微调”，微调所解决的任务称为下游任务（Downstream Tasks）。本章我们将使用Hugging Face模块来体验将Transformer预训练模型用于下游任务的过程。

本章涵盖以下主题：

- BERT
- BERT的架构
- BERT两步训练框架
- 准备预训练环境
- 定义编码器的预训练过程
- 定义微调过程
- 下游多任务处理
- 构建微调的BERT模型
- 加载数据集进行可接受性判断
- 创建注意力掩码
- BERT模型配置
- 评估微调模型的性能

我们首先来看BERT模型的背景。

```toc
:maxdepth: 2
:numbered:

section_01_the_architecture_of_bert

section_02_fine_tuning_bert

section_03_exercise
```
