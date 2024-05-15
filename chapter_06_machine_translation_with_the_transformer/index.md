# 基于Transformer的机器翻译
:label:`chapter-6`

人类可以掌握序列传导，将一个表示转移到另一个对象上。我们可以轻松地想象一个序列的心理表征。如果有人说“我花园里的花很美”，我们可以轻松地想象一个花园里有花的场景。尽管我们可能从未见过那个花园，但我们可以看到花园的图像。我们甚至可以想象到鸟儿的鸣叫和花的芬芳气味。

机器必须从头开始学习使用数值表示进行传导。循环神经网络或卷积神经网络的方法已经产生了一些有趣的结果，但还没有达到显著的BLEU翻译评估分数。翻译需要将语言A的表示转换为语言B。

Transformer模型的自注意力机制创新增强了机器智能的分析能力。在尝试将语言A的序列翻译成语言B之前，可以充分地表示语言A的序列。自注意力机制提供了机器所需的智能水平，以获得更好的BLEU分数。

经典的"Attention Is All You Need" Transformer模型在2017年的英德翻译和英法翻译任务中取得了最佳结果。自那时以来，其他Transformer模型对这些任务的得分进行了改进。

在本书的这一部分，我们已经涵盖了Transformer的关键方面：Transformer的架构，从头开始训练RoBERTa模型，微调BERT模型，评估微调的BERT模型，并通过一些Transformer示例探索下游任务。

在本章中，我们将通过三个附加主题来介绍机器翻译。首先，我们将定义什么是机器翻译。然后，我们将预处理一个机器翻译研讨会（WMT）数据集。最后，我们将看到如何实现机器翻译。

本章涵盖以下主题：

- 定义机器翻译
- 人类的传导和翻译
- 机器的传导和翻译
- 预处理WMT数据集
- 使用BLEU评估机器翻译
- 几何评估
- 平滑评估
- 介绍Google Translate的API
- 使用Trax初始化英德翻译问题

```toc
:maxdepth: 2

section_01_defining_machine_translation
section_02_preprocessing_a_wmt_dataset
section_03_evaluating_machine_translation_with_bleu
section_04_translations_with_trax
```
