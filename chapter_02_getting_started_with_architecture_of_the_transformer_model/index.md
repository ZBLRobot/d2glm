# 从Transformer的架构开始
:label:`chapter-2`

语言是人类交流的精髓。如果没有构成语言的词汇序列，文明将永远无法诞生。如今，我们大多生活在数字化语言表达的世界中。我们的日常生活依赖于自然语言处理（NLP）数字化语言功能，如网络搜索引擎、电子邮件、社交网络、帖子、推文、智能手机短信、翻译、网页、流媒体网站上的语音转文字、热线服务上的文字转语音等等，这些都是我们日常生活中常见的功能。

在“:ref:`chapter-1`”中解释了循环神经网络（RNN）的局限性以及云端AI Transformer在设计和开发中所占据的重要地位。工业4.0开发人员的角色是理解原始Transformer的架构以及随后出现的多个Transformer生态系统。

在2017年12月，Google Brain和Google Research发表了具有开创性意义的Vaswani等人的论文*Attention is All You Need*。Transformer应运而生。Transformer在性能上超越了现有的最先进自然语言处理（NLP）模型。相比之前的架构，Transformer的训练速度更快，并获得了更高的评估结果。因此，Transformer已成为NLP的关键组成部分。

Transformer模型中的注意力机制的理念是摒弃循环神经网络的特征。在本章中，我们将揭开Vaswani等人（2017年）所描述的Transformer模型的内部机制，并研究其架构的主要组成部分。我们将探索引人入胜的注意力世界，并说明Transformer的关键组件。

本章涵盖以下主题：

- Transformer的架构
- Transformer的自注意力模型
- 编码和解码堆栈
- 输入和输出嵌入
- 位置嵌入
- 自注意力
- 多头注意力
- 掩码多头注意力
- 残差连接
- 归一化
- 前馈网络
- 输出概率

让我们直接深入原始Transformer架构的结构。

```toc
:maxdepth: 2
:numbered:

section_01_the_rise_of_the_transformer_attention_is_all_you_need

section_02_training_and_performance

section_03_transformer_models_in_hugging_face

section_04_exercise
```
