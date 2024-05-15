# 使用Transformer进行下游NLP任务

预训练和微调一个Transformer模型需要大量的时间和精力，但当我们看到一个拥有数百万参数的Transformer模型在各种NLU任务中发挥作用时，这些努力都是值得的。

本章我们将从超越人类基准开始探索。人类基准代表了人类在NLU任务上的表现。人类在幼年时学习了转导，并迅速发展出归纳思维。我们人类能够直接通过感官来感知世界。而机器智能完全依赖于我们感知到的信息通过语言来理解我们的语言。

接下来，我们将看到如何衡量Transformer的性能。衡量自然语言处理（NLP）任务仍然是一种直接的方法，涉及基于真实和错误结果的准确性得分。这些结果是通过基准任务和数据集获得的。例如，SuperGLUE是一个很好的例子，展示了Google DeepMind、Facebook AI、纽约大学、华盛顿大学等多个机构共同努力制定了衡量NLP性能的高标准。

最后，我们将探索几个下游任务，例如标准情感树库（SST-2）、语言可接受性和Winograd模式等。

Transformer模型通过在设计良好的基准任务上超越其他模型，迅速将自然语言处理（NLP）推向了新的水平。不断涌现和演进的替代Transformer架构将继续出现。

本章涵盖了以下主题：

- 衡量Transformer性能与人类基准的方法
- 测量方法（准确性、F1得分和MCC）
- 基准任务和数据集
- SuperGLUE下游任务
- 语言可接受性与CoLA
- 情感分析与SST-2
- Winograd模式

```toc
:maxdepth: 2

section_01_transformer_performances_versus_human_baselines
section_02_running_downstream_tasks
section_03_exercise
```
