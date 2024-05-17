# GPT-3的崛起
:label:`chapter-7`

2020年,[Brown等人(2020)](https://arxiv.org/pdf/2005.14165)描述了OpenAI GPT-3模型的训练过程,该模型包含1750亿个参数,采用了从Common Crawl数据中提取的4000亿字节对编码令牌这样的大型数据集进行学习。OpenAI在微软Azure超级计算机上进行了训练,拥有28.5万个CPU和1万个GPU。

OpenAI GPT-3引擎的机器智能以及它们的超级计算机,让[Brown等人(2020)](https://arxiv.org/pdf/2005.14165)进行了零样本实验。其思路是使用经过训练的模型进行下游任务,而无需进一步训练参数。目标是让训练好的模型**直接**投入到多任务生产中,提供一个API,甚至可以执行它未经训练的任务。

超人类云端AI引擎时代由此诞生。OpenAI的API不需要高级软件技能或AI知识。你可能会想知道我为什么使用"超人类"这个词。你会发现,在许多情况下,GPT-3引擎可以和人类一样出色地执行许多任务。当前最重要的是理解GPT模型是如何构建和运行的,才能欣赏它的魔力。

这一章将首先研究transformer模型的架构和规模的演进。我们将探讨使用经过训练的transformer模型进行下游任务的零样本挑战,并探索GPT transformer模型的创新架构。OpenAI提供了特别训练过的模型版本,称为引擎。

我们将使用OpenAI仓库中的一个3.45亿参数的GPT-2 transformer模型,基于TensorFlow框架。我们必须亲自动手,才能真正理解GPT模型。我们将与模型互动,生成由通用条件句进行条件化的文本补全。

接下来,我们将使用一个1.17亿参数的定制GPT-2模型。我们将对在第4章:ref:`chapter-4`中使用的Kant高级概念数据集进行标记。

本章还将探讨使用GPT-3引擎,它不需要数据科学家、人工智能专家,甚至是有经验的开发人员就可以开始使用。但这并不意味着后续就不需要数据科学家或AI专家了。

我们会发现,GPT-3引擎有时需要微调。我们将运行一个Google Colab笔记本来微调一个GPT-3 Ada引擎。

本章将以4.0时代AI专家的新思维模式和技能集作为结尾。

通过本章的学习,你将了解GPT模型是如何构建的,以及如何使用无缝的GPT-3 API。你将理解4.0时代的AI专家在2020年代可以完成的令人振奋的任务!

本章涵盖了以下主题:

- 从GPT-3模型开始
- OpenAI GPT模型的架构
- 定义零样本transformer模型
- 从少样本到单样本的路径
- 构建接近人类的GPT-2文本补全模型
- 实现并运行345M参数模型
- 与标准GPT-2模型进行交互
- 训练117M参数的GPT-2语言建模模型
- 导入定制和特定的数据集
- 编码定制数据集
- 对模型进行条件化
- 针对特定文本补全任务对GPT-2模型进行条件化
- Fine-tuning GPT-3模型
- 4.0时代AI专家的角色

```toc
:maxdepth: 2

section_01_suprahuman_nlp_with_gpt3-transformer-models
section_02_the_architecture_of_openai_gpt_transformer_models
section_03_generic_text_completion_with_gpt2
section_04_running_openai_gpt3_tasks
section_05_comparing_the_output_of_gpt2_and_gpt3
section_06_fine_tuning_gpt3
section_07_the_role_of_an_industry_40_ai_specialist
```
