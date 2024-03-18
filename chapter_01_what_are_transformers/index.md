# 什么是Transformer?

Transformer是一种工业化、同质化的后深度学习模型（Post-Deep Learning Model），专为超级计算机上的并行计算而设计。通过将各种任务进行同质化（Homogenization），Transformer模型可以执行多种任务，而无需在特定任务上进行微调训练（Fine-tuning）。Transformer可以利用十亿级的参数对十亿级的原始未标记数据记录执行自监督学习（Self-supervised Learning）。

后深度学习中涉及的这些特定模型架构被统称为基座模型（Foundation Models），其中transformer基座模型更是代表着第四次工业革命的缩影。2015年，随着可实现万物互联的机器对机器自动化（Machine-to-machine Automation）的兴起，第四次工业革命开始。人工智能（AI），特别是针对工业4.0的自然语言处理（NLP），已经远远超出了过去的软件实践。

过了不到五年的时间，AI已经演变为具备无缝API的高效云服务。而在以往的范式中，人们需要通过下载软件库和本地开发来使用AI. 现在这种范式在大多数情况下只被用于教育性质的练习中。

工业4.0时代的项目经理可以访问OpenAI的云平台、注册账号、获取API密钥，然后在几分钟内开始工作。用户可以在这里输入文本，指定自然语言处理任务，并获得由GPT-3变换器引擎发送的回复。最后，用户可以进入GPT-3 Codex，创建应用程序，而无需了解编程知识。提示工程（Prompt Engineering）是从这些模型中崛起的一项新技能。

然而，有时候GPT-3模型可能并不适用于特定任务。例如，项目经理、顾问或开发者可能希望使用由Google AI、亚马逊网络服务（AWS）、Allen人工智能研究所或Hugging Face提供的其他系统。

这时候项目经理应该选择在本地开展工作？还是应该直接在Google Cloud、Microsoft Azure或AWS上开展工作呢？开发团队是否应选择Hugging Face、Google Trax、OpenAI或AllenNLP等提供的软件开发库？人工智能专家或数据科学家是否应该使用几乎不需要AI开发的API呢？

最佳答案是同时考虑以上所有选择。因为我们不知道未来的雇主、客户或用户可能想要或指定什么。因此，我们必须准备好适应任何出现的需求。本书并未描述市场上存在的所有选择。然而，本书为读者提供了足够的解决方案，以应对工业4.0人工智能驱动的自然语言处理挑战。

本章首先从抽象和高层的视角解释了什么是transformer. 然后解释了对实现transformer的各种方法获得灵活理解的重要性。由于市场上可用的API和自动化解决方案数量众多，对于平台、框架、库和语言的定义存在模糊和分歧。最后，本章介绍了在嵌入式transformer技术进展下的工业4.0人工智能专家的角色。在我们开始探索本书中描述的各种transformer模型实现之前，我们需要解决这些关键概念。

本章涵盖以下主题：
- 第四次工业革命，工业4.0的出现
- 基础模型的范式变革
- 引入新技能：提示工程
- transformer的背景知识
- 实施transformer的挑战
- 颠覆性的transformer模型API
- 选择transformer库的困难
- 选择transformer模型的困难
- 工业4.0人工智能专家的新角色
- 嵌入式transformers

我们的第一步将是探索transformer的生态系统。

```toc
section_01_the_ecosystem_of_transformers
```
