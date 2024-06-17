# Transformer的性能 VS 人类基准

Transformers，就像人类一样，可以通过继承预训练模型的属性进行微调，以执行下游任务。预训练模型通过其参数提供其架构和语言表示。

预训练模型在关键任务上进行训练，以获得对语言的一般知识。微调模型则用于下游任务的训练。并非每个Transformer模型都使用相同的任务进行预训练。潜在地，可以对所有任务进行预训练或微调。

每个自然语言处理（NLP）模型都需要使用标准方法进行评估。

本节首先将介绍一些关键的度量指标。然后，我们将介绍一些主要的基准任务和数据集。

## 使用度量指标来评估模型

在没有使用度量标准的普遍测量系统时，无法比较一个Transformer模型与另一个Transformer模型（或任何其他NLP模型）。

在本节中，我们将分析GLUE和SuperGLUE使用的三种度量评分方法。

### 准确率（Accuracy）

准确率（accuracy）是一种实用的评估方法。对于给定测试集$\mathcal{D}$，其中第$i$个元素为$(\mathbf{x}^{(i)},y^{(i)})\in\mathcal{D}$，其中$\mathbf{x}^{(i)}$表示输入句子，$y^{(i)}$是句子的分类标签。设分类模型对$\mathbf{x}^{(i)}$预测的分类为$\hat{y}^{(i)}$，则该分类模型在测试集$\mathcal{D}$上的准确率计算公式为：

$$
\text{Accuracy}(\mathcal{D})=\frac{1}{|\mathcal{D}|}\sum_{i=1}^{|\mathcal{D}|}\mathbb{I}(y^{(i)}=\hat{y}^{(i)})
$$

### F1评分（F1-score）

在 :ref:`chapter-2-section-2-mcc` 中，我们介绍了真正、假正、真负和假负的概念，我们来回顾一下：

- $TP$表示True Positive（真正），即预测为正，实际标签为正
- $TN$表示True Negative（真负），即预测为负，实际标签为负
- $FP$表示False Positive（假正），即预测为正，实际标签为负
- $FN$表示False Negative（假负），即预测为负，实际标签为正

F1分数引入了一种更灵活的方法，可以应对数据集中存在不均匀的类别分布的情况。

F1分数考虑了精确率（precision）和召回率（recall）的，是精确率和召回率值的加权平均：

$$
\text{F1score}=\frac{2pr}{p+r}
$$

其中精确率$p$的计算公式为：

$$
p=\frac{TP}{TP+FP}
$$

召回率$r$的计算公式为：

$$
r=\frac{TP}{TP+FN}
$$

精确率和召回率计算公式的分子相同。精确率的分母是所有预测为正的样本数，而召回率的分母则是所有实际标签为正的样本数。因此，精确率也被称为查准率，即预测为正的样本中有多少是准的；召回率也被称为查全率，即实际为正的样本中，有多少被找出来了。

### 马修斯相关系数（MCC）

在 :ref:`chapter-2-section-2-mcc` 中我们介绍了马修斯相关系数（Matthews Correlation Coefficient, MCC），其公式为：

$$
MCC = \frac{TP\times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

MCC为二元分类模型提供了一个优秀的度量指标，即使类别的大小不同。

现在，我们对如何衡量给定的Transformer模型的结果并将其与其他Transformer模型或NLP模型进行比较有了一个很好的理解。

## 标杆（Benchmark）任务和数据集

要证明Transformers达到了最先进的（state-of-the-art, SOTA）性能水平，必须有以下三样东西：

- 模型
- 任务（数据集）
- 度量指标

我们将从探索SuperGLUE基准测试开始，以说明对Transformer模型进行评估的过程。

### 从GLUE到SuperGLUE

SuperGLUE基准测试是由[Wang等人（2019年）](https://arxiv.org/abs/1804.07461)设计并公开的。[Wang等人（2019年）](https://arxiv.org/abs/1804.07461)首先设计了通用语言理解评估（GLUE）基准测试。

GLUE基准测试的动机是要展示自然语言理解（NLU）必须适用于各种任务才能有用。相对较小的GLUE数据集旨在鼓励NLU模型解决一组任务。

然而，随着Transformer技术的出现，NLU模型的性能开始超过了普通人的水平，这一点可以在[GLUE排行榜](https://gluebenchmark.com/leaderboard)（2021年12月）中看到。

新模型和人类基准排名将不断变化。这些排名只是给我们一个关于经典自然语言处理和Transformer技术的发展程度的想法！

我们注意到GLUE人类基准没有处于前列的位置，这表明NLU模型已经在GLUE任务上超过了非专业人类。人类基准代表了我们人类所能达到的水平。如今，人工智能已经能够超越人类。在2021年12月，人类基准只排在第17位。这是一个问题。在没有一个标准可以超越的情况下，盲目寻找基准数据集来改进我们的模型是具有挑战性的。

我们还注意到，Transformer模型已经取得了领先地位。

随着自然语言理解的进展，GLUE排行榜将不断演变。然而，[Wang等人（2019年）](https://arxiv.org/abs/1905.00537)引入了SuperGLUE，为人类基准设定了更高的标准。

### 引入更高的人类基准标准

[Wang等人（2019年）](https://arxiv.org/abs/1905.00537)意识到了GLUE的局限性，并设计了用于更加困难的NLU问题的SuperGLUE.

然而，随着我们不断提升NLU模型的表现，SuperGLUE排行榜也在不断演变。在2021年，Transformer模型已经超过了人类基准。到了2021年12月，人类基准已经下降到第5名。

随着新的创新模型的出现，AI算法排名将不断变化。这些排名只是给我们一个关于争夺自然语言处理领域至高无上地位的艰巨程度的概念！

### SuperGLUE评分

[Wang等人（2019年）](https://arxiv.org/abs/1905.00537)为SuperGLUE基准选择了[10个任务](https://super.gluebenchmark.com/tasks)。这些任务的选择标准比GLUE更为严格。例如，这些任务不仅需要理解文本，还需要进行推理。推理的水平并不等同于顶级人类专家。然而，性能水平已经足以替代许多人类任务。

每个任务都包含了执行该任务所需的信息链接：

- Name是微调预训练模型对应的下游任务的名称
- Identifier是名称的缩写或简短版本
- Download提供了数据集的下载链接
- More Info通过链接到设计数据集驱动任务的团队的论文或网站，提供了更详细的信息
- Metric是用于评估模型的度量指标

SuperGLUE提供任务说明、软件、数据集以及描述要解决问题的论文或网站。某个模型在运行SuperGLUE中的benchmark任务后，将得到每个任务的分数和总体的分数。

例如，[Wang等人（2019年）](https://arxiv.org/abs/1905.00537)论文中的“选择合理答案任务”（Choice of Plausible Answers, COPA）可以表述为以下形式：

```
前提：I knocked on my neighbor's door.
问题：What happened as a result?
选择1：My neighbor invited me in.
选择2：My neighbor left his house.
```

模型需要给出正确的选择。

这个问题对于一个人来回答需要一两秒钟的时间，这表明它需要一些常识性的机器思考。COPA.zip是一个准备就绪的数据集，可以直接从SuperGLUE任务页面下载。提供的度量指标使得整个基准竞赛对所有参与者公平可靠。

## SuperGLUE中的Benchmark任务

一个任务可以作为预训练任务来生成训练好的模型。同一个任务也可以作为另一个模型的下游任务进行微调。然而，SuperGLUE的目标是展示给定的NLU模型可以通过微调执行多个下游任务。多任务模型证明了Transformer的思考能力。

任何Transformer模型的强大之处在于其能够使用预训练模型执行多个任务，并将其应用于微调的下游任务。原始的Transformer模型及其变种现在在所有的GLUE和SuperGLUE任务中处于领先地位。我们将继续专注于SuperGLUE的下游任务，其中人类基准难以超越。

在前面的部分，我们已经介绍了COPA任务。在本节中，我们将介绍[Wang等人（2019年）](https://arxiv.org/abs/1905.00537)在他们的论文中在表2中定义的其他七个任务。

### BoolQ

BoolQ是一个布尔类型的是或否回答任务。根据SuperGLUE的定义，该数据集包含15,942个自然问题。以下是一条示例数据：

```json
{
    "question": "is window movie maker part of windows essentials",
    "passage": "Windows Movie Maker -- Windows Movie Maker (formerly known as Windows Live Movie Maker in Windows 7) is a discontinued video editing software by Microsoft. It is a part of Windows Essentials software suite and offers the ability to create and edit videos as well as to publish them on OneDrive, Facebook, Vimeo, YouTube, and Flickr.",
    "idx": 2,
    "label": true
}
```

其中`"question"`表示问题，`"passage"`表示问题所参考的文章，`"label"`表示问题的答案（真/假）。

### Commitment Bank (CB)

Commitment Bank（CB）是一个困难的蕴涵（Entailment）任务。蕴含任务要求Transformer模型阅读一个前提（Premise），并检查基于该前提构建的假设（Hypothesis）。假设可能确认前提，也可能否认前提，还可能是中立的关系。这三种关系分别对应三个标签：蕴含（Entailment），冲突（Contradiction）和中立（Neutral）。

以下通过CB中的一条数据来举例说明：

```json
{
    "premise": "The Susweca. It means ''dragonfly'' in Sioux, you know. Did I ever tell you that's where Paul and I met?",
    "hypothesis": "Susweca is where she and Paul met,",
    "label": "entailment",
    "idx": 77
}
```

根据前提，这里的假设是成立的，即假设确认了前提，因此二者是蕴含关系，标签为`"entailment"`.

### Multi-Sentence Reading Comprehension (MultiRC)

多句子阅读理解（Multi-Sentence Reading Comprehension, MultiRC）要求模型阅读一段文本，并从多个可能的选项中进行选择。这个任务对人类和机器来说都很困难（想象一下你做阅读理解时的心情）。模型会被呈现一段文本，几个问题，并针对每个问题给出可能的答案，每个答案都带有一个0（假）或1（真）的标签。

我们来看MultiRC中的一条数据：

```json
{
    "text": "The rally took place on October 17, the shooting on February 29. Again, standard filmmaking techniques are interpreted as smooth distortion: \"Moore works by depriving you of context and guiding your mind to fill the vacuum -- with completely false ideas. It is brilliantly, if unethically, done.\" As noted above, the \"from my cold dead hands\" part is simply Moore's way to introduce Heston. Did anyone but Moore's critics view it as anything else? He certainly does not \"attribute it to a speech where it was not uttered\" and, as noted above, doing so twice would make no sense whatsoever if Moore was the mastermind deceiver that his critics claim he is. Concerning the Georgetown Hoya interview where Heston was asked about Rolland, you write: \"There is no indication that [Heston] recognized Kayla Rolland's case.\" This is naive to the extreme -- Heston would not be president of the NRA if he was not kept up to date on the most prominent cases of gun violence. Even if he did not respond to that part of the interview, he certainly knew about the case at that point. Regarding the NRA website excerpt about the case and the highlighting of the phrase \"48 hours after Kayla Rolland is pronounced dead\": This is one valid criticism, but far from the deliberate distortion you make it out to be; rather, it is an example for how the facts can sometimes be easy to miss with Moore's fast pace editing. The reason the sentence is highlighted is not to deceive the viewer into believing that Heston hurried to Flint to immediately hold a rally there (as will become quite obvious), but simply to highlight the first mention of the name \"Kayla Rolland\" in the text, which is in this paragraph. ",
    "questions": [
        {
            "question": "When was Kayla Rolland shot?",
            "answers": [
                {"text": "February 17", "idx": 168, "label": 0},
                {"text": "February 29", "idx": 169, "label": 1},
                {"text": "October 29", "idx": 170, "label": 0},
                {"text": "October 17", "idx": 171, "label": 0},
                {"text": "February 17", "idx": 172, "label": 0}
            ]
        },
        {
            "question": "Who was president of the NRA on February 29?",
            "answers": [
                {"text": "Charleton Heston", "idx": 173, "label": 1},
                {"text": "Moore", "idx": 174, "label": 0},
                {"text": "George Hoya", "idx": 175, "label": 0},
                {"text": "Rolland", "idx": 176, "label": 0},
                {"text": "Hoya", "idx": 177, "label": 0},
                {"text": "Kayla", "idx": 178, "label": 0}
            ]
        }
    ]
    
}
```

其中一段文本对应多个问题，每个问题下有多个答案，但是只有一个答案是正确的（标签为1），模型需要预测出那个答案是正确的。

### Reading Comprehension with Commonsense Reasoning Dataset (ReCoRD)

具备常识推理的阅读理解数据集（Reading Comprehension with Commonsense Reasoning Dataset, ReCoRD）是另一个具有挑战性的任务。该数据集包含来自70,000多篇新闻文章的120,000多个查询。Transformer模型必须使用常识推理来解决这个问题。

我们来看ReCoRD中的一条数据：

```json
{
    "passage": "The harrowing stories of women and children locked up for so-called 'moral crimes' in Afghanistan's notorious female prison have been revealed after cameras were allowed inside. Mariam has been in Badam Bagh prison for three months after she shot a man who just raped her at gunpoint and then turned the weapon on herself - but she has yet to been charged. Nuria has eight months left to serve of her sentence for trying to divorce her husband. She gave birth in prison to her son and they share a cell together. Scroll down for video Nuria was jailed for trying to divorce her husband. Her son is one of 62 children living at Badam Bagh prison @highlight Most of the 202 Badam Bagh inmates are jailed for so-called 'moral crimes' @highlight Crimes include leaving their husbands or refusing an arrange marriage @highlight 62 children live there and share cells with their mothers and five others",
    "query": "The baby she gave birth to is her husbands and he has even offered to have the courts set her free if she returns, but @placeholder has refused.",
    "entities": ["Badam Bagh", "Nuria", "Mariam", "Afghanistan"],
    "entity_spans": {
        "text": [ "Afghanistan", "Mariam", "Badam Bagh", "Nuria", "Nuria", "Badam Bagh", "Badam Bagh" ],
        "start": [ 86, 178, 197, 357, 535, 627, 672 ],
        "end": [ 97, 184, 207, 362, 540, 637, 682 ]
    },
    "answers": ["Nuria"]
}
```

这里给出了一篇文章（`"passage"`）、一个查询（`"query"`）和4个候选的实体（`"entities"`）。候选的实体用于替换查询中的`"@placeholder"`，模型需要预测使用哪些实体替换`"@placeholder"`是正确的。这里的`"start"`和`"end"`表示实体在句子中开始和结束的位置，便于在Transformer模型中特定的位置进行特征抽取。

### Recognizing Textual Entailment (RTE)

在文本蕴涵识别（Recognizing Textual Entailment, RTE）任务中，Transformer模型需要阅读前提（Premise）并检查一个假设（Hypothesis），并预测假设蕴涵状态的标签，与Commitment Bank（CB）类似。

数据示例如下：

```json
{
    "premise": "U.S. crude settled $1.32 lower at $42.83 a barrel.",
    "hypothesis": "Crude the light American lowered to the closing 1.32 dollars, to 42.83 dollars the barrel.",
    "label": "not_entailment"
}
```

这里的假设与前提是非蕴含的关系，因此标签为`"not_entailment"`.

### Words in Context (WiC)

在上下文中的词语（WiC）和随后的Winograd任务中，测试模型处理模棱两可的词语的能力。在WiC任务中，多任务Transformer模型需要分析两个句子，确定目标词语在这两个句子中是否具有相同的意思。

数据示例如下：

```json
{
    "word": "place",
    "sentence1": "Do you want to come over to my place later?",
    "sentence2": "A political system with no place for the less prominent groups.",
    "label": false,
    "start1": 31,
    "start2": 27,
    "end1": 36,
    "end2": 32
}
```

这里，两个句子中的`"place"`不是同一个意思，因此标签为`false`. 这里的`"start1"`和`"end1"`表示该单词在第1个句子中的开始和结束的位置，`"start2"`和`"end2"`表示该单词在第2个句子中的开始和结束的位置.

### The Winograd schema challenge (WSC)

Winograd模式任务以Terry Winograd的名字命名。该数据集构成了一个共指消解问题。具体而言，模型需要预测句子中的代词与某个名词是否有共指关系。

示例数据如下：

```json
{
    "text": "I poured water from the bottle into the cup until it was full.",
    "span1_index": 7,
    "span2_index": 10,
    "span1_text": "the cup",
    "span2_text": "it",
    "label": true
}
```

这里的名词是`"the cup"`，代词是`"it"`，且在句子中，这个`"it"`指的就是`"the cup"`，二者是共指关系，因此标签为`true`.
