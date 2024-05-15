# 运行下游任务
:label:`chapter-05-section-02`

下游任务是对预训练的Transformer模型进行微调，并继承其模型和参数的任务。

因此，下游任务是站在预训练模型运行微调任务的角度来说的。这意味着，根据模型的不同，如果某个任务没有被用于完全预训练模型，那么它就是下游任务。在本节中，我们将把所有任务都视为下游任务，因为我们没有对它们进行预训练。

模型会不断发展，数据库、基准方法、准确度评估方法和排行榜标准也会随之演变。但是，通过本章的下游任务所反映出的人类思维结构将保持不变。

现在详细介绍一些下游任务。

## CoLA

[语言可接受性语料库（Corpus of Linguistic Acceptability, CoLA）](https://nyu-mll.github.io/CoLA/)是GLUE任务中的一个任务，包含了数千个英语句子样本及其语法可接受性标签。

CoLA的目标是是评估NLP模型对句子的语言可接受性进行判断的语言能力，期望NLP模型能够相应地对句子进行分类。

这些句子被标记为语法正确或语法错误。如果句子不符合语法规范，则标记为0。如果句子符合语法规范，则标记为1。例如：

- 对于句子 'we yelled ourselves hoarse.'，分类标记为1。
- 对于句子 'we yelled ourselves.'，分类标记为0。

在:ref:`chapter-03-section-02`中我们叙述了在BERT预训练模型上运行CoLA下游任务的详细过程和代码，其大致流程如下：

- 加载下游任务（CoLA）数据集
- 加载预训练模型（BERT）
- 使用下游任务数据集微调预训练模型
- 使用MCC指标评估微调后模型的性能

## SST-2

Stanford Sentiment TreeBank (SST-2) 是一个电影评论的数据集。在本节中，我们将描述SST-2（二分类）任务。然而，SST-2不仅可以做二分类任务，还可以对情感从0（负面）到n（正面）进行多分类。

在本节中，我们将使用Hugging Face Transformer Pipeline模型运行从SST中提取的样本，以说明二元分类。

```{.python .input}
from transformers import pipeline

nlp = pipeline("sentiment-analysis")
review1 = "If you sometimes like to go to the movies to have fun , Wasabi is a good place to start."
print(nlp(review1), review1)
review2 = "Effective but too-tepid biopic."
print(nlp(review2), review2)
```

这里的`nlp = pipeline("sentiment-analysis")`初始化了一个已经进行过情感分类下游任务的模型，`nlp(review1)`表示对`review1`进行情感分类。

从分类输出结果中可以看出，`review1`被分类为积极情感，`review2`被分类为消极情感，且分数都接近1.0，表明置信度很高。

## MRPC

Microsoft Research Paraphrase Corpus (MRPC)是GLUE任务中的一个任务，它包含从网络新闻来源中提取的句子对。每个句子对都由人工标注两个句子是否等价。等价的判断基于以下属性：

- 释义等价（Paraphrase equivalent）
- 语义等价（Semantic equivalent）

我们尝试运行一下MRPC任务的一个示例：

```{.python .input}
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased-finetuned-mrpc")
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased-finetuned-mrpc")
classes = ["not paraphrase", "is paraphrase"]
sequence_A = "The DVD-CCA then appealed to the state Supreme Court."
sequence_B = "The DVD CCA appealed that decision to the U.S. Supreme Court."
paraphrase = tokenizer.encode_plus(sequence_A, sequence_B, return_tensors="tf")
paraphrase_classification_logits = model(paraphrase)[0]
paraphrase_results = tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
print(sequence_B, "should be a paraphrase")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")
```

上述代码输出了`sequence_A`和`sequence_B`不等价和等价的概率。
