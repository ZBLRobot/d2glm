# Transformer的兴起: Attention is All You Need

在*Attention is All You Need*这篇论文中，原始的Transformer模型由6层的堆栈组成，其中第$l$层的输出是第$l+1$层的输入，直到最后一层。如图 :numref:`fig-1` 所示，左边是一个6层的编码器堆栈，右边是一个6层的解码器堆栈。

![Transformer原始模型的架构](screenshots/transformers.svg)
:label:`fig-1`

图中左边，输入数据在Transformer的编码器部分经过了一个注意力子层（Multi-Head Attention）和一个前馈子层（Feedforward）。图中右边，目标输出数据在Transformer的解码器部分经过了两个注意力子层（Masked Multi-Head Attention, Multi-Head Attention）和一个前馈网络子层（Feedforward）。我们注意到，Transformer架构中没有使用RNN、LSTM和CNN. 递归机制不复存在。

在RNN中，当两个单词的距离增加时，递归次数也会增加，从而参与递归运算的递归函数的参数也会增加。Transformer中注意力机制取代了递归函数。注意力机制是一种"词对词"的操作，更严格地说是一种标记对标记（Token-to-Token）的操作。在有的分词器中，一个单词会被拆成多个标记，而有的分词器中，多个单词会被聚合为一个标记。为了表述的简单，我们大多数时候不区分单词和标记的概念。注意力机制将找出每个单词与序列中的所有其他单词的关系，包括正在分析的单词本身。让我们来看下面的序列：

```
The cat sat on the mat.
```

注意力机制会在两个词向量之间执行点积运算，并且确定一个单词与其他所有单词的关系中最强的关系，包含与自身的关系（“cat”到“cat”）。图 :numref:`fig-2` 可视化了注意力机制作用在一个句子上的过程。

![单词“cat”对其他所有单词的注意力](screenshots/2024-03-20-11-11-25.png)
:label:`fig-2`

注意力机制将提供单词之间更深入的关系，并产生更好的结果。

对于每个注意力子层，原始的Transformer模型并行地运行了8组注意力机制，以加快计算速度。接下来我们将探索编码器堆栈和多头注意力（Multi-Head Attention）。多头注意力机制的优势如下：

- 对序列进行更广泛的深入分析
- 减少计算操作，避免了递归的需求
- 实现并行化，减少训练时间
- 每个注意力机制学习相同输入序列的不同视角

至此，我们已经浅显地从外部的视角了解了Transformer. 接下来我们来看一下Transformer编码器的内部结构。

## 编码器堆栈

原始Transformer模型的编码器和解码器的层都是一堆叠的层。编码器堆栈的每一层具有如图 :numref:`fig-3` 所示的结构。图中分别展示了第1层、中间层和第$N$层的情况，因为它们的两端连接着不同的其他模块。

![Transformer编码器的每一层](screenshots/transformer-encoder-layer.svg)
:label:`fig-3`

原始Transformer模型中，所有编码器层的结构是相同的。每一层包含两个主要的子层：多头注意力机制（Multi-Head Attention）和全连接的逐位置前馈网络（Fully Connected Position-wise Feedforward Network）。

请注意，Transformer模型中的每个主要子层$\text{sublayer}(x)$周围都有一个残差连接（Residual Connection）。残差连接将未经子层处理的输入$x$传递到层归一化（Layer Normalization）函数中。这样，我们可以确保关键信息（如位置编码）在传递过程中不会由于子层的处理而丢失。因此，每一个子层的归一化输出可以表示为：

$$
\text{LayerNorm}(x+\text{sublayer}(x))
$$

尽管编码器的每个N=6层的结构相同，但每一层的内容与前一层并不一定相同。例如，只有第1层才与嵌入层（Input Embedding）相连。其他五层不与嵌入层相连，这确保了通过所有层的编码输入是稳定的。

此外，从第1层到第6层，多头注意力机制执行的运算是相同的。然而，它们并不执行相同的任务。每一层都从前一层的输出中学习，并探索不同的方式来将序列中的tokens相关联。就像我们在玩填字游戏的时候，会寻找字母和单词的不同种类的关联。

Transformer的设计者引入了一个约束，方便层以及子层之间进行堆叠。模型的每个子层的输出都具有固定的维度，包括嵌入层和残差连接。我们用$d_\text{model}$来表示这个维度，根据不同的模型大小和用户需求可以设置不同的值。在原始的Transformer架构中，$d_\text{model}=512$.

$d_\text{model}$具有强大的影响。在Transformer中几乎所有关键的运算都是点积运算。因此，需要一个稳定的维度来减少所需的运算数量，降低机器的资源消耗，并且我们更容易追踪随运算而在模型中流动的信息。

上述Transformer编码器的整体视图展示了Transformer的高度优化架构。接下来我们将深入研究编码其中的每个子层及其机制。

### 输入嵌入层（Input Embedding Layer）

在原始Transformer模型中，输入嵌入层（Input Embedding Layer）使用学习到的嵌入（Embeddings）将输入的tokens转换为维度为$d_\text{model}=512$的向量。

![输入嵌入层](screenshots/input-embedding.svg)

在使用输入嵌入层之前，首先需要使用分词器将一个句子转换为tokens。每种分词器都有自己的算法，例如BPE、Word Piece和Sentence Piece等。Transformer最初使用的是BPE，但后续基于Transformer的模型可能使用了其他分词算法。

我们通过一个例子来看分词器是如何工作的。给定一个句子“The cat slept on the couch.It was too tired to get up.”，某个分词器可能将其分成如下形式：

```python
['the', 'cat', 'slept', 'on', 'the', 'couch', '.', 'It', 'was', 'too', 'tired', 'to', 'get', 'up', '.']
```

注意到，这个分词器将字符串转为小写，并分成一个个的token，并且不是严格按照空格来分的。另外，分词器通常还能将token转为整数表述，以便于某些嵌入层的输入：

```python
[1996, 4937, 7771, 2006, 1996, 6411, 1012, 2009, 2001, 2205, 5458, 2000, 2131, 2039, 1012]
```

嵌入层的作用是将离散的token序列映射到连续的向量空间中。

我们借用2013年Google提出的word2vec嵌入方法中的[skip-gram架构](https://zhuanlan.zhihu.com/p/29305464)来说明Transformer的嵌入层。skip-gram在训练时会关注一个窗口中的中心词，并预测其上下文的单词。例如，如果`word[i]`是一个窗口大小为2的窗口中的中心词，skip-gram模型将利用中心词来预测`word[i-2]`、`word[i-1]`、`word[i+1]`和`word[i+2]`。然后窗口会滑动并重复这个过程。在完成训练后，一个句子中拥有类似上下文的单词会拥有类似的词嵌入。

假设我们需要对以下句子进行嵌入：

```{.python .input}
# Hide outputs
text = "The black cat sat on the couch and the brown dog slept on the rug."
```

我们将关注“black”和“brown”两个词，因为这两个词都表示颜色，且是相近的颜色，因此它们的嵌入向量应该是相似的。

对于每一个词，我们必须产生一个长度等于$d_\text{model}=512$的嵌入向量，以满足Transformer模型的维度约束。

我们首先对句子进行分词：

```{.python .input}
# Hide outputs
# 安装NLP工具包NLTK
!pip install nltk
```

```{.python .input}
# Hide outputs
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
```

```{.python .input}
tokens
```

然后我们利用word2vec对分出来的token序列进行词嵌入。

```{.python .input}
# Hide outputs
# 安装gensim以使用word2vec模型
!pip install gensim
```

```{.python .input}
# Hide outputs
from gensim.models import word2vec

# 利用tokens来初始化word2vec模型
model = word2vec.Word2Vec([tokens], vector_size=512, window=2, min_count=1)
```

在完成嵌入后，我们可以看到，“black”的嵌入向量如下：

```{.python .input}
model.wv["black"]
```

“brown”的嵌入向量如下：

```{.python .input}
model.wv["brown"]
```

tokens序列中所有单词的嵌入向量依次拼接起来可以得到整个句子的嵌入矩阵：

```{.python .input}
# Hide outputs
import numpy as np


all_vectors = [model.wv[token] for token in tokens]
embedding_matrix = np.stack(all_vectors)
```

```{.python .input}
embedding_matrix
```


我们计算“black”与“brown”的相似度，发现得到的相似度并不高：

```{.python .input}
model.wv.similarity("black", "brown")
```

这是因为上面使用的word2vec模型只在一个句子上进行了词嵌入向量的训练，训练数据太少。接下来，我们使用Text8语料库来对word2vec进行训练。

```{.python .input}
# Hide outputs
# 下载并解压text8语料库
!wget http://mattmahoney.net/dc/text8.zip
!unzip text8.zip
```

```{.python .input}
# Hide outputs
import os

if not os.path.exists("word2vec.model"):
    # 加载text8语料库
    corpus = word2vec.Text8Corpus("text8")

    # 使用text8语料库初始化word2vec
    model = word2vec.Word2Vec(corpus, vector_size=512, window=2, min_count=1)

    # 保存训练后的模型
    model.save("word2vec.model")

else:
    # 直接加载上次训练好的模型
    model = word2vec.Word2Vec.load("word2vec.model")
```

在使用Text8语料库训练后，我们再来看看“black”和“brown”的相似度，发现已经提高了很多：

```{.python .input}
model.wv.similarity("black", "brown")
```

Transformer中嵌入层以后的层可以利用已经学习好的词嵌入向量。这些词嵌入向量为后续的注意力机制提供了重要的信息，这些信息将高速注意力机制如何将词语之间进行关联。

然而，有一种信息在Transformer架构中被丢失了，即词的位置信息。在递归神经网络中，token序列的输入是按顺序串行进行的，这一顺序自然隐含了token的位置信息。而在Transformer中，所有的token是被并行输入的，没有额外的向量或信息来指示一个词在序列中的位置。

Transformer的设计者提出了另一个创新特性：位置编码（Positional Encoding）。让我们看看位置编码是如何工作的。

### 位置编码

![位置编码](screenshots/positional-encoding.svg)

为了控制Transformer的训练时间和注意力的运算复杂度，Transformer没有单独创建另一个向量空间来描述token的位置信息。一方面是因为这样需要像训练词嵌入模型一样通过训练来得到位置信息向量（有一些基于Transformer的模型采用了可训练的位置编码），增加了训练时间。另一方面，需要将位置向量传给后续的层进行处理，需要进行更多点积运算。

Transformer的做法是使用一个固定的（不可训练的）位置编码函数，其输出向量具有固定的大小dmodel = 512（或模型的其他常量值）。然后将位置编码函数输出的位置编码向量直接加到词嵌入向量上。

原始Transformer中使用正弦和余弦函数来为每个位置$\text{pos}$和$d_\text{model}=512$维中每一个维度$i$生成相应的值：

$$
\begin{aligned}
\text{PE}(\text{pos}, 2i)&=\sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_\text{model}}}}\right) \\
\text{PE}(\text{pos}, 2i+1)&=\cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_\text{model}}}}\right)
\end{aligned}
$$

例如，第0个位置上的token的位置编码向量为$(\text{PE}(1, 1),\text{PE}(1, 2),\cdots,\text{PE}(1, 512))$.

以下是位置编码函数的简单代码实现：

```{.python .input}
# Hide outputs
import math


def positional_encoding(pos, d_model=512):
    pe = [0 for _ in range(d_model)]
    for i in range(0, 512, 2):
        pe[i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
        pe[i+1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe
```

可以由此计算得到任意位置的位置编码向量，它们都是长度为`d_model`的向量：

```{.python .input}
for pos in range(5):
    print(positional_encoding(pos))
```
