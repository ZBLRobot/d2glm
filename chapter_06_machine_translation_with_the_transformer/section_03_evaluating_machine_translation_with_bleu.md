# 使用BLEU评估机器翻译的质量

[Papineni等人（2002年）](https://aclanthology.org/P02-1040.pdf)提出了一种评估人类翻译的有效方法。人类基准很难定义。然而，他们意识到，如果我们逐字逐词地将人类翻译与机器翻译进行比较，我们可以获得有效的结果。

[Papineni等人（2002年）](https://aclanthology.org/P02-1040.pdf)将他们的方法命名为双语评估替补分数（BiLingual Evaluation Understudy Score, BLEU）。

本章介绍如何使用[自然语言工具包（Natural Language Toolkit, NLTK）](http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu)来计算BLEU分数。

## N-gram

N-gram是一种在自然语言处理和文本分析中常用的技术。它是一种基于文本中连续N个项（例如单词、字母或音节）的序列进行建模和分析的方法。

在N-gram中，N代表项的数量。例如，一个2-gram（也称为bigram）表示由两个连续项组成的序列，如单词序列"the cat"。类似地，一个3-gram（也称为trigram）由三个连续项组成，如单词序列"I am a"。

BLEU分数的评估一般考虑1-gram到4-gram.

## 几何评估

BLEU方法将候选句子的部分与参考句子或多个参考句子进行比较。

以下代码模拟了机器翻译模型生成的候选翻译与数据集中实际翻译参考之间的比较。请记住，一句话可能会被重复多次，并由不同的翻译人员以不同的方式翻译，这使得找到有效的评估策略变得具有挑战性。

```{.python .input}
from nltk.translate.bleu_score import sentence_bleu

#Example 1
reference = [['the', 'cat', 'likes', 'milk'], ['cat', 'likes' 'milk']]
candidate = ['the', 'cat', 'likes', 'milk']
score = sentence_bleu(reference, candidate)
print('Example 1', score)
#Example 2
reference = [['the', 'cat', 'likes', 'milk']]
candidate = ['the', 'cat', 'likes', 'milk']
score = sentence_bleu(reference, candidate)
print('Example 2', score)
```

候选翻译$C$、参考翻译$R$以及在$C$中找到的正确N-gram数量$N$之间的直接评估$P$可以表示为一个几何函数：

$$
P(N,C,R)=\prod_{i=1}^Np_i
$$

其中$p_i$表示第$i$个N-gram是否出现在参考翻译中，如果出现则取值1，否则取值0.

然而这种直接的几何评估方法缺乏灵活性，例如：

```{.python .input}
# Example 3
reference = [['the', 'cat', 'likes', 'milk']]
candidate = ['the', 'cat', 'enjoys','milk']
score = sentence_bleu(reference, candidate)
print('Example 3', score)
```

人类可以看出来，候选翻译与参考翻译是同一个意思，我们也期望机器至少能判断两个句子有一定的相似性。

但是上面的代码输出的BLEU分数接近0，原因在于候选翻译中没有3-gram或4-gram出现在参考翻译中。为了改善这一情况，可以使用平滑评估

## 平滑评估

```{.python .input}
from nltk.translate.bleu_score import SmoothingFunction


# Example 3
reference = [['the', 'cat', 'likes', 'milk']]
candidate = ['the', 'cat', 'enjoys','milk']
smoothing = SmoothingFunction()
score = sentence_bleu(reference, candidate, smoothing_function=smoothing.method2)
print('Example 3 (Smoothed)', score)
```
