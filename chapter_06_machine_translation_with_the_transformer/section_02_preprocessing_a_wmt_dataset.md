# 预处理WMT数据集

Vaswani等人（2017年）在WMT 2014年英德翻译任务和英法翻译任务中展示了Transformer的成就。Transformer在这些任务中取得了最先进的BLEU分数。

2014年的WMT包含了多个欧洲语言的数据集。其中一个数据集包含了来自欧洲议会语料库第7版的数据。我们将使用来自[《欧洲议会会议平行语料库》（the European Parliament Proceedings Parallel Corpus）（1996-2011年）的法英（French-English）数据集](https://www.statmt.org/europarl/v7/fr-en.tgz)。

下载上述链接中的文件并将其解压缩，可以得到两个平行语料文件：

- `europarl-v7.fr-en.en`
- `europarl-v7.fr-en.fr`

其中`.en`结尾的是英文，`.fr`结尾的是法文，二者之间每行一一对应。

我们将对上述平行语料进行预处理。

## 下载数据集

```{.python }
!wget https://www.statmt.org/europarl/v7/fr-en.tgz
!tar -xvf fr-en.tgz
```

## 预处理原始数据

以下代码用于处理上述两个平行语料文件。

首先定义一些函数：

```{.python .input}
# 加载文件中的文本
def load_doc(filename):
    file = open(filename, mode="rt", encoding="utf-8")
    text = file.read()
    file.close()
    return text

# 按换行符进行分割，得到每行的要翻译的句子
def to_sentences(doc):
    return doc.strip().split("\n")

# 获取最短和最长句子的长度
def sentence_lengths(sentences):
    lengths = [len(s.split()) for s in sentences]
    return min(lengths), max(lengths)
```

导入的句子行必须进行清洗，以避免训练无用和带有噪音的标记。行进行规范化处理，按空格进行分词，并转换为小写。从每个标记中删除标点符号，删除非可打印字符，并排除包含数字的标记。清洗后的行以字符串形式存储。

以下是清洗代码（不适用于中文）：

```{.python .input}
# clean lines
import re
import string
import unicodedata
def clean_lines(lines):
    cleaned = list()
    # 正则表达式，用于过滤字符
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # 准备转换表，用于去除标点符号
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # 标准化unicode字符
        line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        # 使用空格进行分词
        line = line.split()
        # 转小写
        line = [word.lower() for word in line]
        # 去除标点符号
        line = [word.translate(table) for word in line]
        # 去除不可打印的字符
        line = [re_print.sub('', w) for w in line]
        # 去除包含数字的token
        line = [word for word in line if word.isalpha()]
        # 保存为字符串
        cleaned.append(' '.join(line))
    return cleaned
```

我们已经定义了准备数据集时将调用的关键函数。现在我们加载并清理英文数据：

```{.python .input}
filename = 'europarl-v7.fr-en.en'
doc = load_doc(filename)
sentences = to_sentences(doc)
minlen, maxlen = sentence_lengths(sentences)
print('English data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))
cleanf = clean_lines(sentences)
```

数据集现在干净了，我们使用`pickle`将其保存为序列化的文件，命名为`English.pkl`：

```{.python .input}
import pickle

filename = 'English.pkl'
outfile = open(filename,'wb')
pickle.dump(cleanf, outfile)
outfile.close()
print(filename," saved")
```

输出显示了数据集的关键信息，并确认`English.pkl`已被保存。

我们对法文数据也进行同样的处理：

```{.python .input}
filename = 'europarl-v7.fr-en.fr'
doc = load_doc(filename)
sentences = to_sentences(doc)
minlen, maxlen = sentence_lengths(sentences)
print('French data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))
cleanf = clean_lines(sentences)
filename = 'French.pkl'
outfile = open(filename,'wb')
pickle.dump(cleanf, outfile)
outfile.close()
print(filename," saved")
```

主要的预处理已完成。但我们仍然需要确保数据集不包含噪音和易混淆的标记。

## 进一步预处理

现在，我们定义了两个函数，用于加载在前一部分中清理过的数据集，并在进一步预处理完成后保存句子：

```{.python .input}
from collections import Counter

# 加载一个干净的数据集
def load_clean_sentences(filename):
    return pickle.load(open(filename, 'rb'))

# 保存进一步预处理后的句子
def save_clean_sentences(sentences, filename):
    pickle.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)
```

一个单词在数据集中出现的频次非常重要。例如，如果一个词在包含两百万行的数据集中只出现一次，那么我们将浪费宝贵的GPU资源来学习它。现在我们定义一个函数来统计单词出现的频次：

```{.python .input}
# 为所有单词创建一个频率表
def to_vocab(lines):
    vocab = Counter()
    for line in lines:
        tokens = line.split()
        vocab.update(tokens)
    return vocab
```

使用以下函数可以过滤掉所有频次小于`min_occurrence`的单词：

```{.python .input}
# 过滤掉所有频次小于min_occurence的单词
def trim_vocab(vocab, min_occurrence):
    tokens = [k for k,c in vocab.items() if c >= min_occurrence]
    return set(tokens)
```

现在我们需要处理词表外词汇（Out-Of-Vocabulary Words, OOV Words）。OOV词汇可以是拼写错误的单词、缩写词或任何不符合标准词汇表示的单词。我们可以使用自动拼写纠正，但这并不能解决所有问题。在这个例子中，我们将简单地用`unk`（Unknown）标记替换OOV词汇：

```{.python .input}
# 将每行中的OOV词汇标记为unk
def update_dataset(lines, vocab):
    new_lines = list()
    for line in lines:
        new_tokens = list()
        for token in line.split():
            if token in vocab:
                new_tokens.append(token)
            else:
                new_tokens.append('unk')
        new_line = ' '.join(new_tokens)
        new_lines.append(new_line)
    return new_lines
```

以上定义了我们所需的功能函数，现在我们在清洗干净的数据的基础上来运行进一步的预处理：

```{.python .input}
# 加载英文数据集
filename = 'English.pkl'
lines = load_clean_sentences(filename)
# 计算词汇频次
vocab = to_vocab(lines)
print('English Vocabulary: %d' % len(vocab))
# 过滤掉出现不足5次的词汇
vocab = trim_vocab(vocab, 5)
print('New English Vocabulary: %d' % len(vocab))
# 将OOV词汇标为unk
lines = update_dataset(lines, vocab)
# 保存进一步处理后的数据集
filename = 'english_vocab.pkl'
save_clean_sentences(lines, filename)
# 检查前20条
for i in range(20):
    print("line",i,":",lines[i])
```

法文数据处理流程相同：

```{.python .input}
# 加载法文数据集
filename = 'French.pkl'
lines = load_clean_sentences(filename)
# 计算词汇频次
vocab = to_vocab(lines)
print('French Vocabulary: %d' % len(vocab))
# 过滤掉出现不足5次的词汇
vocab = trim_vocab(vocab, 5)
print('New French Vocabulary: %d' % len(vocab))
# 将OOV词汇标为unk
lines = update_dataset(lines, vocab)
# 保存进一步处理后的数据集
filename = 'french_vocab.pkl'
save_clean_sentences(lines, filename)
# 检查前20条
for i in range(20):
    print("line",i,":",lines[i])
```

以上展示了在训练之前需要如何处理原始数据。数据集现在准备好了，可以输入到Transformer模型中进行训练。

对于法译英任务，法文数据集的每一行是要翻译的句子，英文数据集的每一行是机器翻译模型的参考翻译。机器翻译模型必须生成与参考翻译相匹配的英文候选翻译。

在下一节中，BLEU提供了一种评估机器翻译模型生成的候选翻译的方法。
