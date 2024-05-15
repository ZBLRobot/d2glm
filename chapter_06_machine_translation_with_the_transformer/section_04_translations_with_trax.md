# 使用Trax进行翻译

Google Brain开发了Tensor2Tensor (T2T)来简化深度学习开发。T2T是TensorFlow的扩展,包含了许多Transformer的示例模型。

虽然T2T是一个不错的起点,但Google Brain随后推出了Trax,一个端到端的深度学习库。Trax包含一个可应用于翻译的Transformer模型。目前,Google Brain团队维护着Trax。

本节将重点关注初始化Vaswani等人（2017年）描述的英语-德语问题的最小功能,以说明Transformer的性能。

我们将使用预处理的英语和德语数据集来显示Transformer架构是与语言无关的。

以下代码需要访问Google Cloud Storage上的远程文件，使用国内网络请先开启全局梯子。

我们将首先安装所需的模块。

## 安装Trax

```{.python }
!pip install -U trax gsutil
```

## 下载预训练权重和词表

从Google Cloud Storage上下载预训练权重和词表到本地。也可以在代码中直接使用gs远程路径。

```{.python }
!gsutil cp gs://trax-ml/models/translation/ende_wmt32k.pkl.gz ./
!mkdir -p vocabs
!gsutil cp gs://trax-ml/vocabs/ende_32k.subword ./vocabs/
```

## 创建原始Transformer模型

```{.python .input}
import os
import numpy as np
import trax

model = trax.models.Transformer(
    input_vocab_size=33300,
    d_model=512,
    d_ff=2048,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    max_len=2048,
    mode='predict'
)
```

该模型是一个具有编码器和解码器的Transformer，编码器和解码器都包含6个层和8个头。$d_\text{model}=512$,与原始Transformer架构一致。

## 使用预训练权重初始化模型

```{.python .input}
# 远程路径为gs://trax-ml/models/translation/ende_wmt32k.pkl.gz
model.init_from_file('./ende_wmt32k.pkl.gz', weights_only=True)
```

## 对句子进行分词

```{.python .input}
sentence = 'I am only a machine but I have machine intelligence.'
# vocab_dir的远程路径为gs://trax-ml/vocabs/
tokenized = list(trax.data.tokenize(iter([sentence]), vocab_dir='./vocabs/', vocab_file='ende_32k.subword'))[0]
```

```{.python .input}
tokenized
```

## 解码

Transformer会将英语句子编码,并将其解码为德语。模型及其权重构成了它的能力集合。

```{.python .input}
tokenized = tokenized[None, :] # 增加一个batch维度
tokenized_translation = trax.supervised.decoding.autoregressive_sample(model, tokenized, temperature=0.0) # temperature参数越高，解码得到的句子越具有多样性
```

```{.python .input}
tokenized_translation
```

## 合词（De-tokenizing）并显示翻译结果

```{.python .input}
tokenized_translation = tokenized_translation[0][:-1] # 移除batch维度和EOS token
translation = trax.data.detokenize(tokenized_translation, vocab_dir='./vocabs/', vocab_file='ende_32k.subword') # 合词
print("The sentence:", sentence)
print("The translation:", translation)
```

可以用谷歌翻译等工具检查上述翻译的准确性。

Transformer告诉我们,虽然它是机器,但它确实有视野。通过Transformer,机器智能正在不断增长,但它并非人类智能。机器通过自己独特的智能来学习语言。
