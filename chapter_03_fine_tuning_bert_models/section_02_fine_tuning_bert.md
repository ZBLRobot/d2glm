# 微调BERT

本节将对一个BERT模型进行微调，以预测下游任务的可接受性判断，并使用马修斯相关系数（Matthews Correlation Coefficient，MCC，后续解释）来衡量预测结果。

#### 硬件限制

Transformer模型需要GPU. 这里强烈建议使用免费的云GPU平台，因为在本地自己配置GPU计算环境较为复杂。

- [Google Colab](https://colab.research.google.com/)
- [Kaggle Notebook](https://www.kaggle.com/code)

#### 安装PyTorch和Hugging Face Transformer

Hugging Face提供了一个预训练的BERT模型。Hugging Face开发了一个名为PreTrainedModel的基类。通过安装这个类，我们可以从预训练的模型配置中加载一个模型。

Hugging Face提供了TensorFlow和PyTorch的模块。我建议开发者对这两个环境都有一定的熟悉。优秀的人工智能研究团队可能会使用其中一个或两个环境。

```{.python }
# Hide outputs
!pip install -q torch transformers
```

#### 导入模块

我们将导入所需的预训练模块，例如预训练的BERT分词器和BERT模型的配置。同时，我们还导入了BERTAdam优化器以及序列分类模块：

```{.python .input}
# Hide outputs
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
```

再导入一个好看的进度条包`tqdm`:

```{.python .input}
from tqdm import tqdm, trange
```

最后导入机器学习中常用的模块：

```{.python .input}
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
```

如果一切顺利，不会显示任何消息，需要注意的是，Google Colab已经在我们使用的虚拟机上预先安装了这些模块。

#### 指定CUDA作为设备

我们现在将指定torch使用CUDA（Compute Unified Device Architecture）来利用NVIDIA GPU的并行计算能力，用于我们的多头注意力模型：

```{.python .input}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
!nvidia-smi
```

输出结果可能会因Google Colab的配置而有所不同。

#### 加载数据集

