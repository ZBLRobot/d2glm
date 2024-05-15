# 微调BERT
:label:`chapter-03-section-02`

本节将对一个BERT模型进行微调，以预测下游任务的可接受性判断，并使用马修斯相关系数（Matthews Correlation Coefficient，MCC，后续解释）来衡量预测结果。本节的官方源码[见此处](https://github.com/Denis2054/Transformers-for-NLP-2nd-Edition/blob/main/Chapter03/BERT_Fine_Tuning_Sentence_Classification_GPU.ipynb)。

## 硬件限制

Transformer模型需要GPU. 这里建议使用免费的云GPU平台:

- [Google Colab](https://colab.research.google.com/)
- [Kaggle Notebook](https://www.kaggle.com/code)

如果使用自己的电脑或自己搭建的服务器上的GPU，请先安装CUDA和cuDNN（Google/问GPT）

## 安装PyTorch和Hugging Face Transformer

Hugging Face提供了一个预训练的BERT模型。Hugging Face开发了一个名为PreTrainedModel的基类。通过安装这个类，我们可以从预训练的模型配置中加载一个模型。

Hugging Face提供了TensorFlow和PyTorch的模块。我建议开发者对这两个环境都有一定的熟悉。优秀的人工智能研究团队可能会使用其中一个或两个环境。

```{.python }
# Hide outputs
!pip install -q torch transformers
```

## 导入模块

我们将导入所需的预训练模块，例如预训练的BERT分词器和BERT模型的配置。同时，我们还导入了AdamW优化器以及序列分类模块：

```{.python .input}
# Hide outputs
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange  #for progress bars
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
```

如果一切顺利，不会显示任何消息。Google Colab已经在我们使用的虚拟机上预先安装了这些模块。

## 指定CUDA作为设备

我们现在将指定torch使用CUDA（Compute Unified Device Architecture）来利用NVIDIA GPU的并行计算能力，用于我们的多头注意力模型：

```{.python .input}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
!nvidia-smi
```

输出结果可能会因Google Colab的配置而有所不同。

## 加载CoLA数据集

使用[CoLA数据集]((https://nyu-mll.github.io/CoLA/))：

```{.python }
# Hide outputs
!curl -L https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-2nd-Edition/master/Chapter03/in_domain_train.tsv --output "in_domain_
train.tsv"
!curl -L https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-2nd-Edition/master/Chapter03/out_of_domain_dev.tsv --output "out_of_
domain_dev.tsv"
```

```{.python .input}
df = pd.read_csv("in_domain_train.tsv", delimiter="\t", header=None, names=["sentence_source", "label", "label_notes", "sentence"])
df.shape
```

可以看到，数据集的形状是`(8551, 4)`，说明我们加载了8551条数据，每条数据包含4个属性。

从中取10条数据看看：

```{.python .input}
df.sample(10)
```

`.tsv`文件中的每个样本包含四列，以制表符分隔：

- 第一列：句子（代码）的来源
- 第二列：标签（0=不可接受，1=可接受）
- 第三列：作者标注的标签
- 第四列：待分类的句子

## 创建句子和标签列表并添加特殊Tokens

将所需分类的句子和分类标签从数据集表格中取出来，并在句子的开头和结尾分别加上`[CLS]`和`[SEP]`特殊token.

```{.python .input}
sentences = df.sentence.values
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values
```

## 使用BERT分词器

在本节中，我们将初始化一个预训练的BERT分词器。这样可以节省从头开始训练分词器所需的时间。

我们使用一个不区分大小写的分词器，然后打印第一个分词后的句子：

```{.python .input}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])
```

可以看到，分词器的输出包含了分类token `[CLS]`和分隔token `[SEP]`.

`BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)`需要从huggingface.com上下载内容，可能会因为网络原因报错，有以下解决方案：

- 使用全局梯子
- 从镜像网站`https://hf-mirror.com/google-bert/bert-base-uncased/tree/main`下载以下文件并保存到运行目录下的`bert-base-uncased`文件夹：
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `vocab.txt`

## 数据处理

为了使数据可以成批地输入模型中，我们需要让他们拥有相同的长度。我们需要确定一个固定的最大长度。数据集中的句子很短，但为了确保这一点，程序将序列的最大长度设置为128，并对不足128的token序列进行填充：

```{.python .input}
MAX_LEN = 128
# 使用BERT分词器将token转换为BERT词表中的索引
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# 对输入tokens进行填充
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
print(input_ids[0])
```

我们打印第一个token序列，发现序列尾部有一大串0，这就是填充后的结果。在BERT模型中，填充token表示为`[PAD]`，其在词表中的索引为0.

## 创建注意力遮罩

填充的tokens唯一作用只是用来保证输入形状统一，它不应该参与注意力的计算而影响最终的结果。因此，我们要创建注意力遮罩来禁止模型对填充字符施加注意力。

一种简单的做法是将非填充token部分的注意力遮罩的值置为1，填充token部分的注意力遮罩的值置为0：

```{.python .input}
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
```

查看相互对应的token序列和注意力遮罩：

```{.python .input}
print(input_ids[2])
print(np.array(attention_masks[2]))
```

可以看到，token序列中填充部分的注意力遮罩的值为0，非填充部分则为1

## 分割训练集和验证集

```{.python .input}
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)
```

其中`test_size=0.1`表示分割10%作为验证集，`random_state=2018`表示固定2018作为随机分割的随机种子，保证每次分割的结果是相同的。

## 把所有数据转换为tensor

我们所微调的BERT模型采用PyTorch张量（tensor）作为输入和输出，因此我们要将数据转化为tensor.

```{.python .input}
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
```

## 设置一个批次大小（Batch Size）并创建数据迭代器

批次大小（Batch Size）表示同时并行输入模型的数据的数量，数据迭代器可以将数据集进行封装，并以指定的batch size和模式（随机/顺序）

```{.python .input}
batch_size = 32

# 将多个tensor列表封装成tensor数据集
train_data = TensorDataset(train_inputs, train_masks, train_labels)
# 随机采样器
train_sampler = RandomSampler(train_data)
# 创建一个DataLoader迭代器，可以以随机的顺序和`batch_size`的批次大小遍历数据集
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
# 顺序采样器
validation_sampler = SequentialSampler(validation_data)
# 创建一个DataLoader迭代器，可以以原始的顺序和`batch_size`的批次大小遍历数据集
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
```

## BERT模型配置

```{.python .input}
from transformers import BertModel, BertConfig
configuration = BertConfig()
# 通过配置初始化BERT模型
model = BertModel(configuration)
# 访问模型的配置
configuration = model.config
print(configuration)
```

我们来看看其中的一些主要配置：

- `attention_probs_dropout_prob: 0.1` 表示对注意力概率应用0.1的Dropout率（Dropout Rate）。
- `hidden_act: "gelu"` 是编码器中的非线性激活函数。它是高斯误差线性单元激活函数。输入通过其幅度加权，使其非线性化。
- `hidden_dropout_prob: 0.1` 是应用于全连接层的Dropout概率。全连接可以在嵌入层、编码器和池化器层中找到。输出并不总是对序列内容的良好反映。对隐藏状态序列进行池化可以改善输出序列。
- `hidden_size: 768` 是编码层和池化器层的维度，即$d_\text{model}$。
- `initializer_range: 0.02` 是初始化权重矩阵时的标准差值。
- `intermediate_size: 3072` 是编码器中前馈层的维度。
- `layer_norm_eps: 1e-12` 是用于层归一化层的epsilon值。
- `max_position_embeddings: 512` 是模型使用的最大长度。
- `model_type: "bert"` 是模型的名称。
- `num_attention_heads: 12` 是注意力头的数量。
- `num_hidden_layers: 12` 是编码器层的数量。
- `pad_token_id: 0` 是填充标记的ID，以避免对填充标记进行训练。
- `type_vocab_size: 2` 是token_type_ids的大小，用于识别序列。例如，"the dog [SEP] The cat [SEP]" 可以用token ID [0,0,0,1,1,1]表示。
- `vocab_size: 30522` 是模型用于表示input_ids的不同标记的数量。

## 加载Hugging Face预训练BERT模型

在了解了BERT模型的配置后我们来加载BERT的预训练模型。

这里使用的是`bert-base-uncased`模型，base表示模型的大小，uncased表示模型的词表和训练数据是不区分大小写的。

```{.python .input}
# 加载预训练模型，`num_labels=2`表示我们将要利用预训练模型来进行2分类任务
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# 对模型采用数据并行
model = nn.DataParallel(model)
# 将模型发送到期望的设备（GPU）上
model.to(device)
```

上面的代码加载了预训练模型，对模型采用数据并行，并将模型发送到GPU上。代码输出了模型的结构，可以看到BERT分类模型`BertForSequenceClassification`中包含了BERT骨干模型（Backbone Model）`BertModel`，`BertModel`中又包含了嵌入层`BertEmbeddings`和编码器`BertEncoder`，编码器`BertEncoder`中又包含一个`ModuleList`，其中有12个`BertLayer`. 有兴趣可以阅读[BertForSequenceClassification的源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1494)。


`BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)`也需要从huggingface.com上下载内容，可能会因为网络原因报错，解决方法同上。

## 将模型参数分组

通过`model.named_paramters()`方法可以获取模型参数的名字和参数的值：

```{.python .input}
param_optimizer = list(model.named_parameters())
names = [item[0] for item in param_optimizer]
values = [item[1] for item in param_optimizer]
```

为了防止过拟合，一种常见的方法是应用[正则化](https://zhuanlan.zhihu.com/p/29360425)。pytorch中的AdamW优化器可以通过指定`weight_decay`参数来控制正则化的力度。默认情况下`weight_decay=0.0`，即不进行正则化.

然而，在BERT模型中，并不是所有参数都需要进行正则化，因此我们需要将模型的参数分成两组，一组`weight_decay=0.1`，另一组`weight_decay=0`:

```{.python .input}
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    # 参数名称不包含`bias`和`LayerNorm.weight`的参数：正则化力度为0.1
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.1},
    # 参数名称包含`bias`和`LayerNorm.weight`的参数：正则化力度为0.0
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
```

## 训练流程

训练循环遵循标准的学习过程。将迭代的次数设置为4，并将损失和准确率的度量结果绘制出来。训练循环使用数据加载器来加载和训练批次数据。训练过程进行度量和评估。

BERT的训练流程遵循标准的深度学习的训练流程。我们将迭代次数（Epochs）设为4，并在训练过程中将模型的损失和准确率绘制出来。

首先创建优化器和学习率调度器，并指定模型训练的相关超参数：

```{.python .input}
# 训练轮数
epochs = 4

# 创建优化器
optimizer = AdamW(
    optimizer_grouped_parameters,
    lr = 2e-5, # 学习率
    eps = 1e-8 # AdamW的epsilon参数，一般是一个接近0的值，防止优化目标计算时分母为0
)

# 总的训练步数 = 数据加载器长度 * 训练轮数
total_steps = len(train_dataloader) * epochs

# 创建学习率调度器
# 这里表示创建一个线性的学习率调度器，
# 不进行warmup，在`total_steps`步中将学习率从上述`2e-5`降到`0`
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)
```

然后创建准确率衡量函数：

```{.python .input}
# 创建准确率衡量函数，计算预测值和标签的准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
```

最后正式开始训练流程：

```{.python }
epochs = 4

# 用于存储我们的损失和准确率，以便绘图
train_loss_set = []

# trange是对普通的Python range函数的tqdm包装
for _ in trange(epochs, desc="Epoch"):

    # 训练

    # 将模型设置为训练模式（与评估模式相对）
    model.train()

    # 追踪变量
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # 遍历训练集的DataLoader，完成1个epoch的训练
    for step, batch in enumerate(train_dataloader):
        # 将一个batch的数据移到GPU上
        batch = tuple(t.to(device) for t in batch)
        # 从DataLoader中unpack输入
        b_input_ids, b_input_mask, b_labels = batch
        # 清除梯度（因为默认情况下pytorch的优化器会对梯度进行累积）
        optimizer.zero_grad()
        # 模型前向传播
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels
        )
        loss = outputs['loss']
        train_loss_set.append(loss.item())
        # 反向传播
        loss.backward()
        # 更新参数并根据计算的梯度进行一步优化
        optimizer.step()

        # 更新追踪变量
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    # 验证

    # 将模型设置为评估模式，以评估验证集上的损失
    model.eval()

    # 追踪变量
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 在一个epoch内评估数据
    for batch in validation_dataloader:
        # 将批次移到GPU上
        batch = tuple(t.to(device) for t in batch)
        # 从数据加载器中解包输入
        b_input_ids, b_input_mask, b_labels = batch
        # 告诉模型不要计算或存储梯度，以节省内存并加快验证速度
        with torch.no_grad():
            # 前向传播，计算logit预测
            logits = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask)

        # 将logits和标签移到CPU上
        logits = logits['logits'].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
```

执行上述代码后，BERT模型将会在CoLA数据集上进行二分类任务的微调。每轮训练的损失和分类预测准确率会被打印出来。

## 训练评估

训练过程中模型的损失存储在`train_loss_set`中，我们可以将损失随训练过程的变化进行可视化：

```{.python }
plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()
```

上述代码执行后可以看到，随着训练的进行，模型的损失呈波动下降趋势。

## 使用测试集进行预测和评估

BERT下游模型是使用`in_domain_train.tsv`数据集进行训练的。现在，我们将使用另一个数据集，`out_of_domain_dev.tsv`数据集来预测句子语法是否正确。

```{.python .input}
df = pd.read_csv("out_of_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
df.shape
```

```{.python }
# 创建句子和标签的列表
sentences = df.sentence.values

# 添加特殊token
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]


MAX_LEN = 128

# 使用BERT分词器将token序列转换为其在词表中的索引
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# 填充输入序列
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# 创建注意力遮罩
attention_masks = []

# 非填充部分为1，填充部分为0
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

batch_size = 32


prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
```

```{.python }
# 计算softmax logits
import numpy as np

def softmax(logits):
    e = np.exp(logits)
    return e / np.sum(e)
```

```{.python }
import torch
import numpy as np

# 启用模型的评估模式
model.eval()

# 追踪变量
raw_predictions, predicted_classes, true_labels = [], [], []

# 预测
for batch in prediction_dataloader:
    # 把batch放到GPU上
    batch = tuple(t.to(device) for t in batch)
    # 从dataloader中解包出数据
    b_input_ids, b_input_mask, b_labels = batch
    # 告诉模型不要保存梯度，以节约显存，提高预测速度
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        # 前向传播，计算logits
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    # 把logits和标签放到CPU上
    logits = outputs['logits'].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 把token序列转换回单词字符串
    b_input_ids = b_input_ids.to('cpu').numpy()
    batch_sentences = [tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in b_input_ids]

    # 应用softmax函数，将模型输出的logits转化为概率
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

    # 取概率最高的类别作为预测结果
    batch_predictions = np.argmax(probabilities, axis=1)

    # 打印句子和预测结果
    for i, sentence in enumerate(batch_sentences):
        print(f"Sentence: {sentence}")
        print(f"Prediction: {logits[i]}")
        print(f"Sofmax probabilities", softmax(logits[i]))
        print(f"Prediction: {batch_predictions[i]}")
        print(f"True label: {label_ids[i]}")

    # 保存logits，预测类别和真实标签
    raw_predictions.append(logits)
    predicted_classes.append(batch_predictions)
    true_labels.append(label_ids)
```

## 使用马修斯相关系数（MCC）进行评估
:label:`chapter-2-section-2-mcc`

马修斯相关系数（MCC）的公式如下：

$$
MCC = \frac{TP\times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

- $TP$表示True Positive（真正），即预测为正，实际标签为正
- $TN$表示True Negative（真负），即预测为正，实际标签为正
- $FP$表示False Positive（假正），即预测为正，实际标签为负
- $FN$表示False Negative（假负），即预测为负，实际标签为正

最终的MCC得分将基于整个测试集，但让我们先看看各个batch的得分，以了解批次之间度量指标的变化情况。各batch的MCC评估代码如下：

```{.python }
from sklearn.metrics import matthews_corrcoef

# 初始化一个空列表来存储每个batch的MCC
matthews_set = []

# 遍历batch
for i in range(len(true_labels)):
    # 计算该batch的MCC

    # true_labels[i]是该batch中所有的真实标签
    # predicted_classes[i]是该batch中所有的预测分类
    # 我们不需要使用np.argmax，因为我们在前面已经将预测分类保存在了predicted_classes中

    matthews = matthews_corrcoef(true_labels[i], predicted_classes[i])

    # 把MCC结果加到列表中
    matthews_set.append(matthews)

# Now matthews_set contains the Matthews correlation coefficient for each batch
# 现在，matthews_set包含了每个batch的MCC
```

整个测试集上的MCC评估代码如下：

```{.python }
from sklearn.metrics import matthews_corrcoef

# true_labels和predicted_classes中每个数据对应一个batch的数据
# 现在我们将其展平，使列表中每个元素对应测试集的一条数据
true_labels_flattened = [label for batch in true_labels for label in batch]
predicted_classes_flattened = [pred for batch in predicted_classes for pred in batch]

# 为整个测试集的预测结果计算MCC分数
mcc = matthews_corrcoef(true_labels_flattened, predicted_classes_flattened)

print(f"MCC: {mcc}")
```

## 保存模型

现在我们的模型一直存储在显存和内存中，程序结束时会丢失。为了能在以后也能使用该模型，需要将模型保存下来。

如果我们使用了`DataParallel`将模型封装为多GPU训练对象，原始的模型会被封装到`DataParallel`对象中，称为该对象的一个属性。

因此，我们在保存模型时需要区分模型是否经过`DataParallel`封装：

```{.python }
!mkdir saved_models/
```

```{.python }
# Specify a directory to save your model and tokenizer
save_directory = "saved_models/"


# If your model is wrapped in DataParallel, access the original model using .module and then save
if isinstance(model, torch.nn.DataParallel):
    model.module.save_pretrained(save_directory)
else:
    model.save_pretrained(save_directory)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)
```
