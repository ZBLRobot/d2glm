# 模型训练和表现

原始的Transformer模型的训练数据包含450万条英文-德文数据集和3600万条英文-法文数据集。

数据集来源于[Workshops on Machine Translation（WMT）](http://www.statmt.org/wmt14/)

原始Transformer基础模型在8张NVIDIA P100 GPU上训练了10万步，花费12小时。Transformer大模型训练了30万步，花费3.5天。

原始Transformer在WMT英译法的数据集上取得了41.8的[BLEU分数](https://zhuanlan.zhihu.com/p/338488036)，超越了以往所有的机器翻译模型。

BLEU表示Bilingual Evaluation Understudy，是一种用于评估翻译质量的算法。

Google Research和Google Brain团队应用了优化策略来改善Transformer的性能。例如，使用Adam优化器，使用带预热（Warmup）的学习率调度，即先让学习率从0线性提升到所设置的学习率，然后让学习率缓慢地线性递减。

在嵌入的求和部分应用了不同类型的正则化技术，例如[暂退（Dropout）](https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html)和残差暂退（Residual Dropout）。

此外，Transformer还应用了标签平滑（Label Smoothing）取代了过度自信的独热编码（One-hot），以防止过拟合。

其他几种Transformer模型的变体已经促成了新的模型和用法，我们将在后续章节中进行探索。

在结束本章之前，让我们感受一下Hugging Face中现成可用的Transformer模型的简便性。