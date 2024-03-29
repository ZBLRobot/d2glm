# Hugging Face上的Transformer模型

您在本章中看到的所有内容都可以简化为一个可直接使用的Hugging Face Transformer模型。

有了Hugging Face，用三行代码就能实现机器翻译！

首先我们需要在Python中安装Hugging Face transformers:

```{.python .input}
!pip -q install transformers
```

然后我们导入Hugging Face的pipeline方法，其中包含了transformer的多种用法：

```{.python .input}
from transformers import pipeline
```

然后，我们创建一个翻译模型并输入一个要从英文翻译成法文的句子，以演示原始Transformer的功能：

```{.python .input}
translator = pipeline("translation_en_to_fr")
print(translator("It is easy to translate languages with Transformers.", max_length=40))
```

可以看到，我们得到了一个翻译后的法文句子。

使用Hugging Face上发布的其他模型也可以实现将英文翻译成中文：

```{.python .input}
translator = pipeline("translation_en_to_zh", model="liam168/trans-opus-mt-en-zh")
print(translator("It is easy to translate languages with Transformers.", max_length=40))
```
