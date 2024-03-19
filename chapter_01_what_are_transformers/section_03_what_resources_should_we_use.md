# 我们需要什么资源

工业4.0的人工智能模糊了云平台、框架、库、编程语言和模型之间的界限。Transformer是一种新兴的模型，其生态系统的范围和数量令人惊叹。谷歌云提供了现成的Transformer模型供使用。OpenAI部署了一个几乎不需要编程既可以使用的Transformer API。Hugging Face提供了一个云端库服务。类似的服务还有很多。

本节将对我们在本书中将要实现的一些Transformer生态系统进行高层次的分析。

在实现用于自然语言处理的Transformer时，资源的选择非常关键，关乎项目的生存。在面对不同的项目和客户时，可能需要选用不同的资源。如果你只专注于自己喜欢的解决方案，很可能在某个时刻会遭受质疑。你应该专注于你所需要的系统，而不是你喜欢的系统。本书的目标不是解释市场上存在的每种Transformer解决方案，而是旨在解释足够多的Transformer生态系统，让你能够灵活应对在NLP项目中遇到的任何情况。

在本节中，我们将讨论在资源的选择和使用中面临的一些挑战。但首先，让我们从API开始。

## Transformer API

我们现在已经进入了人工智能的工业化时代。微软、谷歌、亚马逊网络服务（AWS）和IBM等公司，提供了无法超越的人工智能服务，任何开发者或开发团队都难以望其项背。科技巨头拥有价值百万美元的超级计算机，以及海量数据集用于训练Transformer模型和其他人工智能模型。

大型科技巨头拥有广泛的企业客户群，这些客户已经在使用它们的云服务。因此，将Transformer API添加到现有的云架构中所需的工作量比其他解决方案要少。

小公司甚至个人可以通过API以几乎没有开发投入的方式访问最强大的Transformer模型。一个实习生可以在几天内实现这个API。对于这样一个简单的实现，不需要成为工程师或拥有博士学位。

例如，OpenAI平台现在提供了一种基于SaaS（软件即服务）的API，用于市场上一些最有效的Transformer模型。OpenAI的API购买和使用方法请参考[这里](https://openai.com/blog/openai-api)

使用OpenAI的API只需要以下简单的步骤：

1. 购买并获取API密钥
2. 使用一行代码在笔记本中导入OpenAI
3. 输入元语言提示词指定你所希望完成的NLP任务
4. 收到以自然语言描述的任务完成结果

AllenNLP提供了一些机器学习的[演示程序（Demo）](https://allenai.org/demos)供用户学习。同时AllenNLP可以作为软件包安装到Python环境中进行调用。例如，假设我们需要进行一个共指消解（Coreference Resolution）任务。共指消解任务是指找到一个词所指的实体。下面展示了使用AllenNLP进行共指消解的代码及其运行结果：

```{.python }
# 安装allennlp和相关软件包
!pip install -U allennlp allennlp-models spacy
# 安装可视化工具
!git clone https://github.com/sattree/gpr_pub.git
```

```{.python .input}
from allennlp.predictors import Predictor
from gpr_pub import visualization
from IPython.core.display import display, HTML

# 加载样式文件
display(HTML(open('gpr_pub/visualization/highlight.css').read()))
display(HTML(open('gpr_pub/visualization/highlight.js').read()))

# 加载指代消解模型
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

# 示例输入文本
text = "Barack Obama was born in Hawaii. He was the 44th President of the United States."

# 进行指代消解预测
output = predictor.predict(document=text)

# 可视化指代消解的结果
visualization.render(output, allen=True, jupyter=True)
```

可以看到，这里的共指消解为单词“He”找到了其所指的实体“Barack Obama”。

## 基于API的软件库

在这本书中，我们将探索几个库。例如，谷歌拥有世界上一些最先进的人工智能实验室。谷歌的Trax库可以在Google Colab中通过几行代码进行安装。您可以选择免费或付费的服务。我们可以获取源代码，调整模型，甚至在自己的服务器或谷歌云上进行训练。例如，从使用现成的API定制一个用于翻译任务的Transformer模型是一个进一步的步骤。

然而，在某些情况下，这可能会证明既具有教育意义又有效果。我们将探索谷歌在翻译方面的最新进展，并在“:ref:`chapter-6`”中实现谷歌Trax。

在众多算法中，使用了Transformer的最著名的在线应用之一是谷歌翻译（Google Translate）。谷歌翻译可以通过在线平台或API进行使用。

我们看一个使用谷歌翻译API将英文翻译成中文的例子：

```{.python }
!pip install googletrans==3.1.0a0
```

```{.python .input}
from googletrans import Translator

translator = Translator()
result = translator.translate("Beijing is the capital of China.", src="en", dest="zh-CN")
result.text
```

## 预训练Transformer模型

大型科技公司主导着NLP市场。仅谷歌、Facebook和微软每天就运行数十亿次NLP例程，增强了它们的AI模型无与伦比的能力。这些巨头现在提供了各种各样的Transformer模型，并拥有排名靠前的基座模型。

然而，小型公司也看到了庞大的NLP市场，并加入了竞争。Hugging Face现在也提供免费或付费的服务方式。对于Hugging Face来说，要达到通过谷歌研究实验室投入数十亿美元和微软对OpenAI的资金支持所获得的效率水平将是具有挑战性的。基础模型的入门点是在超级计算机上进行完全训练的Transformer，例如GPT-3或Google BERT。

Hugging Face采用了不同的方法，并为各种任务提供了广泛且数量众多的Transformer模型，这是一种有趣的理念。Hugging Face提供了灵活的模型选择。此外，Hugging Face还提供了高级API和开发者可控的API。在本书的多个章节中，我们将探索Hugging Face作为一种教育工具和特定任务的可能解决方案。

我们看一下利用Hugging Face提供的软件库和不同的预训练Transformer模型实现自然语言生成任务的代码：

```{.python }
# 安装Hugging Face transformer库
!pip install transformers
```

```{.python .input}
from transformers import pipeline, set_seed

# 固定随机种子
set_seed(42)
# 选择gpt2模型
gpt2 = pipeline("text-generation", model="gpt2")

gpt2("Hello, I'm a language model.", max_length=10)
```

```{.python .input}
from transformers import pipeline, set_seed

# 固定随机种子
set_seed(42)
# 选择Bart模型
dialogpt = pipeline("text-generation", model="facebook/bart-base")
dialogpt("Hello, I'm a language model.", max_length=10)
```

然而，OpenAI专注于全球最强大的少数几个Transformer引擎，并可以在许多NLP任务上达到人类水平。在“:ref:`chapter-7`”中，我们将展示OpenAI的GPT-3引擎的强大能力。

至此，对于Transformer的资源，我们有多种选择（Transformer API、基于API的软件库、以及预训练Transformer模型）。这些相对立且经常冲突的选择给我们留下了许多可能的实施方式。因此，我们必须明确工业4.0人工智能专家的角色。

## 工业4.0时代人工智能专家的角色

工业4.0将一切与一切连接在一起，无所不在。机器直接与其他机器进行通信。由人工智能驱动的物联网信号触发自动化决策，无需人为干预。自然语言处理算法发送自动报告、摘要、电子邮件、广告等等。

人工智能专家将需要适应这个新时代下日益自动化的任务，包括Transformer模型的实现。人工智能专家将有新的职能。如果我们按照从高到低的顺序列出一个人工智能专家需要完成的Transformer自然语言处理任务，似乎有一些高级任务对于人工智能专家来说需要很少或几乎不需要开发。一个人工智能专家可以成为在人工智能领域提供设计理念、解释和实施的大师。

对于一个人工智能专家来说，Transformer的实际定义将会因生态系统而异。让我们通过几个例子来说明：

- API：OpenAI API不需要AI开发人员。一个网页设计师可以创建一个表单，一个语言学家或领域专家可以准备提示输入文本。一个AI专家的主要角色将需要语言技巧，以展示而不仅仅是告诉GPT-3引擎如何完成任务。例如，展示涉及对输入的上下文进行处理。这项新任务被称为提示工程。一个提示工程师在人工智能领域有着广阔的前景！
- API驱动的软件库：Google Trax库需要一定程度的开发工作来使用现成的模型。精通语言学和自然语言处理任务的AI专家可以处理数据集和输出。
- 训练和微调：Hugging Face的一些功能需要一定程度的开发，提供API和库。然而，在某些情况下，我们仍然需要亲自动手。在这种情况下，训练、微调模型和找到正确的超参数将需要人工智能专家的专业知识。
- 开发级技能：在一些项目中，分词器和数据集可能不匹配，如“:ref:`chapter-9`”中所述。在这种情况下，例如与语言学家合作的人工智能开发人员可以发挥关键作用。因此，在这个层面上，计算语言学的培训非常有用。

最近的NLP AI发展可以称为"嵌入式Transformer"，这正在颠覆AI开发生态系统：

- GPT-3 Transformer目前嵌入在几个Microsoft Azure应用程序中，例如GitHub Copilot。正如本章的Foundation models部分介绍的那样，Codex是我们将在“:ref:`chapter-16`”中研究的另一个例子。
- 这些嵌入式Transformer不能直接访问，但可以提供自动化开发支持，例如自动生成代码。
- 对于终端用户来说，使用嵌入式Transformer是无缝的，通过辅助文本完成来实现。

要直接访问GPT-3引擎，您首先需要创建一个OpenAI账户。然后，您可以使用API或直接在OpenAI用户界面中运行示例。也可以通过(poe.com)[https://poe.com/ChatGPT]等代理平台间接访问。
