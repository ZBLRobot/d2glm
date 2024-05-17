# 微调GPT-3

## 微调OpenAI的Ada引擎

OpenAI提供了对GPT-3进行微调的API，只需要设置API key，准备数据，指定模型，即可在云端进行微调。

```{.python }
!export OPENAI_API_KEY="[REPLACE THIS WITH YOUR KEY]"
!openai api fine_tunes.create -t "kantgpt_prepared.jsonl" -m "ada"
```

上述命令表示使用`"kantgpt_prepared.jsonl"`数据对`"ada"`模型进行微调。

## 使用微调后模型进行文本补全

```{.python }
!openai api completions.create -m ada:[YOUR_MODEL_INFO] "Several concepts are a priori such as"
```

其中`[YOUR_MODEL_INFO]`会在完成微调之后显示。

补全结果如下：

Several concepts are a priori such as the term freedom and the concept of free will.
