# lm-evaluation-harness

## 概述

评测大模型的通用框架

[EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models. (github.com)](https://github.com/EleutherAI/lm-evaluation-harness)



## 例子

```sh
lm_eval --model hf \
    --model_args pretrained=/data/xhao/work_space/llm_resource/deepseek_llm_7b_chat,parallelize=True\
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```





## Evaluate

### 快速上手

可以直接加载评估函数

安装：pip install evaluate

**注意**：Evaluate is tested on **Python 3.7+**

```python
import evaluate
accuracy_metric = evaluate.load("accuracy")
accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])

evaluate.list_evaluation_modules("metric")   #这个可以查看有哪些评估函数
accuracy_metric.description   #有一些介绍
```



重要属性：

* description

* features



```python
#单一值计算
for ref, pred in zip([0,1,0,1], [1,0,0,1]):
    accuracy.add(references=ref, predictions=pred)
accuracy.compute()

#使用batch计算
for refs, preds in zip([[0,1],[0,1]], [[1,0],[0,1]]):
    accuracy.add_batch(references=refs, predictions=preds)
accuracy.compute()
```



### 计算多个指标

```python
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])
```



### 保存结果

```python
result = accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1])

hyperparams = {"model": "bert-base-uncased"}
evaluate.save("./results/"experiment="run 42", **result, **hyperparams)


{
    "experiment": "run 42",
    "accuracy": 0.5,
    "model": "bert-base-uncased",
    "_timestamp": "2022-05-30T22:09:11.959469",
    "_git_commit_hash": "123456789abcdefghijkl",
    "_evaluate_version": "0.1.0",
    "_python_version": "3.9.12 (main, Mar 26 2022, 15:51:15) \n[Clang 13.1.6 (clang-1316.0.21.2)]",
    "_interpreter_path": "/Users/leandro/git/evaluate/env/bin/python"
}

```



### Evaluator

只需要提供pipeline, dataset, metric ，不需要提供predictions



```python
from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator
import evaluate

pipe = pipeline("text-classification", model="lvwerra/distilbert-imdb", device=0)
data = load_dataset("imdb", split="test").shuffle().select(range(1000))
metric = evaluate.load("accuracy")
task_evaluator = evaluator("text-classification")

results = task_evaluator.compute(model_or_pipeline=pipe, data=data, metric=metric,
                       label_mapping={"NEGATIVE": 0, "POSITIVE": 1},)

print(results)

results = eval.compute(model_or_pipeline=pipe, data=data, metric=metric,
                       label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
                       strategy="bootstrap", n_resamples=200)

print(results)
```



### 画图

```python
import evaluate
from evaluate.visualization import radar_plot
data = [
model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
plot = radar_plot(data=data, model_names=model_names)
plot.show()
```



## Datasets

### 快速上手

安装

```sh
pip install datasets
```



```
from datasets import load_dataset
datasets = load_dataset("madao33/new-title-chinese")
```



加载的datasets一般是Datasetict，里面有一些字段，主要是split中的train,test等



### 数据映射

处理数据的每一条信息，常用于tokenize

```python
def add_prefix(example):
    example["title"] = 'Prefix: ' + example["title"]
    return example
prefix_dataset = datasets.map(add_prefix)
```



### 数据加载方式

1. 直接写hugging face中的名称
2. 将hugging face的内容下载到本地

```python
dataset = load_dataset('csv', data_files='my_file.csv')
```

3. 加载py文件

```python
dataset = load_dataset("path/to/local/loading_script/loading_script.py", split="train")
```



### 将自己数据上传到hugging face

结合git上传到data目录中，脚本文件放在主目录，需要保持文件名和数据集 的名称一致。
