# Hugging Face



## 安装

```sh
pip install transformers
```



## 模型

### Pipeline

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")  # 情感分析
classifier("I've been waiting for a HuggingFace course my whole life.")

# 输出
# [{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

目前可用的一些pipeline

- `feature-extraction` 特征提取：把一段文字用一个向量来表示
- `fill-mask` 填词：把一段文字的某些部分mask住，然后让模型填空
- `ner` 命名实体识别：识别文字中出现的人名地名的命名实体
- `question-answering` 问答：给定一段文本以及针对它的一个问题，从文本中抽取答案
- `sentiment-analysis` 情感分析：一段文本是正面还是负面的情感倾向
  `summarization` 摘要：根据一段长文本中生成简短的摘要
- `text-generation`文本生成：给定一段文本，让模型补充后面的内容
  `translation` 翻译：把一种语言的文字翻译成另一种语言
- `zero-shot-classification`：可以输入自己的标签，然后模型输出最有可能的标签，相当于迁移。

更多介绍：[Transformers, what can they do? - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt)



### Tokenizer

**AutoTokenizer**是通用封装，根据载入预训练模型来自适应。

```python
#pretrained_model_path 可以是本地存储模型的目录,也可以是模型名字，这样就会将文件进行缓存
auto_tokenizer=transformers.AutoTokenizer.from_pretrained(config.pretrained_model_path) 


sequence = "Using a Transformer network is simple"
tokenizer.tokenize(sequence)
#model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

# 输出 : ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
ids = tokenizer.convert_tokens_to_ids(tokens)
# 输出：[7993, 170, 11303, 1200, 2443, 1110, 3014]
tokenizer(sequence)

'''
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
'''

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
```

1. 修改环境变量`Path`中**TRANSFORMERS_CACHE**
2. 加入缓存参数`cache_dir='./'`

#### **参数**

- `text (str, List[str], List[List[str]])`：就是输入的待编码的序列（或1个batch的），可以是字符串或字符串列表。
- `text_pair (str, List[str], List[List[str]])`：输入待编码的序列对，也就是第二个序列。
- `add_special_tokens(bool, optional, defaults to True) `：True就是给序列加上特殊符号，如[CLS],[SEP]
- `padding (Union[bool, str], optional, defaults to False) `：给序列补全到一定长度，True or ‘longest’: 是补全到batch中的最长长度，max_length’:补到给定max-length或没给定时，补到模型能接受的最长长度。
- `truncation (Union[bool, str], optional, defaults to False) `：截断操作，true or ‘longest_first’：给定max_length时，按照max_length截断，没给定max_lehgth时，到，模型接受的最长长度后截断，适用于所有序列（单或双）。only_first’：这个只针对第一个序列。only_second’：只针对第二个序列。
- `max_length (Union[int, None], optional, defaults to None) `：控制padding和truncation的长度。
- `stride (int, optional, defaults to 0) `：和max_length一起使用时，用于标记截断和溢出的重叠数量（不知道怎么用）。
- `is_pretokenized (bool, defaults to False)`：表示这个输入是否已经被token化。
- `pad_to_multiple_of `：将序列以倍数形式padding
- `return_tensors (str, optional, defaults to None)`：返回数据的类型，可选tf’, ‘pt’ or ‘np’ ，分别表示tf.constant, torch.Tensor或np.ndarray
- `return_token_type_ids (bool, optional, defaults to None)`：默认返回token_type_id（属于哪个句子）。
- `return_attention_mask (bool, optional, defaults to none)`：默认返回attention_mask（是否参与attention计算）。
- `return_overflowing_tokens (bool, optional, defaults to False)`：默认不返回溢出的token
- `return_special_tokens_mask (bool, optional, defaults to False) `：默认不返回特殊符号的mask信息.

#### **返回值**

```python
{
    input_ids: list[int],
    token_type_ids: list[int] if return_token_type_ids is True (default)
    attention_mask: list[int] if return_attention_mask is True (default)
    overflowing_tokens: list[int] if the tokenizer is a slow tokenize, else a List[List[int]] if a ``max_length`` is specified and ``return_overflowing_tokens=True``
    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``and return_special_tokens_mask is True
}
```

#### 代码

```python
from transformers import AutoTokenizer  #还有其他与模型相关的tokenizer，如BertTokenizer

tokenizer=AutoTokenizer.from_pretrained('bert-base-cased') #这里使用的是bert的基础版（12层），区分大小写，实例化一个tokenizer

batch_sentences=["Hello I'm a single sentence","And another sentence","And the very very last one"]

batch=tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")

#inputs_ids是映射，token_type_ids是否是成对句子，attention_mask是否参与attention计算
{'input_ids':tensor([[101,8667,146,112,182,170,1423,5650,102],[101,1262,1330,5650,102,0,0,0,0],[101,1262,1103,1304,1304,1314,1141,102,0]]),'token_type_ids':tensor([[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]),'attention_mask':tensor([[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0]])}
```

[huggingface使用（一）：AutoTokenizer（通用）、BertTokenizer（基于Bert）-CSDN博客](https://blog.csdn.net/u013250861/article/details/124535020)

### **Model**

`Transformers` 提供了一个`AutoModel`类，它也有一个`from_pretrained()`方法：

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = 

(**inputs)
print(outputs.last_hidden_state.shape)
print("Logits:", output.logits)

inputs={
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}

```



### Post-Processing

模型最后一层输出的原始**非标准化分数**。要转换为概率，它们需要经过一个`SoftMax`层（所有` Transformers `模型都输出` logits`，因为用于训练的损耗函数一般会将最后的激活函数(如`SoftMax`)与实际损耗函数(如交叉熵)融合 。

```python
import torch

predictions=torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```



## Fine-tuning

**Transformers**提供了数以千计针对于各种任务的预训练模型模型，开发者可以根据自身的需要，选择模型进行训练或微调，也可阅读api文档和源码， 快速开发新模型。

### 处理数据

* 数据为`batch`的`classifier`

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

* `Trainer`

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train() 
```

* 评估

```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

