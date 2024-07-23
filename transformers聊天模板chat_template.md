# transformers聊天模板chat_template

## 概述

LLMs的一个越来越常见的用例是聊天。在聊天上下文中，模型不是继续单个文本字符串（就像标准语言模型一样），
而是继续由一个或多个消息组成的对话，每个消息都包括一个角色，比如“用户”或“助手”，以及消息文本。

与Tokenizer类似，不同的模型对聊天的输入格式要求也不同。这就是我们添加聊天模板作为一个功能的原因。
聊天模板是Tokenizer的一部分。用来把问答的对话内容转换为模型的输入prompt。

## 例子

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

chat = [
    {"role": "system", "You are a helpful assistant."},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
     {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
 ]

tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"

#output
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
I'm doing great. How can I help you today?<|im_end|>


```



返回`tensor input_ids`

```python
#默认设置的`tokenize=True`，那么该字符串也将被tokenized处理
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
```



```python
add_generation_prompt=True   #相当于提示开始

#output
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
I'm doing great. How can I help you today?<|im_end|>
<|im_start|>assistant
```

并非所有模型都需要生成提示。一些模型，如BlenderBot和LLaMA，在模型回复之前没有任何特殊标记。
在这些情况下，`add_generation_prompt`参数将不起作用。`add_generation_prompt`参数取决于你所使用的模板。



`apply_chat_template`主要是用于推理



模型的聊天模板存储在`tokenizer.chat_template`属性上



## 训练

`qwen-chat`，注意`im_start`  `im_end`并没有`mask`

```python
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
system_message: str = "You are a helpful assistant."
im_start = tok.im_start_id
im_end = tok.im_end_id
nl_tokens = tok('\n').input_ids
_system = tok('system').input_ids + nl_tokens
_user = tok('user').input_ids + nl_tokens
_assistant = tok('assistant').input_ids + nl_tokens
# Apply prompt templates
new_input_ids, new_targets = [], []
system = [im_start] + _system + tok(system_message).input_ids + [im_end] + nl_tokens  #<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
new_input_ids += system
new_targets += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens  #对应的token targets
assert len(new_input_ids) == len(new_targets)
source = [
# {"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": txt[0]},
{"role": "assistant", "content": tgt[0]},
]
for j, sentence in enumerate(source):
    role = roles[sentence["role"]]
    _input_id = tok(role).input_ids + nl_tokens + \
        tok(sentence["content"]).input_ids + [im_end] + nl_tokens
    new_input_ids += _input_id
    if role == '<|im_start|>user':
        _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
    elif role == '<|im_start|>assistant':
        _target = [im_start] + [IGNORE_TOKEN_ID] * len(tok(role).input_ids) + \
            _input_id[len(tok(role).input_ids)+1:-2] + [im_end] + nl_tokens
    else:
        raise NotImplementedError
    new_targets += _target
new_input_ids=torch.tensor(new_input_ids,device=device)
new_targets=torch.tensor(new_targets,device=device)
tokens= dict(
    input_ids=new_input_ids,
    labels=new_targets,
    attention_mask=new_input_ids.ne(tok.pad_token_id),
)
```

[Qwen/finetune.py at main · QwenLM/Qwen (github.com)](https://github.com/QwenLM/Qwen/blob/main/finetune.py#L125)



[如何设置transformers的聊天模板chat_template？_transformers chat-CSDN博客](https://blog.csdn.net/mingzai624/article/details/135952802)

