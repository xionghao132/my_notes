# vLLM

## 概述

vLLM是一个快速且易于使用的LLM推理和服务库。vLLM内部使用的是`pageAttention`加速的，类似操作系统那样按页分陪缓存，以及共享缓存，针对`KV cache`的。

## 安装

```sh
# (Recommended) Create a new conda environment.
conda create -n myenv python=3.12 -y
conda activate myenv

# Install vLLM with CUDA 12.1.
pip install vllm  #不建议这么使用，我在曙光机器安装不了


#可以直接在release中去找对应的编译版本 然后安装
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

[Installation — vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html)

## 快速使用

```python
from vllm import LLM
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

llm = LLM(model="Qwen/Qwen-7B",trust_remote_code=True,gpu_memory_utilization=0.9) 

outputs = llm.generate(prompts)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```



## API

启动服务器，默认情况http://localhost:8000，当然也可以加入`host` 和`port`去指定地址

这个是5.2版本之前方法

```python
python -m vllm.entrypoints.openai.api_server --trust-remote-code --model Qwen/Qwen-7B
```



5.2版本之后可以使用这个

```
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

可以使用`chat-template`参数覆盖原有的聊天模板

```python
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen-7B \
--chat-template ./examples/template_chatml.jinja
```

查询

```sh
curl http://localhost:8000/v1/models
```

在线调用

```sh
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen/Qwen-7B",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'
```

