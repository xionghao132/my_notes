# ollama

## 概述



## 安装



在线安装（这个有点慢，建议离线安装）

```
curl -fsSL https://ollama.com/install.sh | sh
```



离线安装

https://ollama.com/download/ollama-linux-amd64.tgz



将文件解压到其他目录，无法识别ollama，估计得放在环境变量中

```
tar -C /usr/local -xzf ollama-linux-amd64.tgz
```



## 常用操作



启动ollama

```
ollama serve 

ollama -v
```



```
ollama serve # 启动ollama
ollama create # 从模型文件创建模型
ollama show  # 显示模型信息
ollama run  # 运行模型，会先自动下载模型
ollama pull  # 从注册仓库中拉取模型
ollama push  # 将模型推送到注册仓库
ollama list  # 列出已下载模型
ollama ps  # 列出正在运行的模型
ollama cp  # 复制模型
ollama rm  # 删除模型
```



linux 模型的安装位置默认：Linux: /usr/share/ollama/.ollama/models



## OpenAI适配

```
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='llama3.2',
)

response = client.chat.completions.create(
    model="llava",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64,xxx",
                },
            ],
        }
    ],
    max_tokens=300,
)

completion = client.completions.create(
    model="llama3.2",
    prompt="Say this is a test",
)

list_completion = client.models.list()

model = client.models.retrieve("llama3.2")

embeddings = client.embeddings.create(
    model="all-minilm",
    input=["why is the sky blue?", "why is the grass green?"],
)
```

[ollama/docs/openai.md at main · ollama/ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md)



目前看到的都是一些量化模型，暂时不使用ollama部署，改成vllm试试。