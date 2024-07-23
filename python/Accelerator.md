# Accelerator

## 概述

现在很多代码中使用的都是分布式并行计算，需要定义一系列的参数，显得非常复杂。而HuggingFace的Accelerate就能很好的解决这个问题，只需要在平时用的DataParallel版代码中修改几行，就能实现多机多卡、单机多卡的分布式并行计算，另外还支持FP16半精度计算。



## 例子

```diff
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

- device = 'cpu'
+ accelerator = Accelerator()

- model = torch.nn.Transformer().to(device)
+ model = torch.nn.Transformer()
  optimizer = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optimizer, data = accelerator.prepare(model, optimizer, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
-         source = source.to(device)
-         targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
```



移除代码中所有的**to(device)**或者**cuda()**,accelerator会自动帮你处理。如果你知道自己的想法，也可以调用**to(accelerator.device)**

和训练相关的对象都要传递到**prepare**方法中(如果要分布式验证的话，val_dataloader也需要)



## 部署分布式脚本

在开始训练前，我们还需要配置下accelerate的脚本。当然我们也可以用**torch.distributed.launch**，但是要加上**--use_env**



使用脚本的话

```sh
accelerate config
```



会根据你回答的问题生成一个yaml文件，我的位于~/.cache/huggingface/accelerate

HF_HOME是hugging face的路径

然后运行

```text
accelerate test
accelerate test --config_file path_to_config.yaml
```



检查配置项

```
accelerate env
```



来测试脚本能否正常工作。一切都ok后，我们就能开始训练了：

```text
accelerate launch path_to_script.py --args_for_the_script
accelerate launch --config_file path_to_config.yaml path_to_script.py --args_for_the_script
```

**保存模型：**

```python
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), path)
```

**加载模型**：(utils.py中)

```python
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.load_state_dict(torch.load(path))

```

要用tqdm的话需要：

```python
from tqdm.auto import tqdm
for epoch in tqdm(range(args.epochs), disable=not accelerator.is_local_main_
```



### gradient accumulation

```python
accelerator = Accelerator(gradient_accumulation_steps=2)
model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

for input, label in training_dataloader:
    with accelerator.accumulate(model):
        predictions = model(input)
        loss = loss_function(predictions, label)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```



## 加载大模型



```python
model_critic=model_critic.to('cpu')
torch.save(model_critic.state_dict(), 'model/critic')

from accelerate import init_empty_weights,load_checkpoint_and_dispatch
with init_empty_weights():
    model_reward=CriticModel()
    model_reward = load_checkpoint_and_dispatch(
    model_reward, checkpoint='model/critic', device_map="auto")
```



[huggingface/accelerate: 🚀 A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision (github.com)](https://github.com/huggingface/accelerate)

[Quick tour (huggingface.co)](https://huggingface.co/docs/accelerate/quicktour)