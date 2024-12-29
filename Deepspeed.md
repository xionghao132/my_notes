# Deepspeed

## 概述

DeepSpeed的核心就在于，**GPU显存不够，CPU内存来凑**。

DeepSpeed将当前时刻，训练模型用不到的参数，缓存到[CPU](https://so.csdn.net/so/search?q=CPU&spm=1001.2101.3001.7020)中，等到要用到了，再从CPU挪到GPU。这里的“参数”，不仅指的是模型参数，还指optimizer、梯度等。

越多的参数挪到CPU上，GPU的负担就越小；但随之的代价就是，更为频繁的CPU，GPU交互，极大增加了训练推理的时间开销。因此，DeepSpeed使用的一个核心要义是，**时间开销和显存占用的权衡**。

[[LLM\]大模型训练DeepSpeed(一)-原理介绍-CSDN博客](https://blog.csdn.net/zwqjoy/article/details/130732601)

## ZERO

* ZeRO stage 1只对optimizer进行切片后分布式保存
* ZeRO stage 2对optimizer和grad进行切片后分布式保存
* ZeRO stage 3对optimizer、grad和模型参数进行切片后分布式保存



**zero2**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
 gradient_accumulation_steps: 1
 gradient_clipping: 1.0
 offload_optimizer_device: none
 offload_param_device: none
 zero3_init_flag: true
 zero_stage: 2
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```sh
accelerate launch examples/nlp_example.py --mixed_precision fp16
```



**zero3**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```sh
accelerate launch examples/nlp_example.py --mixed_precision fp16
```

