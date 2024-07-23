# Wandb

## 概述

wandb是一个免费的，用于记录实验数据的工具。wandb相比于tensorboard之类的工具，有更加丰富的用户管理，团队管理功能，更加方便团队协作。使用wandb首先要在网站上创建team，然后在team下创建project，然后project下会记录每个实验的详细数据。



## 基本使用

```
pip install wandb
wandb login

import wandb
wandb.init(config=all_args,
               project=your_project_name,
               entity=your_team_name,
               notes=socket.gethostname(),
               name=your_experiment_name
               dir=run_dir,
               job_type="training",
               reinit=True)
```



```
def my_train_loop():
    for epoch in range(10):
        loss = 0 # change as appropriate :)
        wandb.log({'epoch': epoch, 'loss': loss})

```



### lm-evaluation-harness

```
lm_eval \
    --model hf \
    --model_args pretrained=microsoft/phi-2,trust_remote_code=True \
    --tasks hellaswag,mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/phi-2 \
    --limit 10 \
    --wandb_args project=lm-eval-harness-integration \   #加入这样的一个参数即可
    --log_samples
```



### hugging face

```
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # turn on W&B logging
trainer = Trainer(..., args=args)
#其他一些可选参数
args = TrainingArguments(
    # other args and kwargs here
    report_to="wandb",  # enable logging to W&B
    run_name="bert-base-high-lr",  # name of the W&B run (optional)
    logging_steps=1,  # how often to log to W&B
)
```

