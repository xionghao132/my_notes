# Tensorboard



## 概述



## 安装

```sh
pip install tensorboard
```



## Pytorch_lightning访问Tensorbboard

```python
from pytorch_lightning.loggers import TensorBoardLogger
def training_step(self, batch, batch_idx):
    self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
logger = TensorBoardLogger('tb_logs', name='my_model')
# train
model = LightningMNISTClassifier(outdim=outdim)
trainer = pl.Trainer(gpus=None, max_epochs=2, logger=logger)
trainer.fit(model)
tensorboard --logdir ./tb_logs

```



## 显示服务器中的Tensorboard

将服务器中`Tensorboard`的端口`6006`映射到本地端口`16006`，在终端中输入：
`ssh -L 16006:127.0.0.1:6006 用户名@服务器ip -p 22`

`-L`参数用于创建本地端口转发（Local Port Forwarding）。它允许将本地端口与远程服务器的端口之间建立一个安全的通信通道。

具体来说，`-L`参数的语法为`-L [本地地址:]本地端口:目标地址:目标端口`。其中，本地地址是可选的，默认为`localhost`。



激活服务器端`python`环境，在终端中运行：`tensorboard --logdir={tensorboard文件位置}`
在本地服务器输入：http://localhost:16006

```sh
ssh -L 16006:localhost:6006 xhao@192.168.134.23 -p 22
```

