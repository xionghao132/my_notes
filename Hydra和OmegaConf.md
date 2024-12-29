# Hydra和OmegaConf

## Hydra概述

使用Hydra框架能高效配置各种超参数的配置。

多配置文件管理，动态命令行参数。

## 安装

```sh
pip install hydra-core
pip install omegaconf
```



## 简单使用



```python
# version_base用于选择Hydra在不同版本下的表现，不是很重要，具体请自行查阅https://hydra.cc/docs/upgrades/version_base/
# config_path表示配置文件所在路径
# config_name表示配置文件文件名，不包含后缀
@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")  
def my_app(cfg : DictConfig) -> None: 
	pass

#python my_app.py 运行会直接加载yaml里面的内容
```



也可以通过命令行覆盖配置中的值

```sh
python my_app.py train_setting.model=llama train_setting.n_epoches=4
```



## 嵌套使用

可以直接在`config.yaml`相同目录下创建目录`model`，此时配置文件选择的是`llama.yaml`

```
defaults:
	- model: llama
```



`__self__`代表了原始config.yaml文件的相对位置，

```yaml
# config.yaml
defaults:
    - train_setting: GCN 
    - _self_ 
# 此时n_epoches会被修改。但若_self_在train_setting: GCN那一行之前则不会

train_setting:
    GCN:
        n_epoches: 4
```



## 多次运行

```sh
python my_app.py --multirun db=mysql,postgresql schema=warehouse,support,school
```



## Logging

Hydra设置了python的logging，以方便使用。默认情况下Hydra会在控制台和日志文件中记录`INFO`级别的信息

```
import logging
# A logger for this file
log = logging.getLogger(__name__)
```

控制台和日志文件中都会有

> [YYYY-mm-dd HH:MM:SS,653][**main**][INFO] - Info level message

通过设置命令行的`hydra.verbose`可以记录`DEBUG`级别的信息

- `hydra.verbose=true`：将所有logger的级别设为`DEBUG`
- `hydra.verbose=NAME`：将`NAME`的logger的级别设为`DEBUG`
- `hydra.verbose=[NAME1,NAME2]`：将`NAME1`和`NAME2`的logger的级别设为`DEBUG`

通过命令行设置`hydra/job_logging=disabled`取消logging输出

通过命令行设置`hydra/job_logging=none`和`hydra/hydra_logging=none`取消Hydra配置logging

## OmegaConf概述

`OmegaConf` 提供了强大的配置管理工具，能够轻松地组织和访问配置信息，同时支持多种格式的配置文件。



## 简单使用

### 加载配置

```python
# 加载配置文件
config = OmegaConf.load("config.yaml")

# 将配置文件转换为 Python 字典
config_dict = OmegaConf.to_container(config, resolve=True)
```



### 合并配置

会覆盖原先的一些配置内容

```python
config = OmegaConf.merge(config, overrides)
```



### 命令行参数解析

`@` 符号将其标记为可从命令行接受的参数。



```python
import sys
from omegaconf import OmegaConf

# 解析命令行参数
overrides = OmegaConf.from_cli(sys.argv[1:])

# 将命令行参数与配置文件合并
config = OmegaConf.merge(config, overrides)

python my_script.py training.@epochs=20
```



### 嵌套继承

假设有一个名为 `base_config.yaml` 的基础配置文件：

```yaml
training:
  batch_size: 128
  learning_rate: 0.001
```

可以创建一个名为 `custom_config.yaml` 的配置文件，继承并覆盖基础配置：

```yaml
_base_: base_config.yaml
training:
  batch_size: 256
```

[一文看懂如何使用 Hydra 框架高效地跑各种超参数配置的深度学习实验 - 知乎](https://zhuanlan.zhihu.com/p/662221581)

[Python程序配置框架——Hydra（基于OmegaConf） - WangAaayu](https://wangaaayu.github.io/blog/posts/f5d8529f/)

[omegaconf，一个超强的 Python 库！ - 知乎](https://zhuanlan.zhihu.com/p/682336393)