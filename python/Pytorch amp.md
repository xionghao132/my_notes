# Pytorch amp

## 概述

**自动混合精度**在训练一个FP32的模型的时候，一部分算子的操作时，数值精度为 FP16，其余算子的操作精度是 FP32，而具体哪些算子用 FP16，哪些用 FP32，不需要用户关心，amp 自动给它们都安排好了。

**优点**：是缩短训练时间，减低存储需求

**缺点**：会出现上溢出和下溢出。



通过使用torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度的underflow，只在BP时传递梯度信息使用，真正更新权重时还是要把放大的梯度再unscale回去



混合精度计算 enable 区域得到的 FP16 数值精度的变量在 enable 区域外需要显式的转成 FP32



## 代码

### autocast

* 例子

```python
#进入autocast上下文之后，会将对应的cuda上的tensor数据类型转化为半精度。
with autocast():
    output=model(input)
    loss = loss_fn(output,target)
    
    
#多GPU,也需要autocast装饰model的forward方法，保证autocast在进程内部生效。

#alternatively
MyModel(nn.Module):
    def forward(self, input):
        with autocast():
            
#也可以写成注释
MyModel(nn.Module):
    ...
    @autocast()
    def forward(self, input):
       ...
```



```python
from torch.cuda.amp import autocast as autocast,GradScaler
# amp依赖Tensor core架构，所以model参数必须是cuda tensor类型
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)
# GradScaler对象用来自动做梯度缩放
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        # 在autocast enable 区域运行forward
        with autocast():
            # model做一个FP16的副本，forward
            output = model(input)
            loss = loss_fn(output, target)
        # 用scaler，scale loss(FP16)，backward得到scaled的梯度(FP16)
        scaler.scale(loss).backward()
        # scaler 更新参数，会先自动unscale梯度
        # 如果有nan或inf，自动跳过
        scaler.step(optimizer)
        # scaler factor更新
        scaler.update()
```

* 嵌套使用

```python
# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")
c_float32 = torch.rand((8, 8), device="cuda")
d_float32 = torch.rand((8, 8), device="cuda")

with autocast():
    e_float16 = torch.mm(a_float32, b_float32)

    with autocast(enabled=False):

        f_float32 = torch.mm(c_float32, e_float16.float())

    g_float16 = torch.mm(d_float32, f_float32)
```



* autocast自定义函数

```python
class MyMM(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)
```

调用时再 autocast

```python
mymm = MyMM.apply

with autocast():
    output = mymm(input1, input2)
```

### GradScaler

* Gradient clipping

```python
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()        #放大梯度

        # unscale 梯度，可以不影响clip的threshold
        scaler.unscale_(optimizer)

        # clip梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # unscale_（）已经被显式调用了，scaler正常执行step更新参数，有nan/inf也会跳过
        scaler.step(optimizer)  #没有显式调用的时候，会自己调用
        scaler.update()  # 准备着，看是否要增大scaler
```

* Gradient accumulation

```python
scaler = GradScaler()

for epoch in epochs:
    for i, (input, target) in enumerate(data):
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
            # loss 根据 累加的次数归一一下
            loss = loss / iters_to_accumulate

        # scale 归一的loss 并backward  
        scaler.scale(loss).backward()

        if (i + 1) % iters_to_accumulate == 0:
            # may unscale_ here if desired 
            # (e.g., to allow clipping unscaled gradients)

            # step() and update() proceed as usual.
            scaler.step(optimizer)      
            scaler.update()            
            optimizer.zero_grad()
```

* Gradient penalty

```python
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
        # 防止溢出，在不是autocast 区域，先用scaled loss 得到 scaled 梯度
        scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                 inputs=model.parameters(),
                                                 create_graph=True)
        # 梯度unscale
        inv_scale = 1./scaler.get_scale()
        grad_params = [p * inv_scale for p in scaled_grad_params]
        # 在autocast 区域，loss 加上梯度惩罚项
        with autocast():
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

        scaler.scale(loss).backward()

        # may unscale_ here if desired 
        # (e.g., to allow clipping unscaled gradients)

        # step() and update() proceed as usual.
        scaler.step(optimizer)
        scaler.update()
```

* Multiple models

scaler 一个就够，但 scale(loss) 和 step(optimizer) 要分别执行

```python
scaler = torch.cuda.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with autocast():
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn(2 * output0 + 3 * output1, target)
            loss1 = loss_fn(3 * output0 - 5 * output1, target)

        # (retain_graph here is unrelated to amp, it's present because in this
        # example, both backward() calls share some sections of graph.)
        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()

        # You can choose which optimizers receive explicit unscaling, if you
        # want to inspect or modify the gradients of the params they own.
        scaler.unscale_(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()
```

[Pytorch自动混合精度的计算：torch.cuda.amp.autocast-CSDN博客](https://blog.csdn.net/lsb2002/article/details/134399785)





[PyTorch 源码解读之 torch.cuda.amp: 自动混合精度详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/348554267)