# Torch.gather

## 函数定义：

`torch.gather`(*input*, *dim*, *index*, *, *sparse_grad=False*, *out=None*) → Tensor

## 参数：

- **input** (Tensor) – 源tensor
- **dim** (int)) –索引轴
- **index** (LongTensor) – 包含索引元素下标的tensor

`dim`表示维度，二维中`dim=0`就是行，`dim=1`就是列。我们将其推广，比如一个矩阵维度如下：$(d_0,d_1,…,d_{n-1}) $，那么`dim=0`就表示对应到$d_0$ 也就是第一个维度，`dim=1`表示第二个维度。

## gather规则

```python
out[i][j][k] = input[ index[i][j][k] ][j][k]  # if dim == 0
out[i][j][k] = input[i][ index[i][j][k] ][k]  # if dim == 1
out[i][j][k] = input[i][j][ index[i][j][k] ]  # if dim == 2
```

从这个规则可以看出，输出的内容就是替换对应的轴的索引就行。

## 例子

```python
t = torch.tensor([[1, 2], [3, 4]])
torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
#tensor([[ 1,  1],[ 4,  3]])
```

解释：

```python
dim=1  #替换第二个轴的索引
output[0][0]=t[0][index[0][0]]=1
output[0][1]=t[0][index[0][1]]=1
output[1][0]=t[1][index[1][0]]=4
output[1][1]=t[1][index[1][1]]=3
```

