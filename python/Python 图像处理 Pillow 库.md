[TOC]

# Python 图像处理 Pillow 库

## 概述

 	图像处理是常用的技术，python 拥有丰富的第三方扩展库，Pillow 是 Python3 最常用的图像处理库，目前最高版本5.2.0。Python2 使用Pil库，两者是使用方法差不多，区别在于类的引用不同。

> ==注意：==Pil 库与 Pillow 不能同时存在与一个环境中，如果你已经安装 Pil 库，那么请将他卸载。

​	使用 pip 安装 Pillow：

```sh
pip install Pillow
```

## 创建实例

### 通过文件创建Image对象

```python
from PIL import Image
image = Image.open('python-logo.png')  # 创建图像实例
# 查看图像实例的属性
print(image.format, image.size, image.mode) # format 图像格式JPEG  size 宽高 mode 默认 RGB 真彩图像
image.show() # 显示图像
```

### 从打开文件中读取

 	可以从文件对象读取而不是文件名，但文件对象必须实现 read( ) ，seek( ) 和 tell( ) 方法，并且是以二进制模式打开。

```python
from PIL import Image
with open("hopper.ppm", "rb") as fp:
    im = Image.open(fp)
```

==注意：==在读取图像 header 之前将文件倒回（使用 seek(0) ）。

### 从tar文件中读取

```python
from PIL import TarIO

fp = TarIO.TarIO("Imaging.tar", "Imaging/test/lena.ppm")
im = Image.open(fp)
```

## 读写图像

### 格式转换并保存文件

 	Image 模块中的 save 函数可以保存图片，除非你指定文件格式，否则文件的扩展名就是文件格式。

```python
import os
from PIL import Image

image_path='love.png' # 图片位置
f, e = os.path.splitext(image_path) # 获取文件名与后缀
outfile = f + ".jpg"
if image_path != outfile:
    try:
        Image.open(image_path).save(outfile) # 修改文件格式
    except IOError:
        print("cannot convert", image_path)
```

==注意：== 如果你的图片mode是RGBA那么会出现异常,因为 RGBA 意思是红色，绿色，蓝色，Alpha 的色彩空间，Alpha 是指透明度。而 JPG 不支持透明度 ，所以要么丢弃Alpha , 要么保存为.png文件。解决方法将图片格式转换：

```python
Image.open(image_path).convert("RGB").save(outfile)  # convert 转换为 RGB 格式，丢弃Alpha
```

> save() 函数有两个参数，如果文件名没有指定图片格式，那么第二个参数是必须的，他指定图片的格式。

### 创建缩略图

 	创建缩略图 使用 Image.thumbnail( size ), size 为缩略图宽长元组。

```python
import os
from PIL import Image

image_path = 'love.png'  # 图片位置
size = (128, 128)  # 文件大小
f, e = os.path.splitext(image_path)  # 获取文件名与后缀
outfile = f + ".thumbnail"
if image_path != outfile:
    try:
        im = Image.open(image_path)
        im.thumbnail(size)  # 设置缩略图大小
        im.save(outfile, "JPEG")
    except IOError:
        print("cannot convert", image_path)
```

==注意：== 出现异常，同上一个示例，convert("RGB")转换图片mode。

==注意：==除非必须，Pillow不会解码或栅格数据。当你打开文件，Pillow通过文件头确定文件格式，大小，mode等数据，余下数据直到需要时才处理。这意味着打开文件非常快速，它与文件大小和压缩格式无关。

### 剪贴，粘贴，合并图像

 	Image类包含允许您操作图像中的区域的方法。如：要从图像中复制子矩形图像使用 crop() 方法。

#### 从图像复制子矩阵

```python
box = (100, 100, 400, 400)
region = im.crop(box)
```

 	定义box元组，表示图像基于左上角为（0,0）的坐标，box 坐标为 (左，上，右，下）。注意，坐标是基于像素。示例中为 300 * 300 像素。

#### 处理子矩阵并且粘贴回来

```python
region = region.transpose(Image.ROTATE_180) # 颠倒180度
box = (400, 400, 700, 700)  # 粘贴位置，像素必须吻合，300 * 300
im.paste(region, box)
```

==注意：==将子图（region） 粘贴（paste）回原图时，粘贴位置 box 的像素与宽高必须吻合。而原图和子图的 mode 不需要匹配，Pillow会自动处理。

#### 滚动图像

```python
from PIL import Image


def roll(image, delta):
    """ 向侧面滚动图像 """
    xsize, ysize = image.size

    delta = delta % xsize
    if delta == 0: return image

    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    image.paste(part1, (xsize - delta, 0, xsize, ysize))
    image.paste(part2, (0, 0, xsize - delta, ysize))

    return image


if __name__ == '__main__':
    image_path = 'test.jpg'
    im = Image.open(image_path)
    roll(im, 300).show()  # 向侧面滚动 300 像素
```

#### 分离和合并通道

 	Pillow 允许处理图像的各个通道，例如RGB图像有R、G、B三个通道。 split 方法分离图像通道，如果图像为单通道则返回图像本身。merge 合并函数采用图像的 mode 和 通道元组为参数，将它们合并成新图像。

```python
r, g, b = im.split()
im = Image.merge("RGB", (b, g, r))
```

==注意：==如果要处理单色系，可以先将图片转换为’RGB‘

### 几何变换

​	 PIL.Image 包含调整图像大小 resize() 和旋转 rotate() 的方法。前者采用元组给出新的大小，后者采用逆时针方向的角度。

```python
out = im.resize((128, 128))  #注意传入的参数是元组
out = out.rotate(45)   #旋转45度
```

 	要以90度为单位旋转图像，可以使用 rotate() 或 transpose() 方法。后者也可用于围绕其水平轴或垂直轴翻转图像。

```python
out = im.transpose(Image.FLIP_LEFT_RIGHT) # 水平左右翻转
out = im.transpose(Image.FLIP_TOP_BOTTOM) # 垂直上下翻转
out = im.transpose(Image.ROTATE_90) # 逆时针90度
out = im.transpose(Image.ROTATE_180) # 逆时针180度
out = im.transpose(Image.ROTATE_270) # 逆时针270度
```

### 颜色变换

```python
from PIL import Image
im = Image.open("hopper.ppm").convert("L") # 转换为灰阶图像
```

==注意：==它支持每种模式转换为"L" 或 "RGB"，要在其他模式之间进行转换，必须先转换模式（通常为“RGB”图像）。

### 图像增强

#### Filter过滤器

```python
from PIL import ImageFilter
out = im.filter(ImageFilter.DETAIL)
```

#### 像素点处理

 	point() 方法可用于转换图像的像素值（如对比度），在大多数情况下，可以将函数对象作为参数传递格此方法，它根据函数返回值对每个像素进行处理。

```python
out = im.point(lambda i: i * 1.2) #每个像素点扩大1.2倍
```

```python
pix=im.load()  #将图片分成小像素方块
# 获取图片大小
width = img.size[0]
height = img.size[1]
for x in range(0,width):
    for y in range(0,height):
        rgb = pix[x,y]      # 获取一个像素块的rgb
        r, g, b = pix[x, y]
        if b>130 and r<120: # 自定义某些规则
            pix[x, y] = (255, 0, 0) # 修改单个像素点
```

#### 处理单独通道

```python
# 将通道分离
source = im.split()

R, G, B = 0, 1, 2

# 选择红色小于100的区域
mask = source[R].point(lambda i: i < 100 and 255)

# 处理绿色
out = source[G].point(lambda i: i * 0.7)

# 粘贴已处理的通道，红色通道仅限于<100
source[G].paste(out, None, mask)

# 合并图像
im = Image.merge(im.mode, source)
```

### 高级增强

 	其他图像增强功能可以使用 ImageEnhance 模块中的类。从图像创建后，可以使用 ImageEnhance 快速调整图片的对比度、亮度、饱和度和清晰度。

```python
from PIL import ImageEnhance

enh = ImageEnhance.Contrast(im)  # 创建调整对比度对象
enh.enhance(1.3).show("增加30%对比度")
```

ImageEnhance 方法类型：

1. ImageEnhance.Contrast(im) 对比度
2. ImageEnhance.Color(im) 色彩饱和度
3. ImageEnhance.Color(im) 色彩饱和度
4. ImageEnhance.Sharpness(im) 清晰度

### 动态图像

​	 Pillow 支持一些动态图像处理（如FLI/FLC，GIF等格式）。TIFF文件同样可以包含数帧图像。

​	打开动态图像时，PIL 会自动加载序列中的第一帧。你可以使用 seek 和 tell 方法在不同的帧之间移动。

```python
from PIL import Image

im = Image.open("animation.gif")
im.seek(1) # 跳到第二帧  从0开始

try:
    while 1:
        im.seek(im.tell()+1)  # tell() 获取当前帧的索引号
except EOFError: # 当读取到最后一帧时，Pillow抛出EOFError异常。
    pass # 结束
```

==注意：==有些版本的库中的驱动程序仅允许您搜索下一帧。要回放文件，您可能需要重新打开它。都遇到无法回放的库时，可以使用 for 语句循环实现。

示例：for 使用 ImageSequence Iterator 类遍历动态图像

```python
from PIL import ImageSequence
for frame in ImageSequence.Iterator(im):
    # ...处理过程...
```
> 保存动态图像
```python
im.save(out, save_all=True, append_images=[im1, im2, ...])
```

参数说明：

> **out** 需要保存到那个文件
> **save_all** 为True，保存图像的所有帧。否则，仅保存多帧图像的第一帧。
> **append_images** 需要附加为附加帧的图像列表。列表中的每个图像可以是单帧或多帧图像（ 目前只有GIF，PDF，TIFF和WebP支持此功能）。

### Postscript 打印

 	Pillow 允许通过 Postscript Printer 在图片上添加图像或文字。

```python
from PIL import Image
from PIL import PSDraw

im = Image.open("test.jpg")
title = "hopper"
box = (1*72, 2*72, 7*72, 10*72) # in points

ps = PSDraw.PSDraw() # 默认 sys.stdout
ps.begin_document(title)

# 画出图像 (75 dpi)
ps.image(box, im, 75)
ps.rectangle(box)

# 画出标题
ps.setfont("HelveticaNarrow-Bold", 36)
ps.text((3*72, 4*72), title)
```



## opencv与PIL转化

* opencv > pil

```python
import cv2  
from PIL import Image
img = cv2.imread("test.png")
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
```

* pil > opencv

```python
import cv2  
from PIL import Image 
image = Image.open("test.png")  
img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
```

