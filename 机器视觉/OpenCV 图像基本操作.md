[TOC]

# OpenCV 图像基本操作

## 图像读取

### 读取图像

opencv读取的格式是BGR

* cv2.IMREAD_COLOR：彩色图像 （1）彩色图像就相当于三维数组
* cv2.IMREAD_GRAYSCALE：灰度图像（0）灰度图像相当于二维图像

>魔法函数%matplotlib inline 在jupyter展示中,图绘制完了直接进行展示,不用plt.show()显示

```python
import cv2 # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np 
%matplotlib inline 

img = cv2.imread('sky.jpg')

```


### 图像显示
```python
# 图像的显示,也可以创建多个窗口
cv2.imshow('image',img) 
# 等待时间，毫秒级，0表示任意键终止。 如果设定1000，大概10s就销毁窗口
cv2.waitKey(0) 
cv2.destroyAllWindows()
```

>注意这里图像img都是narray形式，所以有这些数组的基础性质
* shape
* size
* dtype

### 图像保存

```python
cv2.imwrite('cat_gray.jpg',img_gray)
```

### 截取部分图像的数据

  感兴趣区域（ROI）的截取 [x:x+w,y:y+h]
```python
img = cv2.imread('cat.jpg')
img_crop = img[100:200,50:200]
cv2.imshow('cat_crop',img_crop)

```

### 颜色通道提取、组合
* split 分离颜色通道
* merge 组合颜色通道(注 两个括号)
```python
import cv2
b,g,r = cv2.split(img)	# cv2.split(img)[0] 等同于 img[:,:,0]取第一个通道
b[100:200,50:200] = 0
img_new = cv2.merge((b,g,r))	# 处理完某个通道，再重新组合b g r3个通道
cv2.imshow('img_new',img_new)
cv2.waitKey(0)

```
### 边界填充
​		扩充图像边界 copyMakeBorder，需要6个参数：图+上+下+左+右(填充的像素大小)+填充方式
填充方式如下：

* BORDER_REPLICATE：复制法，也就是复制最边缘像素。
* BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：* fedcba|abcdefgh|hgfedcb
* BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴(没有a)，对称，gfedcb|abcdefgh|gfedcba
* BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
* BORDER_CONSTANT：常量法，常数值填充。

```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('sky1.jpg')
top_size,bottom_size,left_size,right_size = (50,50,50,50)
img_replace = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REPLICATE)
img_reflect = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)
img_reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
img_wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
img_constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
# cv_show('img_wrap',img_wrap)
plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(img_replace,'gray'),plt.title('Replace')
plt.subplot(233), plt.imshow(img_reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(img_reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(img_wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(img_constant, 'gray'), plt.title('CONSTANT')
plt.savefig('cat_fill.jpg')
plt.show()
```
![image-20211029160437336](https://gitee.com/HB_XN/picture/raw/master/img/20211208191657.png)

>==注意：== plt.subplot(行列索引)  最后要使用plt.show()

### 数值计算

&emsp; &emsp; np+,超出255取余

​		 cv2.add(img1,img2) 数值超出255就选255

### 图像融合

- cv2.resize(img, (新img的宽, 高)) 融合两张图的前提是尺寸一致

```python
import cv2
img_cat = cv2.imread('cat.jpg')  # 414x500x3
img_dog = cv2.imread('dog.jpg')
# 普通resize
img_dog = cv2.resize(img_dog,(img_cat.shape[1],img_cat.shape[0]))
cat_dog = img_cat+img_dog
cv2.imshow('cat_dog',cat_dog)
```

- cv2.addWeighted 就相当于α * X1 + β * X2 + b，α=0.4，β=0.6，分别是两张图片的权重，以这样的形式融合

```python
res = cv2.addWeighted(img_cat,0.4,img_dog,0.6,0)
cv_show('res',res)
```

## 视频读取

### 读取视频

- cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备，例如0,1。
- 如果是视频文件，直接指定好路径即可。

```python
import cv2
vc = cv2.VideoCapture('test.mp4')
# 检查是否正确打开
if vc.isOpened():#opened 表明是否读取到文件末尾
    opened, frame = vc.read()  # 此时frame中存储的是视频的第一帧图片
    cv2.imshow('frame',frame) #frame是narray数组
else:
    open = False	# 第一感觉是,是不是写错了,但还就是open
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY) #颜色空间转换
        cv2.imshow('result', gray)
        if cv2.waitKey(100) & 0xFF == 27:
            break
vc.release()	#关闭相机
cv2.destroyAllWindows()	#关闭窗口

```

- cv2.waitKey(100) 隔多少毫秒显示下一张图片，设置稍大点，符合我们看视频的一个速度。太大就像看视频卡顿的感觉；太小就像几倍速播放，太快了。
- 0xFF == 27 指定退出键退出
- 0xFF == ord(‘q’) 指定q键退出

## 图像处理

### 图像阈值

ret, dst = cv2.threshold(src, thresh, maxval, type)

* src： 输入图，只能输入单通道图像，通常来说为灰度图
* thresh：一般取127和255
* maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
* type：二值化操作的类型，包含以下5种类型：
  * cv2.THRESH_BINARY 超过阈值部分取maxval（最大值），否则取0
    * cv2.THRESH_BINARY_INV THRESH_BINARY的反转
    * cv2.THRESH_TRUNC 大于阈值部分设为阈值，否则不变
    * cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为0
    * cv2.THRESH_TOZERO_INV THRESH_TOZERO的反转
* return返回值
    * dst： 输出图
    * thresh： 阈值
```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,img_bi = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
ret,img_bi_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
ret,img_tr = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)
ret,img_zero = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)
ret,img_zero_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original','Binary','Binary_INV','TRUNC','ZERO','ZERO_INV']
images = [img,img_bi,img_bi_inv,img_tr,img_zero,img_zero_inv]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray'),plt.title(titles[i])
    plt.xticks([]),plt.yticks([])   # 不显示坐标轴
plt.show()

```

### 图像滤波（平滑）

* cv2.blur # 均值滤波：简单的平均卷积操作
* cv2.boxFilter # 方框滤波：基本和均值一样，可以选择归一化
* cv2.GaussianBlur # 高斯滤波：高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
* cv2.medianBlur # 中值滤波：相当于用中值代替

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenaNoise.png')
blur = cv2.blur(img,(3,3))
boxFilter = cv2.boxFilter(img,-1,(3,3),normalize=False)
gussian = cv2.GaussianBlur(img,(3,3),1)
median = cv2.medianBlur(img,5)

titles = ['Original','Binary','Binary_INV','TRUNC','ZERO','ZERO_INV']
images = [img,blur,boxFilter,gussian,median]
# 显示1
# for i in range(5):
#     plt.subplot(1,5,i+1),plt.imshow(images[i],'gray'),plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])   # 不显示坐标轴
# plt.show()
# 显示2
res = np.hstack((blur,gussian,median))
cv2.imshow('median vs average', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 形态学

 	腐蚀与膨胀属于形态学操作，所谓的形态学，就是改变物体的形状，形象理解一些：腐蚀=变瘦 膨胀=变胖
主要是采用 cv2.erode() 和 cv2.dilate()
主要针对二值化图像的白色部分
.

* 腐蚀：是一种消除边界点，使边界向内部收缩的过程

  * 通俗讲法：在原图的每一个小区域里取最小值，由于是二值化图像，只要有一个点为0，则都为0，来达到瘦身的目的
  *  算法：用 3x3 的 kernel，扫描图像的每一个像素；用 kernel 与其覆盖的二值图像做 “与” 操作；若都为1，则图像的该像素为1；否则为0. 最终结果：使二值图像减小一圈.
* 膨胀：是将与物体接触的所有背景点合并到该物体中，使边界向外部扩张的过程，可以用来填补物体中的空洞.

  * 算法：用 3x3 的 kernel，扫描图像的每一个像素；用 kernel 与其覆盖的二值图像做 “与” 操作；若都为0，则图像的该像素为0；否则为1. 最终结果：使二值图像扩大一圈
.
* 先腐蚀后膨胀的过程称为 开运算。用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积.【cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)】

* 先膨胀后腐蚀的过程称为 闭运算。用来填充物体内细小空洞、连接邻近物体、平滑其边界的同时并不明显改变其面积.【cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)】

* 膨胀 - 腐蚀的过程称为 梯度运算。用来计算轮廓【cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)】

* 顶帽：原始输入 - 开运算结果 【cv2.morphologyEx(img,cv.MORPH_TOPHAT,kernel)】

* 黑帽：闭运算 - 原始输入 【cv2.morphologyEx(img,cv.MORPH_BLACKHAT,kernel)】

#### 腐蚀/膨胀操作

```python
img = cv2.imread('dige.png')
kernel = np.ones((3,3),np.uint8) 
erosion = cv2.erode(img,kernel,iterations = 1)  # 迭代次数越多 和 kernel越大 效果越明显
dilate = cv2.dilate(img,kernel,iterations = 1)

res = np.hstack((img,erosion,dilate))
cv_show('dige and erode and dilate',res)
```

```python
pie = cv2.imread('pie.png')
kernel = np.ones((30,30),np.uint8) 
erosion_1 = cv2.erode(pie,kernel,iterations = 1)
erosion_2 = cv2.erode(pie,kernel,iterations = 2)
erosion_3 = cv2.erode(pie,kernel,iterations = 3)

dilate_1 = cv2.dilate(pie,kernel,iterations = 1)
dilate_2 = cv2.dilate(pie,kernel,iterations = 2)
dilate_3 = cv2.dilate(pie,kernel,iterations = 3)

res_e = np.hstack((pie,erosion_1,erosion_2,erosion_3))
res_d = np.hstack((pie,dilate_1,dilate_2,dilate_3))
# cv2.imshow('res', res_e)
cv2.imwrite('pie_erode_res.jpg',res_e)
cv2.imwrite('pie_dilate_res.jpg',res_d)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

#### 开运算&闭运算

- 开：先腐蚀，再膨胀
- 闭：先膨胀，再腐蚀
```python
import cv2
import numpy as np
img = cv2.imread('dige.png')
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)	# 开：把刺去掉了
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)	# 闭：字和刺都胖了

result = np.hstack((img,opening,closing))
cv2.imshow('open and close',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
#### 梯度运算

- 梯度 = 膨胀-腐蚀
  多出来的白边 减去 减少的白边，即计算一个轮廓出来

```python
# 梯度 = 膨胀-腐蚀
pie = cv2.imread('pie.png')
kernel = np.ones((7,7),np.uint8) 
dilate = cv2.dilate(pie,kernel,iterations = 5)
erosion = cv2.erode(pie,kernel,iterations = 5)
gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)

res = np.hstack((dilate,erosion,gradient))

cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 礼帽&黑帽

- 礼帽 = 原始输入 - 开运算结果 （原图 - 没刺的 = 剩下刺）

- 黑帽 = 闭运算 - 原始输入 （字和刺胖了的 - 原图 = 胖的边缘部分）

```python
img = cv2.imread('dige.png')
kernel = np.ones((7,7),np.uint8) 
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)	# 只剩刺了
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)	# 只剩白边的轮廓
res = np.hstack((img,tophat,blackhat))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 图像梯度-Sobel算子

* cv2.Sobel(src, ddepth, dx, dy, ksize) 进行sobel算子计算
  参数说明：src表示当前图片，ddepth表示图片深度，这里使用cv2.CV_64F使得结果可以是负值， dx表示x轴方向，dy表示y轴方向, ksize表示移动方框的大小
* cv2.convertScalerAbs(src) 将像素点进行绝对值计算
  参数说明: src表示当前图片
* sobel算子：分为x轴方向和y轴方向上的，
  * x轴方向上的算子如图中的Gx，将sober算子在图中进行平移，当前位置的像素值等于sobel算子与(当前位置与周边位置8个点)进行对应位置相乘并相加操作，作为当前位置的像素点（右减左）
  * y轴方向的算子如Gy， 对于x轴方向上，即左右两边的比较（下减上）
* 计算方程为：x轴： p3 - p1 + 2 * p6 - 2 * p4 + p9 - p7， 右边的像素值减去左边的像素值
```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('pie.png',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)# 1,0: 表示只算水平的，不算竖直的
sobelxx = cv2.convertScaleAbs(sobelx)# 

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelyy = cv2.convertScaleAbs(sobely)

# 分别计算x和y,再求和,融合的较好
sobelxy_1 = cv2.addWeighted(sobelxx,0.5,sobelyy,0.5,0)
# 不建议直接计算,融合的不好
sobelxy_2 = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)

# cv_show('pie',img)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobelxx',sobelxx)
cv2.imshow('sobely',sobely)
cv2.imshow('sobelyy',sobelyy)
cv2.imshow('sobelxy_1',sobelxy_1)
cv2.imshow('sobelxy_2',sobelxy_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> ==注意：==白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要用convertScalerAbs取绝对值

#### 图像梯度算子

**laplacian算子**计算方程为：p2 + p4 + p6 +p8 - 4 * p5

Scharr算子和Sobel算子的用法相同，Laplacian算子就不存在x y了

```python
img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)   
sobely = cv2.convertScaleAbs(sobely)  
sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)  

scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)   
scharry = cv2.convertScaleAbs(scharry)  
scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0) 

laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)   

res = np.hstack((img,sobelxy,scharrxy,laplacian))
cv_show(res,'res')

```

> Scharr算子比Sobel算子更敏感，捕获更多细节，更丰富

### Canny边缘检测

1. 使用高斯滤波器，以平滑图像，滤除噪声。
2. 计算图像中每个像素点的梯度强度和方向。
3. 应用非极大值（Non-Maximum Suppression, NMS）抑制，以消除边缘检测带来的杂散响应。
4. 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
5. 通过抑制孤立的弱边缘最终完成边缘检测。


```python
img=cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)

v1=cv2.Canny(img,80,150)
v2=cv2.Canny(img,50,100)	# 阈值设置的合适,就可以把细节信息展示更多,发丝和细纹理都显现出来了

res = np.hstack((img,v1,v2))
cv_show(res,'Canny_res')

```

![image-20211107202726642](https://gitee.com/HB_XN/picture/raw/master/img/20211208200017.png)

```python
img=cv2.imread("car.png",cv2.IMREAD_GRAYSCALE)
v1=cv2.Canny(img,120,250)
v2=cv2.Canny(img,50,100)	# 阈值设置的合适,就可以把细节信息展示更多,高楼的边缘也显现出来了
res = np.hstack((img,v1,v2))
cv_show(res,'res')
```

### 图像金字塔

### 图像轮廓

#### 绘制轮廓

* cv2.findContours(img,mode,method)
  **img: 二值图像**
  **mode: 轮廓检索模式**
* RETR_EXTERNAL ：只检索最外面的轮廓；
* RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
* RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
* **RETR_TREE**：检索所有的轮廓，并重构嵌套轮廓的整个层次; （通常选这个）
**method:轮廓逼近方法**
* CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
* CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
**返回值**
* contours 是一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示
* hierarchy 是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数
>如果报错
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
ValueError: not enough values to unpack (expected 3, got 2)



> 要想返回三个参数：
> 把OpenCV 升级为4.2 就可以了，在终端输入pip install opencv-python==4.2.0
> OpenCV 新版返回两个参数：
> contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
> .
> 查看opencv版本：print(cv2.version)
> 我的opecv版本为：3.4.2

```python
gray=cv2.imread('e.png',0)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# 绘制轮廓
draw_img1 = gray.copy()	# 需进行copy,因drawContours会在原图上绘制,改变原图
draw_img2 = gray.copy()
draw_img3 = gray.copy()
all_contours = cv2.drawContours(draw_img1, contours, -1, (0, 0, 255), 2)	# -1是指所有的轮廓
contours_0 = cv2.drawContours(draw_img2,contours,0,(0,0,255),2)		# 检测到的第1个轮廓
contours_1 = cv2.drawContours(draw_img3,contours,1,(0,0,255),2)		# 检测到的第2个轮廓
cv2.imshow('res',np.hstack((all_contours,contours_0,contours_1)))
cv2.waitKey(0)
```

* cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
  1. 第一个参数是指明在哪幅图像上绘制轮廓；image为三通道才能显示轮廓
  2. 第二个参数是轮廓本身，在Python中是一个list;
  3. 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。
  

#### 轮廓特征

```python
cnt = contours[0]
#面积
cv2.contourArea(cnt)

#周长，True表示闭合的
cv2.arcLength(cnt,True)

```

#### 轮廓近似

* cv2.approxPolyDP(contour,epsilon,True) 把一条平滑的曲线曲折化
  参数
  epsilon：表示的是精度，越小精度越高，因为表示的意思是是原始曲线与近似曲线之间的最大距离
  closed：表示输出的多边形是否封闭；true表示封闭，false表示不封闭。
* 算法步骤 ：

  1. 连接曲线首尾两点A、B形成一条直线AB； 计算曲线上离该直线段距离最大的点C，计算其与AB的距离d；
  2. 比较该距离与预先给定的阈值threshold的大小，如果小于threshold，则以该直线作为曲线的近似，该段曲线处理完毕。
  3. 如果距离大于阈值，则用点C将曲线分为两段AC和BC，并分别对两段曲线进行步骤[1~3]的处理。
  4. 当所有曲线都处理完毕后，依次连接各个分割点形成折线，作为原曲线的近似。

```python
img = cv2.imread('contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
epsilon1 = 0.05*cv2.arcLength(cnt,True) 
epsilon2 = 0.1*cv2.arcLength(cnt,True) 
epsilon3 = 0.15*cv2.arcLength(cnt,True) 
approx1 = cv2.approxPolyDP(cnt,epsilon1,True)
approx2 = cv2.approxPolyDP(cnt,epsilon2,True)
approx3 = cv2.approxPolyDP(cnt,epsilon3,True)

draw_img1 = img.copy()
draw_img2 = img.copy()
draw_img3 = img.copy()

approx1_res = cv2.drawContours(draw_img1, [approx1], -1, (0, 0, 255), 2)
approx2_res = cv2.drawContours(draw_img2, [approx2], -1, (0, 0, 255), 2)
approx3_res = cv2.drawContours(draw_img3, [approx3], -1, (0, 0, 255), 2)
cv_show('res',np.hstack((approx1_res,approx2_res,approx3_res)))
# cv_show(res,'res')

```

* cv2.boundingRect：矩形边框(Bounding Rectangle)是用一个最小的矩形，把找到的形状包起来
  返回四个值，分别是x，y，w，h；
  x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
* cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩行
  第一个参数：img是原图
  第二个参数：(x,y)是矩阵的左上点坐标
  第三个参数：(x+w,y+h)是矩阵的右下点坐标
  第四个参数：(0,255,0)是画线对应的rgb颜色
  第五个参数：2是所画的线的宽度
* cv2.minAreaRect()：得到包覆轮廓的最小斜矩形，
* cv2.minEnclosingCircle()：得到包覆此轮廓的最小圆形
  返回一个二元组，第一个元素为圆心坐标组成的元组，第二个元素为圆的半径值。
* cv2.circle(img, center, radius, color, thickness, lineType, shift) 根据给定的圆心和半径等画圆
  参数说明
  img：输入的图片data
  center：圆心位置
  radius：圆的半径
  color：圆的颜色
  thickness：圆形轮廓的粗细（如果为正）。负厚度表示要绘制实心圆。
  lineType： 圆边界的类型。
  shift：中心坐标和半径值中的小数位数。

```python
# 边界矩形
img = cv2.imread('contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt_2 = contours[2]

x,y,w,h = cv2.boundingRect(cnt_2)
img_rec = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv_show('img_rec',img_rec)
# 轮廓面积与边界矩形比
area = cv2.contourArea(cnt)
rect_area = w * h
extent = float(area) / rect_area
# print ('轮廓面积与边界矩形比',extent)

```

```python
# 外接圆
cnt_8 = contours[8]
(x,y),radius = cv2.minEnclosingCircle(cnt_8) 
center = (int(x),int(y)) 
radius = int(radius) 
img_cir = cv2.circle(img,center,radius,(0,255,255),2)
cv_show('img_cir',img_cir)

```

### 模板匹配

 	模板匹配和卷积原理很像,模板在原图像上从原点开始滑动,计算模板与(图像被模板覆盖的地方)的差别程度,这个差别程度的计算方法在opencv里有6种,然后格每次计算的結果放入一个矩阵里,作为結果输出.假如原图形是AxB大小,而模板是axb大小,则输出结果的矩阵是(A-a+1)x(B-b+1)

简单来说，模板匹配就是在整个图像区域发现与给定子图像匹配的小块区域。

* cv2.matchTemplate(image, templ, method, result=None, mask=None)
  image：待搜索图像
  templ：模板图像
  method：计算匹配程度的方法
  返回参数res：是一个结果矩阵，假设待匹配图像为 I，宽高为(W,H)，模板图像为 T，宽高为(w,h)。那么result的大小就为(W-w+1, H-h+1)

> 其中method：
> TM-SQDIFF:计算平方不同,计算出来的值越小,越相关
> TM_CCORR:计算相关性,计算出来的值越大,越相关
> TM_CCOEFF:计算相关系数,计算出来的值越大,越相关
> TM SQDIFF-NORMED: 计算归一化平方不同,计算出来的值越接近0,越相关
> TM_CCORR-NORMED: 计t算归一化相关性,计算出来的值越接近1,越相关
> TM-CCOEFF-NORMED:计算归一化相关系数,计算出来的值越接近1,越相关
> 公式：
>
> https://docs.opencv.org/3.3.1/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d
> https://blog.csdn.net/a906958671/article/details/89551856

- **cv2.minMaxLoc**(res)
  输入矩阵res
  min_val, max_val, min_loc, max_loc是这个矩阵的最小值，最大值，最大值的索引，最小值的索引

> cv2.rectangle(img, (240, 0), (480, 375), (0, 255, 0), 2)  参数：图像，左上角，右下角，bgr，线条厚度

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
img = cv2.imread('touxiang.jpg',0)
img2 = img.copy()
template = cv2.imread('template.jpg',0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)  #应该是define类型转换为数值类型

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)	# method这不要写成字符串的形式，别加上引号, cv2.xx就行
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  #注意min_loc和max_loc都是二维的情况

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
```

**匹配多个对象**

```python
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)

```

### 直方图

​			直方图（histogram）是灰度级的函数，描述的是图像中每种灰度级像素的个数，反映图像中每种灰度出现的频率。横坐标是灰度级，纵坐标是灰度级出现的频率

* cv2.calcHist(images,channels,mask,histSize,ranges)
  * images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应 用中括号 [] 括来例如[img]
  * channels: 同样用中括号括来它会告函数我们统幅图 像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0][1][2] 它们分别对应着 BGR。
  * mask: 掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
  * histSize:BIN 的数目。也应用中括号括来
  * ranges: 像素值范围常为 [0-256]

```python
import cv2 #opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB
%matplotlib inline 
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('cat.jpg',0) #0表示灰度图
hist = cv2.calcHist([img],[0],None,[256],[0,256])
# hist.shape	# (256,1) 256指0-255的256个像素; 1指1维,像素x 多少多少个

plt.hist(img.ravel(),256); 	# 此处不显示图像,用plt展示方便,若用cv2,要进行颜色转换
plt.show()  #img.ravel()平铺

```

​	

![image-20211030102330822](https://gitee.com/HB_XN/picture/raw/master/img/20211208200028.png)

 	分别显示3个颜色通道的直方图

```python
img = cv2.imread('cat.jpg') 
color = ('b','g','r')
for i,col in enumerate(color): 
    histr = cv2.calcHist([img],[i],None,[256],[0,256]) 
    plt.plot(histr,color = col) 
    plt.xlim([0,256]) 

```

![image-20211030102201137](https://gitee.com/HB_XN/picture/raw/master/img/20211208200031.png)

### mask 操作

   	==bitwise_and(src1, src2, dst=None, mask=None)==

- [ ]  src1、src2：为输入图像或标量，标量可以为单个数值或一个四元组

- [ ] dst：可选输出变量，如果需要使用非None则要先定义，且其大小与输入变量相同
- [ ] mask：图像掩膜，可选参数，为8位单通道的灰度图像，用于指定要更改的输出图像数组的元素，即输出图像像素只有mask对应位置元素不为0的部分才输出，否则该位置像素的所有通道分量都设置为0

```python
img = cv2.imread('cat.jpg', 0)

# 创建mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
cv_show(mask,'mask')

masked_img = cv2.bitwise_and(img, img, mask=mask)	#与操作
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

```

![image-20211030230352999](https://gitee.com/HB_XN/picture/raw/master/img/20211208200034.png)

### 直方图均衡化

```python
img = cv2.imread('cat.jpg',0) #0表示灰度图 #clahe
equ = cv2.equalizeHist(img)   #直方图均衡化

res = np.hstack((img,equ))
cv_show(res,'res')

```

> ​		np.hstack将参数元组的元素数组按水平方向进行叠加  显示图像能显示两张图片对比

```python
arr1 = np.array([[1,3], [2,4] ])
arr2 = np.array([[1,4], [2,6] ])
res = np.hstack((arr1, arr2))
print (res)

[[1 3 1 4]
 [2 4 2 6]]
```

![image-20211030234034559](https://gitee.com/HB_XN/picture/raw/master/img/20211208200036.png)

### 自适应直方图均衡化

​	HE 直方图增强，大家都不陌生，是一种比较古老的对比度增强算法，它有两种变体：AHE 和 CLAHE；两者都是自适应的增强算法，功能差不多，但是前者有一个很大的缺陷，就是有时候会过度放大图像中相同区域的噪声，为了解决这一问题，出现了 HE 的另一种改进算法，就是 CLAHE；CLAHE 是另外一种直方图均衡算法，CLAHE 和 AHE 的区别在于前者对区域对比度实行了限制，并且利用插值来加快计算。它能有效的增强或改善图像（局部）对比度，从而获取更多图像相关边缘信息有利于分割。还能够有效改善 AHE 中放大噪声的问题。另外，CLAHE 的有一个用途是被用来对图像去雾。

**createCLAHE**函数原型：createCLAHE([, clipLimit[, tileGridSize]]) -> retval
 **clipLimit**参数表示对比度的大小。
 **tileGridSize**参数表示每次处理块的大小 。

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
res_clahe = clahe.apply(img)
res = np.hstack((img,equ,res_clahe))
cv_show(res,'res')
```

![image-20211031001120547](https://gitee.com/HB_XN/picture/raw/master/img/20211208200039.png)

### 傅里叶变换

- 傅里叶变换的作用
  高频：**变化剧烈**的灰度分量，例如边界
  低频：**变化缓慢**的灰度分量，例如一片大海
  在原图中做低频/高频的变换较难，因此转换到频域中处理较方便
- 滤波
  低通滤波器：只保留低频，会使得图像模糊
  高通滤波器：只保留高频，会使得图像细节增强

opencv中主要就是 **cv2.dft()** 和 **cv2.idft()** ，输入图像需要**先转换成np.float32 格式**。
得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）, 用逆变换cv2.idft()。

```python
img = cv2.imread('lena.jpg',0)
# 1.转换成np.float32 格式
img_float32 = np.float32(img)
# 2.傅里叶变换 dft
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
# 3.np的fftshift得到低频在中心位置的频谱图
dft_shift = np.fft.fftshift(dft)
# 4.转换一下 得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))	# 0 1两个通道

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

```

### 低通滤波

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)
# 1.转换成np.float32 格式
img_float32 = np.float32(img)
# 2.傅里叶变换 dft
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
# 3.np的fftshift得到低频在中心位置的频谱图
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置

# 4.低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1	# 所以计算图像长宽是为制作mask,中心(30+30)x(30x30)区域为1

# 5.IDFT
fshift = dft_shift*mask					# 仅保留了中心区域
f_ishift = np.fft.ifftshift(fshift)		# 把中心位置的东西放回原位	ifftshift
img_back = cv2.idft(f_ishift)			# 从频谱图转换回来 idft
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])	# 转换一下 得到灰度图能表示的形式

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show() 

```

### 高通滤波

```python
img = cv2.imread('lena.jpg',0)
img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置

# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()

```

