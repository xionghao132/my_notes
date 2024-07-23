# Pyhanlp

## 安装

我在pyhanlp安装过程中遇到了挺多问题。

Github地址：[hankcs/pyhanlp: 中文分词 (github.com)](https://github.com/hankcs/pyhanlp)

正常按照官网的下载方式

```sh
conda install -c conda-forge openjdk python=3.8 jpype1=0.7.0 -y
pip install pyhanlp
```

`pip`安装的时候报红，于是我就手动安装。

直接把官网的**pyhanlp-master.zip**下载下来，解压放入一个文件夹，进入这个文件夹，`cmd`进入当前目录

```sh
python setup.py install
```

安装成功后，导入包还是报错。

```sh
"ModuleNotFoundError: No module named 'pyhanlp'"
```

于是我又调用官方的命令安装，又报了下方的错误。

```sh
File "F:\a_repo\Lib\site-packages\pip\compat\__init__.py", line 75, in console_to_str
return s.decode('utf_8')
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc8 in position 3: invalid continuation byte
```

找到报错中提到的文件`F:\a_repo\Lib\site-packages\pip\compat\__init__.py`，将第73行**sys.stdout.encoding**改为**‘gbk’**即可。

然后我们在这个python环境中，导入这个包的时候，还是在下载一些**data**，可能比较慢。

于是就下载了`hanlp-1.8.3-release .zip`和`data-for-1.7.5 .zip`放在了我的虚拟环境**pytorch**的目录`F:\a_repo\envs\pytorch\Lib\site-packages\pyhanlp-0.1.84-py3.6.egg\pyhanlp\static`下，并且解压。

> **data-for-1.7.5.zip**下载网址[file.hankcs.com](https://file.hankcs.com/hanlp/data-for-1.7.5.zip)
>
> **hanlp-1.8.3-release .zip**下载网址[Releases · hankcs/HanLP (github.com)](https://github.com/hankcs/HanLP/releases)

![image-20220717223454119](C:\Users\我想静静\AppData\Roaming\Typora\typora-user-images\image-20220717223454119.png)

再次进入**python**环境中，`import pyhanlp`下载内容会很快，并且自动解压。

测试一下：

```python
from pyhanlp import *
sentence = "我爱自然语言处理技术！"
print(HanLP.segment(sentence))

#输出
#[我/rr, 爱/v, 自然语言处理/nz, 技术/n, ！/w]
```

成功！

## API

### 分词和词性标注

```python
sentence = "我爱自然语言处理技术！"
s_hanlp = HanLP.segment(sentence)
for term in s_hanlp:
    print(term.word, term.nature)

我 rr
爱 v
自然语言处理 nz
技术 n
！ w
```

### 依存句法分析

```python
s_dep = HanLP.parseDependency(sentence)
print(s_dep)

1   我   我   r   r   _   2   主谓关系    _   _
2   爱   爱   v   v   _   0   核心关系    _   _
3   自然语言处理  自然语言处理  v   v   _   4   定中关系    _   _
4   技术  技术  n   n   _   2   动宾关系    _   _
5   ！   ！   wp  w   _   2   标点符号    _   _
```

### 关键词提取

```python
document = u'''
自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
所以它与语言学的研究有着密切的联系，但又有重要的区别。
自然语言处理并不是一般地研究自然语言，
而在于研制能有效地实现自然语言通信的计算机系统，
特别是其中的软件系统。因而它是计算机科学的一部分。
'''
doc_keyword = HanLP.extractKeyword(document, 3)
for word in doc_keyword:
    print(word)
研究
自然语言
自然语言处理
```

### 摘要抽取

```python
doc_keysentence = HanLP.extractSummary(document, 3)
for key_sentence in doc_keysentence:
    print(key_sentence)
自然语言处理并不是一般地研究自然语言
自然语言处理是计算机科学领域与人工智能领域中的一个重要方向
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法
```

### 感知机词法分析器

```python
PerceptronLexicalAnalyzer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
analyzer = PerceptronLexicalAnalyzer()
print(analyzer.analyze("上海华安工业（集团）公司董事长谭旭光和秘书胡花蕊来到美国纽约现代艺术博物馆参观"))
[上海/ns 华安/nz 工业/n （/w 集团/n ）/w 公司/n]/nt 董事长/n 谭旭光/nr 和/c 秘书/n 胡花蕊/nr 来到/v [美国纽约/ns 现代/ntc 艺术/n 博物馆/n]/ns 参观/v
```

### 中国人名识别

```python
NER = HanLP.newSegment().enableNameRecognize(True)
p_name = NER.seg('王国强、高峰、汪洋、张朝阳光着头、韩寒、小四')
print(p_name)
[王国强/nr, 、/w, 高峰/n, 、/w, 汪洋/n, 、/w, 张朝阳/nr, 光着头/l, 、/w, 韩寒/nr, 、/w, 小/a, 四/m]
```

### 音译人名识别

```python
sentence = '微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克，这些硅谷的科技人'
person_ner = HanLP.newSegment().enableTranslatedNameRecognize(True)
p_name = person_ner.seg(sentence)
print(p_name)
[微软/ntc, 的/ude1, 比尔盖茨/nrf, 、/w, Facebook/nx, 的/ude1, 扎克伯格/nrf, 跟/p, 桑德博格/nrf, 、/w, 亚马逊/nrf, 的/ude1, 贝索斯/nrf, 、/w, 苹果/nf, 的/ude1, 库克/nrf, ，/w, 这些/rz, 硅谷/ns, 的/ude1, 科技/n, 人/n]
```

### 短语提取

```python
phraseList = HanLP.extractPhrase(document, 3)
print(phraseList)
[计算机科学, 中的重要, 之间自然语言]
```

### 拼音转换

```python
s = '重载不是重任'
pinyinList = HanLP.convertToPinyinList(s)
for pinyin in pinyinList:
    print(pinyin.getPinyinWithoutTone(),pinyin.getTone(), pinyin, pinyin.getPinyinWithToneMark())
chong 2 chong2 chóng
zai 3 zai3 zǎi
bu 2 bu2 bú
shi 4 shi4 shì
zhong 4 zhong4 zhòng
ren 4 ren4 rèn
```

声母、韵母

```python
for pinyin in pinyinList:
    print(pinyin.getShengmu(), pinyin.getYunmu())
ch ong
z ai
b u
sh i
zh ong
r en
```

### 繁简转换

```python
Jianti = HanLP.convertToSimplifiedChinese("我愛自然語言處理技術！")
Fanti = HanLP.convertToTraditionalChinese("我爱自然语言处理技术！")
print(Jianti)
print(Fanti)
我爱自然语言处理技术！
我愛自然語言處理技術！
```

## 词性

对于分割后出现的词性，参考[ICTPOS 3.0 汉语词性标记集](https://www.knowledgedict.com/tutorial/nlp-chinese-pos-tagging-ictpos-version-3.html)

## 引用

[ pythonAPI utf-8出错_pip 安装报utf-8错解决办法_weixin_39761558的博客-CSDN博客](https://blog.csdn.net/weixin_39761558/article/details/111785305)

[自然语言处理基础技术工具篇之HanLP - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/51419818)