# LangChain

## 概述

# Langchain文档分割器代码详解



```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_community.document_loaders import PyPDFLoader
```



word

mode="single"加载整个文档，mode="elements"按行来切分

```python
loader = UnstructuredFileLoader(filepath, mode="single")
print(loader.load())  #文档可视化
```



txt、xlsx都是一个文档读取

```python
loader = UnstructuredFileLoader(filepath)
print(loader.load())
```



json

jq_schema='.'表示全部读取，如果是'.[].input'索引其中一个key为input的所有value作为page_content.

```python
data = json.loads(Path(file_path).read_text())

loader2 = JSONLoader(
    file_path=json_path,
    jq_schema='.', text_content=False)   #jq_schema='.'表示全部读取  '.[].input'索引其中一个key
```



PDF

PDF标准化为ISO 32000，可以使用PyPDFLoader读取，其中会按页面划分Document, 每个文档包含页面内容和元数据，以及`page`号码。

```python
loader = PyPDFLoader(path)
pages = loader.load_and_split()
print(loader.load())
```



mode="single"将整个文档读取，mode="elements"标题和和描述文本分开

```python
loader = UnstructuredPDFLoader(path,mode="elements")   # Title and NarrativeText
print(loader.load())
```



csv

将一行数据作为一个Document(page_content)

```python
loader = CSVLoader(file_path='xxx/waimai_10k.csv')
data = loader.load()
```



文档分割

经过上面的工具加载称Document

```python
loader.load_and_split(text_splitter)
#或者
text_splitter.split_documents(loader)
#单纯字符
text_splitter.split_text(text)
```



CharacterTextSplitter按照字符拆分

它根据字符（默认为 "\n\n"）进行拆分，并通过字符数来衡量块的长度。

```python
with open('../../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.create_documents([state_of_the_union])
```





加载word文档并分割，



```python
loader = UnstructuredFileLoader('xxx/xx.docx', mode="elements")   
docs_all = loader.load()      # 这里加载文档。
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)    # 换行划分\n\n
docs = text_splitter.split_documents(docs_all)   
```



也可以自定义分割类ChineseTextSplitter

```python
class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:  # 如果传入是pdf
            text = re.sub(r"\n{3,}", "\n", text)  # 将连续出现的3个以上换行符替换为单个换行符，从而将多个空行缩减为一个空行。
            text = re.sub('\s', ' ', text)  # 将文本中的所有空白字符（例如空格、制表符、换行符等）替换为单个空格
            text = text.replace("\n\n", "")  # 将文本中的连续两个换行符替换为空字符串
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # 用于匹配中文文本中的句子分隔符，例如句号、问号、感叹号等
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)  
        return sent_list
--------------------------------------------
textSplitter = ChineseTextSplitter(True)
loader = UnstructuredFileLoader(filepath, mode="elements")
docs= loader.load_and_split(textSplitter)   # 这里进入.split_text()
```



RecursiveCharacterTextSplitter

递归按字符切分，它由一个字符列表参数化。它尝试按顺序在它们上进行切割，直到块变得足够小。默认列表是`["\n\n", "\n", " ", ""]`。这样做的效果是尽可能保持所有段落（然后句子，然后单词）在一起，因为它们在语义上通常是最相关的文本片段。

```
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    separators=["\n"]     # 自定义切分
)

test = """a\nbcefg\nhij\nk"""
print(len(test))
text = r_splitter.split_text(test)      # test没进过加载器
```

