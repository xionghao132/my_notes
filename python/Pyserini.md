# Pyserini

## 概述

主要是复现了很多信息检索模型，包括稀疏检索和稠密检索，可以作为一个`python`工具包直接使用。

## 安装

```sh
pip install pyserini
faiss-cpu==1.7.2,
transformers==4.21.3
torch==1.12.1
python==3.8
conda install -c conda-forge openjdk=11
```

## 使用

### Sparse Model

> 通过自动下载`pyserini`构建好的索引去检索

```python
rom pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('wiki-dpr')
hits = searcher.search('hubble space telescope')

# Print the first 10 hits:
for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:15 {hits[i].score:.5f}')
#这个方法可以列出所有的可行的    
LuceneSearcher.list_prebuilt_indexes()
```

> 如果下载很慢，我们也可以手动安装，然后直接加载

```python
searcher = LuceneSearcher('indexes/index-robust04-20191213/')  #注意有个/，表示查找该目录下的索引
```

> 获取更多信息

```python
# Grab the raw text:
hits[0].raw

# Grab the raw Lucene Document:
hits[0].lucene_document

#another way
doc = searcher.doc('7157715')
doc.contents()                                                                             
# Raw document
doc.raw()

for i in range(searcher.num_docs):
    print(searcher.doc(i).docid())
```

### Learned Sparse Model

```python
from pyserini.search.lucene import LuceneImpactSearcher

searcher = LuceneImpactSearcher.from_prebuilt_index(
    'msmarco-v1-passage-unicoil',
    'castorini/unicoil-msmarco-passage')
hits = searcher.search('what is a lobster roll?')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
```

### Dense Model

```python
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder

encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
searcher = FaissSearcher.from_prebuilt_index(
    'msmarco-passage-tct_colbert-hnsw',
    encoder
)
hits = searcher.search('what is a lobster roll')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
```

### Hybrid Model

> 主要是对输出进行结合

```python
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from pyserini.search.hybrid import HybridSearcher

ssearcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
dsearcher = FaissSearcher.from_prebuilt_index(
    'msmarco-passage-tct_colbert-hnsw',
    encoder
)
hsearcher = HybridSearcher(dsearcher, ssearcher)
hits = hsearcher.search('what is a lobster roll')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
```

## 创建BM25索引

有三种`json`格式

```json
#1 每个文档写成一个json
{
  "id": "doc1",
  "contents": "contents of doc one."
}

#2 一个文档
[
  {
    "id": "doc1",
    "contents": "contents of doc one."
  },
  {
    "id": "doc2",
    "contents": "contents of document two."
  },
  {
    "id": "doc3",
    "contents": "here's some text in document three."
  }
]

#3 一个文档
{"id": "doc1", "contents": "contents of doc one."}
{"id": "doc2", "contents": "contents of document two."}
{"id": "doc3", "contents": "here's some text in document three."}
```

```python
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input tests/resources/sample_collection_jsonl \
  --index indexes/sample_collection_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

- `--storePositions`: builds a standard positional index
- `--storeDocvectors`: stores doc vectors (required for relevance feedback)
- `--storeRaw`: stores raw documents
> 创建好索引后就可以直接使用了
```python
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/sample_collection_jsonl')
hits = searcher.search('document')

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
```

> 直接进行`batch`检索

```sh
python -m pyserini.search.lucene \
  --index indexes/sample_collection_jsonl \
  --topics tests/resources/sample_queries.tsv \
  --output run.sample.txt \
  --bm25
```

> 还可以添加一些特征

```json
{
  "id": "doc1",
  "contents": "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.",
  "NER": {
            "ORG": ["The Manhattan Project"],
            "MONEY": ["World War II"]
         }
}
```

> 使用中文搜索的时候要进行设置

```python
searcher.set_language('zh')
hits = searcher.search('滑铁卢')

python -m pyserini.search.lucene \
  --index indexes/sample_collection_jsonl_zh \
  --topics tests/resources/sample_queries_zh.tsv \
  --output run.sample_zh.txt \
  --language zh \
  --bm25
```

## 创建Dense Index

> 数据格式可以是一个json文件，也可以是一个json文档

```json
{
  "id": "CACM-2636",
  "contents": "Generation of Random Correlated Normal ... \n"
}
```

```
python -m pyserini.encode \
  input   --corpus tests/resources/simple_cacm_corpus.json \
          --fields text \  # fields in collection contents
          --delimiter "\n" \
          --shard-id 0 \   # The id of current shard. Default is 0
          --shard-num 1 \  # The total number of shards. Default is 1
  output  --embeddings path/to/output/dir \
          --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco \
          --fields text \  # fields to encode, they must appear in the input.fields
          --batch 32 \
          --fp16  # if inference with autocast()
```



## 创建Sparse Index

```
python -m pyserini.encode \
  input   --corpus tests/resources/simple_cacm_corpus.json \
          --fields text \
  output  --embeddings path/to/output/dir \
  encoder --encoder castorini/unicoil-d2q-msmarco-passage \
          --fields text \
          --batch 32 \
          --fp16 # if inference with autocast()
```



## 创建不同类型的index

* [HNSWPQ](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSWPQ.html#struct-faiss-indexhnswpq)

```
python -m pyserini.index.faiss \
  --input path/to/encoded/corpus \  # either in the Faiss or the jsonl format
  --output path/to/output/index \
  --hnsw \
  --pq
```

* [HNSW](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSW.html#struct-faiss-indexhnsw)

```
python -m pyserini.index.faiss \
  --input path/to/encoded/corpus \  # either in the Faiss or the jsonl format
  --output path/to/output/index \
  --hnsw
```

* [PQ](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexPQ.html)

```
python -m pyserini.index.faiss \
  --input path/to/encoded/corpus \  # either in the Faiss or the jsonl format
  --output path/to/output/index \
  --pq
```

* [Flat](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlat.html)

```
python -m pyserini.index.faiss \
  --input path/to/encoded/corpus \  # in jsonl format
  --output path/to/output/index \
```

>  使用创建好的索引去搜索

```python
from pyserini.search import FaissSearcher

searcher = FaissSearcher(
    'indexes/dindex-sample-dpr-multi',
    'facebook/dpr-question_encoder-multiset-base'
)
hits = searcher.search('what is a lobster roll')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
```



## 配置BM25的参数

```python
searcher.set_bm25(0.9, 0.4)
searcher.set_rm3(10, 10, 0.5)

hits2 = searcher.search('hubble space telescope')

# Print the first 10 hits:
for i in range(0, 10):
    print(f'{i+1:2} {hits2[i].docid:15} {hits2[i].score:.5f}')
```

```
python -m pyserini.search --topics tools/topics-and-qrels/ctqueries2021.tsv \
 --index indexes/lucene-index-ct \
 --output runs/run.msmarco-doc.bm25.rm3.txt \
 --hits 100 \
 --bm25 --rm3 --k1 0.9 --b 0.4
```



[pyserini/README.md at master · castorini/pyserini (github.com)](https://github.com/castorini/pyserini/blob/master/docs/installation.md)

```
python -m pyserini.search.lucene --topics /data3/xhao/bpr-master/downloads/data/retriever/nq/nq-test.qa.tsv --index indexes_download/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6 --output output/run.msmarco-doc.bm25.rm3.txt --hits 100  --bm25 --rm3 --k1 0.9 --b 0.4
```

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class BM25RM3:
    def __init__(self, k1=1.2, b=0.75, k3=1000, top_n=10):
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.top_n = top_n
        self.count_vectorizer = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()
        self.corpus = None
        self.query = None
        self.document_scores = None

    def fit(self, corpus):
        self.corpus = self.count_vectorizer.fit_transform(corpus)
        self.tfidf_transformer.fit(self.corpus)

    def calculate_bm25_scores(self, query):
        self.query = self.count_vectorizer.transform([query])
        doc_lengths = self.corpus.sum(axis=1)
        average_doc_length = np.mean(doc_lengths)
        document_term_matrix = self.corpus.T.tocsr()
        document_scores = np.zeros(self.corpus.shape[0])
        for term in self.query.indices:
            df = document_term_matrix[term].indices.size
            idf = np.log((self.corpus.shape[0] - df + 0.5) / (df + 0.5))
            tf = self.query.data[self.query.indices == term][0]
            document_scores += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_lengths / average_doc_length))
        self.document_scores = document_scores

    def calculate_rm3_scores(self, query):
        expanded_query = self.query.copy()
        top_documents = np.argsort(self.document_scores)[::-1][:self.top_n]
        for doc in top_documents:
            doc_vector = self.corpus[doc]
            doc_terms = doc_vector.indices
            doc_tf = doc_vector.data
            doc_length = doc_vector.sum()
            for term in doc_terms:
                if term in expanded_query.indices:
                    term_index = np.where(expanded_query.indices == term)[0][0]
                    expanded_query.data[term_index] += (self.k3 * doc_tf[doc_terms == term] * (1 - self.b + self.b * doc_length / average_doc_length))
                else:
                    expanded_query.indices = np.append(expanded_query.indices, term)
                    expanded_query.data = np.append(expanded_query.data, (self.k3 * doc_tf[doc_terms == term] * (1 - self.b + self.b * doc_length / average_doc_length)))
        self.query = expanded_query

    def search(self, query, corpus):
        self.fit(corpus)
        self.calculate_bm25_scores(query)
        self.calculate_rm3_scores(query)
        top_documents = np.argsort(self.document_scores)[::-1][:self.top_n]
        return top_documents

# 示例用法
corpus = ["This is the first document.",
          "This document is the second document.",
          "And this is the third one.",
          "Is this the first document?"]
query = "document"
bm25rm3 = BM25RM3()
top_documents = bm25rm3.search(query, corpus)
print(top_documents)
```

```
# 导入所需的库和工具
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 步骤1：准备语料库和查询
corpus = [...]  # 语料库中的文档
queries = [...]  # 用户查询

# 步骤2：文本预处理
def preprocess(text):
    # 分词、去停用词、词干提取等文本预处理操作
    return processed_text

corpus = [preprocess(doc) for doc in corpus]
queries = [preprocess(query) for query in queries]

# 步骤3：初始检索（使用BM25算法）
# 这里假设使用BM25算法进行初始检索
from gensim.summarization import bm25
import numpy as np

tokenized_corpus = [doc.split() for doc in corpus]
bm25_model = bm25.BM25(tokenized_corpus)
average_idf = np.mean(list(bm25_model.idf.values()))

# 步骤4：反馈机制（RM3）
N = 10  # 选择排名前N的文档作为反馈文档
feedback_docs = []

for query in queries:
    scores = bm25_model.get_scores(query.split(), average_idf)
    top_N_docs = np.argsort(scores)[-N:]
    feedback_docs.append(top_N_docs)

# 提取关键词并计算权重
feedback_keywords = []

for doc_indices in feedback_docs:
    doc_text = [corpus[i] for i in doc_indices]
    vectorizer = TfidfVectorizer()
    doc_tfidf_matrix = vectorizer.fit_transform(doc_text)
    doc_tfidf_sum = doc_tfidf_matrix.sum(axis=0)
    feedback_keywords.extend([vectorizer.get_feature_names_out()[i] for i in doc_tfidf_sum.argsort()[0, -N:]])

# 步骤5：扩展查询
expanded_queries = []

for query in queries:
    expanded_query = query + ' ' + ' '.join(feedback_keywords)
    expanded_queries.append(expanded_query)

# 步骤6：重新执行查询
# 使用扩展后的查询重新计算文档的相关性分数

# 步骤7：合并排名
# 将初始检索结果和重新执行查询结果进行合并，例如通过加权求和

# 步骤8：返回结果
# 返回最终的检索结果

```

