# How to train word2vec
### Gensim 词向量数据训练

#### 1.处理语料库

##### 1.1搜狗数据库

下载地址：[SougouLab](http://www.sogou.com/labs/resource/ca.php)

下载格式为 完整版(711MB).zip

下载完以后利用editplus或者ultraedit将'.dat'数据转为unicode的txt文本

利用

```
cat news_tensite_xml.txt | grep "<content>"  > corpus.txt 
```

提取文本内容

接着去除错误符号和标签

```python
# -*-coding:utf-8 -*-
#remove.py
import re 

content = ''
with open('corpus.txt ','r') as obj:
	content = obj.read()
	# content = re.sub('<content>|</content>|','',content)
	content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]^{\n}', ' ', content)
	obj.close()

with open('newcorpus.txt', 'w') as obj:
	obj.write(content)
	obj.close()
```

##### 1.2 维基百科

维基百科数据地址：[zhwiki](https://dumps.wikimedia.org/zhwiki/)

下载最新的 zhwiki-latest-pages-articles.xml.bz2  文件

```python
#-*-coding: utf-8 -*-
#wiki_to_texts.py
import logging
import sys

from gensim.corpora import WikiCorpus

def main():
    if len(sys.argv)!=2:
        print("Usage:python3"+sys.argv[0]+"wiki_data_path")
        exit()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    wiki_corpus=WikiCorpus(sys.argv[1],dictionary={})
    texts_num=0

    with open("wiki_texts.txt","w",encoding='utf-8') as output:
        for text in wiki_corpus.get_texts():
            filesss = ''
            for i in text:
                filesss += ' ' + i

            output.write(filesss +"\n")
            texts_num+=1
            if texts_num%10000==0:
                logging.info("已处理%d篇文章" %texts_num)

if __name__=="__main__":
    main()
```

利用

```
python3 wiki_to_texts.py zhwiki-latest-pages-articles.xml.bz2 
```

得到wiki_texts.txt

接着将繁体转为简体

```
opencc -i wiki_texts.txt -o wiki_zh_s.txt -c t2s.json
```

最后得到 wiki_zh_s.txt文件，将其重命名为newcorpus.txt

#### 2.分词

运行下面文件

```Python
# -*-coding:utf-8 -*-
#split.py
import jieba
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #jieba custom setting
    # jieba.set_dictionary('jieba_dict/dict.txt.big')

    #load stopwords set
    stopwordset=set()
    with open('/Users/yummy/Desktop/Projects/stopword/中文停用词库.txt','r',encoding='utf-8') as sw:
        for line in sw:
            stopwordset.add(line.strip("\n"))

    output=open('wiki_seg','w')

    texts_num=0

    with open('newcorpus.txt','r') as content:
        for line in content:
            line=line.strip('\n')
            words=jieba.cut(line,cut_all=False)  #cut_all:False for all pattern,True for accurate pattern
            for word in words:
                if word not in stopwordset:
                    output.write(word+ " ")

            texts_num+=1
            if texts_num%10000==0:
                logging.info('已完成%d行的分词' %texts_num)

    output.close()

if __name__ == '__main__':
    main()
```

得到分词后的wiki_seg文件

#### 3.训练word2vec

```python
# -*-coding:utf-8 -*-
from gensim.models import word2vec
import logging

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences=word2vec.Text8Corpus('wiki_seg')
    model=word2vec.Word2Vec(sentences,size=250)     #size is the dimensionality of the future vectors
    # model.save_word2vec_format(u'med250.model.bin',binary=True)

    model.wv.save_word2vec_format(u'med250.model.bin',binary=True)
    #how to load a model?
    #model=word2vec.Word2Vec.load_word2vec_format('your_model.bin',binary=True)


if __name__ == '__main__':
    main()
```

#### 4.测试数据

```python
# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim import models
from sklearn.metrics.pairwise import cosine_similarity as cosine
import logging

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #model=models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
    model=models.KeyedVectors.load_word2vec_format('med250.model.bin',binary=True)

    print(model.most_similar('hello'))

#    cos = cosineSimilarity(model['hello'],model['kitty'])
    print(cos)
if __name__=='__main__':
    main()
```

利用

```python
model=models.KeyedVectors.load_word2vec_format('med250.model.bin',binary=True)
#读取训练词库
model['A'] #查找A的词向量
model.most_similar('A') #与A最接近的数据
model.vector_size #词向量维数
model.wv.vocab	#训练后的词库列表
```



