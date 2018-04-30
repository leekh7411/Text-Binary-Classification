"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import numpy as np
import time
from gensim.models import Word2Vec
from konlpy.tag import Twitter
LOCAL_DATASET_PATH = './sample_data/kin/'
from config import get_config
class KinQueryDataset:
    """
        지식인 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, config):

        self.config = config

        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        self.test_idx = -1
        with open(queries_path, 'rt', encoding='utf8') as f:
            self.vq1_train,self.vq1_test,self.vq2_train,self.vq2_test = self.preprocess(f.readlines(), config.strmaxlen)

        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])
            self.test_idx = len(self.vq1_train)
            if self.test_idx != -1:
                self.labels_test = self.labels[self.test_idx:]
                self.labels = self.labels[:self.test_idx]

        print("label-train size:", self.labels.shape)
        print("label-test  size:", self.labels_test.shape)

    def __len__(self):
        """
        :return: 전체 train 데이터의 수를 리턴합니다
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        :param idx: 필요한 train 데이터의 인덱스
        :return: 인덱스에 맞는 train 데이터, train 레이블 pair를 리턴합니다
        """
        return self.vq1_train[idx], self.vq2_train[idx], self.labels[idx]



    def _batch_loader(self,iterable, n=1):
        """
        데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다
        :param iterable: 데이터 list, 혹은 다른 포맷
        :param n: 배치 사이즈
        :return:
        """
        length = len(iterable)
        for n_idx in range(0, length, n):
            yield iterable[n_idx:min(n_idx + n, length)]



    def preprocess(self,data: list, max_length: int):

        # if we use dataset word's vector may be get lower accuracy
        # but here we plan to use dataset's word-vector
        q_words = []
        vector_size = self.config.w2v_size
        twitter = Twitter()

        # Tokenize using Twitter
        for d in data:
            malist = twitter.pos(d, norm=True, stem=True)
            r = []
            for word in malist:
                if word[1] not in ["josa","Eomi","Punctuation"]:
                    r.append(word[0])
            q_words.append(r)

        # WARNING: Find and replace better word2vec model! not like this..
        model = Word2Vec(q_words,size=vector_size,window=2,hs=1,min_count=1,iter=100,sg=1)
        vocab_len = len(model.wv.vocab)
        print("word2vec vector size :", vector_size)
        print("word2vec vocab  size :", vocab_len)

        # Split sentence(query)
        query1 = []
        query2 = []
        for d in data:
            q1, q2 = d.split('\t')
            q2 = q2.replace('\n', '')
            query1.append(q1)
            query2.append(q2)


        # make query to vector data using upon w2v model
        def query2vector(query):
            vecquery = np.zeros((len(query), max_length, vector_size), dtype=np.float32)
            for i,d in enumerate(query):
                w_cnt = 0
                for wd in (twitter.pos(d, norm=True, stem=True)):
                    if w_cnt < max_length and wd[0] in model.wv.vocab:
                        vecquery[i,w_cnt] = model[wd[0]]
                        w_cnt += 1
            vecquery = np.expand_dims(vecquery, axis=3) # for CNN layer?
            return vecquery

        vec_query1 = query2vector(query1)
        vec_query2 = query2vector(query2)

        def split_train_test(query):
            idx = (int)(len(query) * self.config.test_set_rate)
            return query[:idx],query[idx:]

        del model

        vq1_train , vq1_test = split_train_test(vec_query1)
        del vec_query1

        vq2_train , vq2_test = split_train_test(vec_query2)
        del vec_query2

        print("query-train size:",vq1_train.shape)
        print("query-test  size:",vq1_test.shape)

        return vq1_train,vq1_test,vq2_train,vq2_test


# for testing
#config = get_config()
#dataset = KinQueryDataset(LOCAL_DATASET_PATH,config)