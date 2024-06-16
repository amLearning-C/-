import random
import re
import jieba
import numpy as np

DATA_PATH = r'D:/Users/Vincent W/Desktop/研究生/学习/研一下NIP/作业2/jyxstxtqj_downcc.com1/'


def get_single_corpus(file_path):
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@★、…【】《》‘’[\\]^_`{|}~「」『』（）]+'
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
        corpus = re.sub(r1, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        f.close()
    words = list(jieba.cut(corpus))
    #print("Corpus length: {}".format(len(words)))
    return words


def get_dataset(data):
    max_len = 60
    step = 3
    sentences = []
    next_tokens = []
    tokens = list(set(data))
    tokens_indices = {token: tokens.index(token) for token in tokens}
    #print('Unique tokens:', len(tokens))

    for i in range(0, len(data) - max_len, step):
        sentences.append(
            list(map(lambda t: tokens_indices[t], data[i: i + max_len])))
        next_tokens.append(tokens_indices[data[i + max_len]])
    #print('Number of sequences:', len(sentences))
    #print('Vectorization...')
    next_tokens_one_hot = []
    for i in next_tokens:
        y = np.zeros((len(tokens),))
        y[i] = 1
        next_tokens_one_hot.append(y)
    return sentences, next_tokens_one_hot, tokens, tokens_indices
