import pandas as pd
import numpy as np
import json
import csv
import codecs
from keras.preprocessing.sequence import pad_sequences
def readFile(file_name):
    head_id_col_vector = ['token_id', 'token', "BIO", "relation", 'head']
    file = pd.read_csv(file_name, names=head_id_col_vector, encoding="utf-8",
                           engine='python', sep="\t", quoting=csv.QUOTE_NONE).as_matrix()
    return file

def collect_data_set(file,save_file):
    datas = []
    text = []
    BIOS = []
    relations = []
    heads = []
    for i in range(file.shape[0]):
        if '#doc' not in file[i][0]:
            text.append(file[i][1])
            BIOS.append(file[i][2])
            relations.append(file[i][3])
            heads.append(file[i][4])
        else:
            dic = {}
            dic['text'] = text
            dic['BIOS'] = BIOS
            dic['relations'] = relations
            dic['heads'] = heads
            datas.append(dic)

            text = []
            BIOS = []
            relations = []
            heads = []

    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump(datas, f, indent=4, ensure_ascii=False)
    return datas

def collect_char2id(datasets,save_file):
    chars = {}
    for data in datasets:
        for word in data['text']:
            for char in word:
                chars[char] = chars.get(char, 0) + 1
    id2char = {i+1:j for i,j in enumerate(chars)} # padding: 0
    char2id = {j:i for i,j in id2char.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

def collect_n_char2id(datasets,save_file,n):
    chars = {}
    for data in datasets:
        for word in data['text']:
            n_chars = n_char(word,n)
            for _n_char in n_chars:
                chars[_n_char] = chars.get(_n_char, 0) + 1
    id2char = {i+1:j for i,j in enumerate(chars)} # padding: 0
    char2id = {j:i for i,j in id2char.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

def collect_word2id(datasets,save_file):
    words = {}
    for data in datasets:
        for word in data['text']:
            words[word] = words.get(word,0)+1
    id2word = {i+1:j for i,j in enumerate(words)} #padding:0
    word2id = {j:i for i,j in id2word.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2word, word2id], f, indent=4, ensure_ascii=False)

def collect_BIO2id(datasets,save_file):
    BIOs = {}
    for data in datasets:
        for bio in data['BIOS']:
            if bio != 'O':
                BIOs[bio] = BIOs.get(bio,0) +1

    id2BIO = {i+1:j for i,j in enumerate(BIOs)} #padding:0
    id2BIO[0] = 'O'
    BIO2id = {j:i for i,j in id2BIO.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2BIO, BIO2id], f, indent=4, ensure_ascii=False)

# def collect_relations2id(datasets,save_file):
#     BIOs = {}
#     for data in datasets:
#         for bio in data['BIOS']:
#              BIOs[bio] = BIOs.get(bio,0) +1
#     id2BIO = {i+1:j for i,j in enumerate(BIOs)} #padding:0
#     BIO2id = {j:i for i,j in id2BIO.items()}
#     with codecs.open(save_file, 'w', encoding='utf-8') as f:
#         json.dump([id2BIO, BIO2id], f, indent=4, ensure_ascii=False)


#这里有bug只能输出字符串，到时候重写一下
class read_properties:
    def __init__(self,filepath, sep='=', comment_char='#'):
        """Read the file passed as parameter as a properties file."""
        self.props = {}
        #print filepath
        with open(filepath, "rt") as f:
            for line in f:
                #print line
                l = line.strip()
                if l and not l.startswith(comment_char):
                    key_value = l.split(sep)
                    self.props[key_value[0].strip()] = key_value[1].split("#")[0].strip('" \t')


    def getProperty(self,propertyName):
        return self.props.get(propertyName)


def sentence_pad(X,maxlen_sentence):
    #sentence_level pad for word input and bio tagging
    #use the maxlen of batch datas to pad the sentence level inputs
    """

    :param datas: [batch_size,None]
    :return: datas : [batch_size,maxlen of sentence]
    """
    # L = [len(x) for x in X]
    # ML = max(L)
    ML = maxlen_sentence
    return  [x + [0] * (ML - len(x)) for x in X]


def n_char(word,n):
    """
    split the word use n_gram
    n = 2
    word = love
    ==>  lo ov ve e<pad>
    n =3
    word = love
    ==> lov ove ve<pad>
    :param word:
    :return:
    """
    word = str(word)
    n_char = []
    n_char.append('<pad>'*(n-1) + word[0])
    temp = ''
    for index,char in enumerate(word):
        if index+n < len(word):
            temp += word[index:index+n]
            n_char.append(temp)
            temp = ''
        else:
            temp += word[index:]
            temp += '<pad>' * (n - len(temp))
            n_char.append(temp)
            temp = ''
    return n_char

def char_pad(datas,maxlen_sentence,maxlen_word):
    #word_leve pad for char input
    #use the maxlen of batch data of words to pad the char levels and use the maxlen of batch datas to pad the sentence level inputs
    """
    :param datas: [batch_size,None,None]
    :return: [batch_size,maxlen of sentence , maxlen of words]
    """
    new_data = []
    for sentence in datas:
        _sentence = []
        for word in sentence:
            if len(word) < maxlen_word:
                word+=[0]*(maxlen_word - len(word))
            else:
                word = word[:maxlen_word]
            _sentence.append(word)

        pad_word = [0]*maxlen_word
        if len(_sentence) < maxlen_sentence:
            for i in range(maxlen_sentence - len(_sentence)):
                _sentence.append(pad_word)
        else:
            _sentence = _sentence[:maxlen_sentence]
        new_data.append(_sentence)
    return new_data

#TODO complete the function for joint extraction
def load_data(mode):
    #only for ner prediction now , then i will compelet the function for joint extraction
    #load data for predict
    config_file = read_properties('config/CoNLL04/bio_config')
    is_use_n_char = bool(config_file.getProperty('is_use_n_char'))
    filename_char2id = config_file.getProperty("filename_char2id")
    filename_word2id = config_file.getProperty("filename_word2id")
    filename_BIO2id = config_file.getProperty("filename_BIO2id")
    filename_relation2id = config_file.getProperty("filename_relation2id")
    id2char, char2id = json.load(open(filename_char2id, encoding='utf-8'))
    id2n_char, n_char2id = json.load(open(filename_char2id, encoding='utf-8'))
    id2word, word2id = json.load(open(filename_word2id, encoding='utf-8'))
    id2BIO, BIO2id = json.load(open(filename_BIO2id, encoding='utf-8'))
    filename_train_me = config_file.getProperty("filename_train_me")
    filename_dev_me = config_file.getProperty("filename_dev_me")
    filename_test_me = config_file.getProperty("filename_test_me")
    maxlen_sentence = int(config_file.getProperty('maxlen_sentence'))
    maxlen_word = int(config_file.getProperty('maxlen_word'))
    eval_data=  []
    # import ipdb
    # ipdb.set_trace()
    if mode == 'dev':
        eval_data = json.load(open(filename_dev_me, encoding='utf-8'))
    if mode == 'test':
        eval_data = json.load(open(filename_test_me, encoding='utf-8'))
    if mode == 'train':
        eval_data = json.load(open(filename_train_me,encoding='utf-8'))

    TEXT_WORD, TEXT_CHAR, BIO = [], [], []
    for data in eval_data:
        text = data['text']
        bio = data['BIOS']
        _text_word = [word2id.get(word,0) for word in text]
        _text_char = []  # 2 dimmensions
        if is_use_n_char:
            for word in _text_word:
                n_chars = n_char(word,3)
                chars = [n_char2id.get(_char) for _char in n_chars]
                _text_char.append(chars)
        else:
            for word in _text_word:
                chars = [char2id.get(_char) for _char in str(word)]
                _text_char.append(chars)
        _bio = [BIO2id.get(b) for b in bio]
        TEXT_WORD.append(_text_word)
        TEXT_CHAR.append(_text_char)  # [batch,word,char] #padding two times,
        # first in word dimensions for sentence maxlen ,then ,in char dimensions for maxlen_word
        BIO.append(_bio)
    TEXT_WORD = pad_sequences(TEXT_WORD, maxlen=maxlen_sentence, padding='post', value=0)
    TEXT_CHAR = np.array(char_pad(TEXT_CHAR,maxlen_sentence,maxlen_word))
    # BIO = pad_sequences(BIO, maxlen=30, padding='post', value=0)
    return TEXT_WORD,TEXT_CHAR,BIO

#TODO
class data_generator():
    def __init__(self,data,char2id,n_char2id,word2id,BIO2id,maxlen_sentence,maxlen_word,is_use_n_char,batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.char2id = char2id
        self.n_char2id = n_char2id
        self.word2id = word2id
        self.BIO2id = BIO2id
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.is_use_n_char = is_use_n_char
        self.steps = len(self.data)//self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True :
            index = list(range(len(self.data)))
            np.random.shuffle(index)
            TEXT_WORD,TEXT_CHAR,BIO = [],[],[]
            for idx in index:
                _data = self.data[idx]
                text = _data['text']
                bio = _data['BIOS']
                _text_word = [self.word2id.get(word) for word in text]
                _text_char = [] # 2 dimmensions
                if self.is_use_n_char:
                    for word in _text_word:
                        n_chars = n_char(word,3)
                        chars = [self.n_char2id.get(_char) for _char in n_chars]
                        _text_char.append(chars)
                else:
                    for word in _text_word:
                        chars = [self.char2id.get(_char) for _char in str(word)]
                        _text_char.append(chars)
                _bio = [self.BIO2id.get(b) for b in bio]
                TEXT_WORD.append(_text_word)
                TEXT_CHAR.append(_text_char) #[batch,word,char] #padding two times,
                # first in word dimensions for sentence maxlen ,then ,in char dimensions for maxlen_word
                BIO.append(_bio)
                if len(TEXT_WORD) == self.batch_size or idx == index[-1]:
                    TEXT_WORD = pad_sequences(TEXT_WORD,maxlen=self.maxlen_sentence,padding='post',value=0)
                    TEXT_CHAR = np.array(char_pad(TEXT_CHAR,self.maxlen_sentence,self.maxlen_word))
                    BIO = pad_sequences(BIO,maxlen=self.maxlen_sentence,padding='post',value=0)
                    yield [TEXT_WORD,TEXT_CHAR,BIO ],None
                    TEXT_WORD,TEXT_CHAR,BIO =[],[],[]

def _load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr)[:100]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='utf-8'))

    return embeddings_index

def _load_embedding_matrix(word_index, embedding):
    embed_word_count = 0
    # nb_words = min(max_features, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(size=(nb_words+1, 100))

    for word, i in word_index.items():
#        if i >= max_features: continue
        if word not in embedding:
            word = word.lower()
        if word.islower and word not in embedding:
            word = word.title()
        embedding_vector = embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embed_word_count += 1
    print('词向量的覆盖率为{}'.format(embed_word_count / len(word_index)))
    return embedding_matrix

def get_embedding_matrix(word_index):
    embedding_dir = 'data/CoNLL04/glove.6B.100d.txt'
    embedding = _load_embed(embedding_dir)
    embedding_matrix = _load_embedding_matrix(word_index, embedding)

    return embedding_matrix