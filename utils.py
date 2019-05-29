import pandas as pd
import numpy as np
import json
import csv
import codecs

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

def sentence_pad(datas):
    #sentence_level pad for word input and bio tagging
    #use the maxlen of batch datas to pad the sentence level inputs
    """

    :param datas: [batch_size,None]
    :return: datas : [batch_size,maxlen of sentence]
    """

    return datas
def char_pad(datas):
    #word_leve pad for char input
    #use the maxlen of batch data of words to pad the char levels and use the maxlen of batch datas to pad the sentence level inputs
    """

    :param datas: [batch_size,None,None]
    :return: [batch_size,maxlen of sentence , maxlen of words]
    """

    return datas

#TODO complete the function for joint extraction
def load_data(mode):
    #only for ner prediction now , then i will compelet the function for joint extraction
    #load data for predict
    config_file = read_properties('config/CoNLL04/bio_config')
    filename_char2id = config_file.getProperty("filename_char2id")
    filename_word2id = config_file.getProperty("filename_word2id")
    filename_BIO2id = config_file.getProperty("filename_BIO2id")
    filename_relation2id = config_file.getProperty("filename_relation2id")
    id2char, char2id = json.load(open(filename_char2id, encoding='utf-8'))
    id2word, word2id = json.load(open(filename_word2id, encoding='utf-8'))
    id2BIO, BIO2id = json.load(open(filename_BIO2id, encoding='utf-8'))
    filename_dev_me = config_file.getProperty("filename_dev_me")
    filename_test_me = config_file.getProperty("filename_test_me")
    eval_data=  []

    if mode == 'dev':
        eval_data = json.load(open(filename_dev_me, encoding='utf-8'))
    if mode == 'test':
        eval_data = json.load(open(filename_test_me, encoding='utf-8'))

    for data in eval_data:
        TEXT_WORD, TEXT_CHAR, BIO = [], [], []
        text = data['text']
        bio = data['BIOS']
        _text_word = [word2id.get(word) for word in text]
        _text_char = []  # 2 dimmensions
        for word in _text_word:
            chars = [char2id.get(_char) for _char in word]
            _text_char.append(chars)
        _bio = [BIO2id.get(b) for b in bio]
        TEXT_WORD.append(_text_word)
        TEXT_CHAR.append(_text_char)  # [batch,word,char] #padding two times,
        # first in word dimensions for sentence maxlen ,then ,in char dimensions for maxlen_word
        BIO.append(_bio)
        TEXT_WORD = np.array(sentence_pad(TEXT_WORD))
        TEXT_CHAR = np.array(char_pad(TEXT_CHAR))
        BIO = np.array(sentence_pad(BIO))

        return TEXT_WORD,TEXT_CHAR,BIO

#TODO
class data_generator():
    def __init__(self,data,char2id,word2id,BIO2id,batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.char2id = char2id
        self.word2id = word2id
        self.BIO2id = BIO2id
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
                for word in _text_word:
                    chars = [self.char2id.get(_char) for _char in word]
                    _text_char.append(chars)
                _bio = [self.BIO2id.get(b) for b in bio]
                TEXT_WORD.append(_text_word)
                TEXT_CHAR.append(_text_char) #[batch,word,char] #padding two times,
                # first in word dimensions for sentence maxlen ,then ,in char dimensions for maxlen_word
                BIO.append(_bio)
                if len(TEXT_WORD) == self.batch_size or idx == index[-1]:
                    TEXT_WORD = np.array(sentence_pad(TEXT_WORD))
                    TEXT_CHAR = np.array(char_pad(TEXT_CHAR))
                    BIO = np.array(sentence_pad(BIO))
                    yield [TEXT_WORD,TEXT_CHAR,BIO ],None
                    TEXT_WORD,TEXT_CHAR,BIO =[],[],[]
