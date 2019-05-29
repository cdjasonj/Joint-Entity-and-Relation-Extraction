import pandas as pd
import numpy as np
import json
import csv
import codecs

def readFile(file_name):
    head_id_col_vector = ['token_id', 'token', "BIO", "relation", 'head']
    headfile = pd.read_csv(file_name, names=head_id_col_vector, encoding="utf-8",
                           engine='python', sep="\t", quoting=csv.QUOTE_NONE).as_matrix()
    return headfile

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
             BIOs[bio] = BIOs.get(bio,0) +1
    id2BIO = {i+1:j for i,j in enumerate(BIOs)} #padding:0
    BIO2id = {j:i for i,j in id2BIO.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2BIO, BIO2id], f, indent=4, ensure_ascii=False)

def collect_relations2id(datasets,save_file):
    BIOs = {}
    for data in datasets:
        for bio in data['BIOS']:
             BIOs[bio] = BIOs.get(bio,0) +1
    id2BIO = {i+1:j for i,j in enumerate(BIOs)} #padding:0
    BIO2id = {j:i for i,j in id2BIO.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2BIO, BIO2id], f, indent=4, ensure_ascii=False)


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

