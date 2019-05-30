import json
import numpy as np
from utils import read_properties,data_generator,load_data
from comparative_model import lstm_model_ner_part,self_attention_model_ner_part
import keras
from keras.callbacks import LearningRateScheduler
from eval import NER_result_Evaluator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_file= read_properties('config/CoNLL04/bio_config')
#datasets
filename_train_me = config_file.getProperty("filename_train_me")
filename_test_me = config_file.getProperty("filename_test_me")
filename_dev_me = config_file.getProperty("filename_dev_me")

filename_char2id = config_file.getProperty("filename_char2id")
filename_word2id = config_file.getProperty("filename_word2id")
filename_BIO2id = config_file.getProperty("filename_BIO2id")
filename_relation2id = config_file.getProperty("filename_relation2id")

#training
epochs = config_file.getProperty("epochs")
batch_size = config_file.getProperty('batch_size')
model_save_file = config_file.getProperty('save_model_file')

#hyperparameters
is_use_char_embedding = config_file.getProperty('is_use_char_embedding')
hidden_size = config_file.getProperty('hidden_size')
word_embed_size = config_file.getProperty('word_embed_size')
char_embed_size = config_file.getProperty('char_embed_size')
embedding_dropout_prob  = config_file.getProperty('embedding_dropout_prob')
nn_dropout_prob = config_file.getProperty('nn_dropout_prob')
multi_layers = config_file.getProperty('multi_layers')
nb_head = config_file.getProperty('config_file')
learning_rate = config_file.getProperty('learning_rate')
# maxlen_sentence = config_file.getProperty('maxlen_sentence')
# maxlen_word = config_file.getProperty('maxlen_word')

train_data = json.load(open(filename_train_me,encoding='utf-8'))
dev_data = json.load(open(filename_dev_me,encoding='utf-8'))
id2char, char2id = json.load(open(filename_char2id,encoding='utf-8'))
id2word, word2id = json.load(open(filename_word2id,encoding='utf-8'))
id2BIO,BIO2id = json.load(open(filename_BIO2id,encoding='utf-8'))
# id2relation,relation2id = json.load(open(filename_relation2id,encoding='utf-8'))
char_vocab_size = len(char2id) +1  # 0,padding
word_vocab_size = len(word2id) +1 # 0 ,padding
ner_classes_num = len(BIO2id)

sequence_length = 100
word_length = 30
epochs = 300
word_embedding_size = 300
char_embedding_size = 300
hidden_size = 128
# lstm_model = lstm_model_ner_part(hidden_size, nb_head, word_embed_size, char_embed_size, word_vocab_size, char_vocab_size, multi_layers,
#                                  ner_classes_num, learning_rate, embedding_dropout_prob, nn_dropout_prob, is_use_char_embedding)
#
# self_att_model = self_attention_model_ner_part(hidden_size, nb_head, word_embed_size, char_embed_size, word_vocab_size, char_vocab_size, multi_layers,
#                                                ner_classes_num, learning_rate, embedding_dropout_prob, nn_dropout_prob, is_use_char_embedding)
# self_att_model = self_attention_model_ner_part(hidden_size, 8, word_embedding_size, char_embedding_size, word_vocab_size, char_vocab_size, 8,
#                                                ner_classes_num, sequence_length,word_length,1e-3, 0.15, 0.15, 'adam',False)
#
lstm_model = lstm_model_ner_part(hidden_size, 8, word_embedding_size, char_embedding_size, word_vocab_size, char_vocab_size, 3,
                                               ner_classes_num, sequence_length,word_length,1e-3, 0.25, 0.25, 'adam',False)
train_model,pred_model = lstm_model.model()

#TODO  only ner part now, then complete it

def pred_op(mode):
    eval_TEXT_WORD, eval_TEXT_CHAR, true_bio = load_data(mode)
    ner_pred = pred_model.predict([eval_TEXT_WORD, eval_TEXT_CHAR],batch_size=800,verbose=1)#[batch,sentence,num_classe]
    ner_pred = np.argmax(ner_pred,axis=-1) #[batch,sentence]
    return ner_pred,true_bio

train_data +=dev_data
def train_op():
    train_D = data_generator(train_data,char2id,word2id,BIO2id,sequence_length,word_length,128)
    best_f1 = 0
    for i in range(1,300): #epochs
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=1,
                                  )
        if (i) % 2 == 0 : #两次对dev进行一次测评,并对dev结果进行保存
            # print('进入到这里了哟~')
            ner_pred,true_bio = pred_op('test')
            P, R, F = NER_result_Evaluator(ner_pred,true_bio)
            if F > best_f1 :
                train_model.save_weights(model_save_file)
                best_f1 = F
                print('当前第{}个epoch，准确度为{},召回为{},f1为：{}'.format(i,P,R,F))

    ner_pred, true_bio = pred_op('test')
    P, R, F = NER_result_Evaluator(ner_pred, true_bio)
    print('训练集,准确度为{},召回为{},f1为：{}'.format( P, R, F))
train_op()