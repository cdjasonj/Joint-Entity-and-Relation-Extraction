import json
import numpy as np
from utils import read_properties,data_generator,load_data,get_embedding_matrix
from comparative_model import lstm_model_ner_part,lstm_attention_model_ner_part
import keras
import keras.backend as K
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
is_use_n_char = bool(config_file.getProperty('is_use_n_char'))

#hyperparameters
is_use_char_embedding = bool(config_file.getProperty('is_use_char_embedding'))
hidden_size = int(config_file.getProperty('hidden_size'))
word_embed_size = int(config_file.getProperty('word_embed_size'))
char_embed_size = int(config_file.getProperty('char_embed_size'))
embedding_dropout_prob  = float(config_file.getProperty('embedding_dropout_prob'))
nn_dropout_prob = float(config_file.getProperty('nn_dropout_prob'))
multi_layers = int(config_file.getProperty('multi_layers'))
nb_head = int(config_file.getProperty('nb_head'))
learning_rate = float(config_file.getProperty('learning_rate'))
maxlen_sentence = int(config_file.getProperty('maxlen_sentence'))
maxlen_word = int(config_file.getProperty('maxlen_word'))

train_data = json.load(open(filename_train_me,encoding='utf-8'))
dev_data = json.load(open(filename_dev_me,encoding='utf-8'))
id2char, char2id = json.load(open(filename_char2id,encoding='utf-8'))
id2n_char, n_char2id = json.load(open(filename_char2id,encoding='utf-8'))
id2word, word2id = json.load(open(filename_word2id,encoding='utf-8'))
id2BIO,BIO2id = json.load(open(filename_BIO2id,encoding='utf-8'))
# id2relation,relation2id = json.load(open(filename_relation2id,encoding='utf-8'))
char_vocab_size = len(char2id) +1  # 0,padding
word_vocab_size = len(word2id) +1 # 0 ,padding
ner_classes_num = len(BIO2id)
embedding_martrix = get_embedding_matrix(word2id)

# lstm_model = lstm_model_ner_part(hidden_size, nb_head, word_embed_size, char_embed_size, word_vocab_size, char_vocab_size, multi_layers,
#                                  ner_classes_num, learning_rate, embedding_dropout_prob, nn_dropout_prob, is_use_char_embedding)
#
# self_att_model = self_attention_model_ner_part(hidden_size, nb_head, word_embed_size, char_embed_size, word_vocab_size, char_vocab_size, multi_layers,
#                                                ner_classes_num, learning_rate, embedding_dropout_prob, nn_dropout_prob, is_use_char_embedding)
# self_att_model = self_attention_model_ner_part(embedding_martrix,hidden_size, 5, word_embed_size, char_embed_size, word_vocab_size, char_vocab_size, 5,
#                                                ner_classes_num, maxlen_sentence,maxlen_word,learning_rate, embedding_dropout_prob, nn_dropout_prob, 'adam',False)

word_char_embed_mode = 'concate'
# lstm_model = lstm_model_ner_part(embedding_martrix,hidden_size, nb_head, word_embed_size, char_embed_size, word_vocab_size, char_vocab_size, multi_layers,
#                                 ner_classes_num, maxlen_sentence,maxlen_word,word_char_embed_mode,learning_rate,embedding_dropout_prob,nn_dropout_prob,'adam',True)
lstm_model = lstm_attention_model_ner_part(embedding_martrix,hidden_size, nb_head, word_embed_size, char_embed_size, word_vocab_size, char_vocab_size, multi_layers,
                                ner_classes_num, maxlen_sentence,maxlen_word,word_char_embed_mode,learning_rate,embedding_dropout_prob,nn_dropout_prob,'adam',True)

train_model,pred_model = lstm_model.model()

#TODO  only ner part now, then complete it

def pred_op(mode):
    eval_TEXT_WORD, eval_TEXT_CHAR, true_bio = load_data(mode)
    ner_pred = pred_model.predict([eval_TEXT_WORD, eval_TEXT_CHAR],batch_size=800,verbose=1)#[batch,sentence,num_classe]
    ner_pred = np.argmax(ner_pred,axis=-1) #[batch,sentence]
    return ner_pred,true_bio

#
# def scheduler(epoch):
#     # 每隔1个epoch，学习率减小为原来的1/2
#     # if epoch % 100 == 0 and epoch != 0:
#     #再epoch > 3的时候,开始学习率递减,每次递减为原来的1/2,最低为2e-6
#     if (epoch+1) % 50 == 0:
#         lr = K.get_value(train_model.optimizer.lr)
#         lr = lr*0.5
#         if lr < 2e-6:
#             return 2e-6
#         else:
#             return lr

def train_op():
    # reduce_lr = LearningRateScheduler(scheduler, verbose=1)
    train_D = data_generator(train_data,char2id,n_char2id,word2id,BIO2id,maxlen_sentence,maxlen_word,is_use_n_char,128)
    best_f1 = 0
    for i in range(1,150): #epochs
        print(i)
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=1,
                                  # callbacks=[reduce_lr]
                                  )
        # if (i) % 2 == 0 : #两次对dev进行一次测评,并对dev结果进行保存
        ner_pred,true_bio = pred_op('dev')
        P, R, F = NER_result_Evaluator(ner_pred,true_bio)
        if F > best_f1 :
            train_model.save_weights(model_save_file)
            best_f1 = F
            print('当前第{}个epoch，验证集,准确度为{},召回为{},f1为：{}'.format(i,P,R,F))

            ner_pred, true_bio = pred_op('test')
            P, R, F = NER_result_Evaluator(ner_pred, true_bio)
            print('当前第{}个epoch，测试集,准确度为{},召回为{},f1为：{}'.format(i,P,R,F))

        if i % 50 == 0:
            ner_pred, true_bio = pred_op('train')
            P, R, F = NER_result_Evaluator(ner_pred, true_bio)
            print('训练集,准确度为{},召回为{},f1为：{}'.format(P, R, F))

    print(best_f1)
train_op()