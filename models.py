import keras
import numpy as np
from keras.layers import *
from utils import seq_and_vec
from keras.models import Model
from keras import backend as K

class base_model():
    def __init__(self,hidden_size,embed_size,vocab_size,maxlen,num_classes_part1,num_classes_part2,dropout_prob=0.1):
        """
        base multi-head selection joint extracton model
        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param dropout_prob:
        """
        self.num_classes_part1 = num_classes_part1
        self.num_classes_part2 = num_classes_part2
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout_prob = dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>LSTM*2=>attention=>dense=>outputs
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(self.maxlen,))
        outputs_part1 = Input(shape=(self.maxlen,))
        outputs_part2 = Input(shape=(self.maxlen,))
        #word_embedding_layer
        word_embedding = Embedding(self.vocab_size,self.embed_size,name='word_embedding')(word_input)

        if self.dropout_prob:
            word_embedding = Dropout(self.dropout_prob)(word_embedding)
        #lstm layer
        lstm1 = Bidirectional(CuDNNLSTM(self.hidden_size//2,return_sequences=True,name='lstm_layer1'))(word_embedding)
        lstm2 = Bidirectional(CuDNNLSTM(self.hidden_size//2,return_sequences=True,name='lstm_layer2'))(lstm1)

        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm2)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_size)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = multiply([lstm2, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))([lstm2, attention]) #[这里考虑下用相加的方法，以及门控相加]
        entity_pred = Dense(self.num_classes_part1,activation='softmax')(lstm_attention)
        entity_model = Model([word_input],[entity_pred])

        ###############################################################################################################################
        #part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([lstm2,entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2,activation='softmax')(h)
        relation_model = Model([word_input],[multi_head_selection_pred])
        train_model = Model([word_input,outputs_part1,outputs_part2],[multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1,entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2,multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss+part2_loss)
        return entity_model,relation_model,train_model

class word_char_model_1():
    def __init__(self,hidden_size,embed_size,vocab_size,maxlen,num_classes_part1,num_classes_part2,dropout_prob=0.1):
        """
        实验拼接方式的 word_char_embedding
        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param maxlen:
        :param num_classes_part1:
        :param num_classes_part2:
        :param dropout_prob:
        """

class word_char_model_2():
    def __init__(self,hidden_size,embed_size,vocab_size,maxlen,num_classes_part1,num_classes_part2,dropout_prob=0.1):
        """
        try the concatenate way for word_char_embedding
        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param maxlen:
        :param num_classes_part1:
        :param num_classes_part2:
        :param dropout_prob:
        """

class word_char_model_3():
    def __init__(self,hidden_size,embed_size,vocab_size,maxlen,num_classes_part1,num_classes_part2,dropout_prob=0.1):
        """
        try the add way for word_char_embedding
        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param maxlen:
        :param num_classes_part1:
        :param num_classes_part2:
        :param dropout_prob:
        """

class word_char_model_4():
    def __init__(self,hidden_size,embed_size,vocab_size,maxlen,num_classes_part1,num_classes_part2,dropout_prob=0.1):
        """
        try the add way for word_char_embedding and self-attention for part1
        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param maxlen:
        :param num_classes_part1:
        :param num_classes_part2:
        :param dropout_prob:
        """