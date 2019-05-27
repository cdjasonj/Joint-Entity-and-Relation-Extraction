import keras
import numpy as np
from keras.layers import *
from utils import seq_and_vec
from keras.models import Model
from keras import backend as K

"""
5/28 ，为LSTM,OUTPUTS,EMBEDDING都加上dropout
完成attention_layer,Gate_Add_Layer
"""
class Attention_Layer(keras.layers.Layer):
    """
    dot attention for word_char_embedding
    """

class Gate_Add_Lyaer(keras.layers.Layer):
    """
    gate add mechanism for word_char embedding
    z =  sigmoid(W(1)tanh(W(2)word_embedding + W(3)char_att))
    word_char_embedding = z*word_embedding + (1-z)char_att
    """

class Self_Attention_Layer(keras.layers.Layer):
    """
    use the self_attention to replace the lstm layer
    2018-AAAI-deep semantic role labeling with self-attention
    multi RNN/CNN/FNN + self-attention(multi-head attention)
    """
    def __init__(self,multi_layers):
        self.multi_layers = multi_layers

class base_model():
    def __init__(self,hidden_size,embed_size,vocab_size,multi_layers,maxlen,num_classes_part1,num_classes_part2,
                 learning_rate = 5e-5,dropout_prob=0.1,optmizer='adam'):
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
        self.multi_layers = multi_layers
        self.learning_rate = learning_rate
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

        lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer0'))(word_embedding)
        #multi_lstm_layers
        if self.multi_layers >= 2:
            for i in range(self.multi_layers -1 ):
                lstm = Bidirectional(CuDNNLSTM(self.hidden_size//2,return_sequences=True,name='lstm_layer{}'.format(i+1)))(lstm)

        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_size)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = multiply([lstm, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))([lstm, attention]) #[这里考虑下用相加的方法，以及门控相加]
        entity_pred = Dense(self.num_classes_part1,activation='softmax')(lstm_attention)
        entity_model = Model([word_input],[entity_pred])

        ###############################################################################################################################
        #part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([lstm,entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2,activation='softmax')(h)
        relation_model = Model([word_input],[multi_head_selection_pred])
        train_model = Model([word_input,outputs_part1,outputs_part2],[multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1,entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2,multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss+part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model,relation_model,train_model

class word_char_model_1():
    def __init__(self,hidden_size,word_embed_size,char_embed_size,word_vocab_size,char_vocab_size,multi_layers,maxlen,maxlen_word,num_classes_part1,
                 num_classes_part2,learning_rate = 5e-5,dropout_prob=0.1,optmizer='adam'):
        """
        try the concatenate way for word_char_embedding，char_embedding from the char-level of word maxpooling

        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param dropout_prob:
        """
        self.num_classes_part1 = num_classes_part1
        self.num_classes_part2 = num_classes_part2
        self.hidden_size = hidden_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.maxlen = maxlen
        self.maxlen_word = maxlen_word
        self.multi_layers = multi_layers
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>LSTM*2=>attention=>dense=>outputs
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(self.maxlen,)) #[batch_size,sentence]
        char_input = Input(shape=(self.maxlen,self.maxlen_word,)) #[batch_size,sentence,word]
        outputs_part1 = Input(shape=(self.maxlen,))
        outputs_part2 = Input(shape=(self.maxlen,))
        #word_embedding_layer
        word_embedding = Embedding(self.word_vocab_size,self.word_embed_size,name='word_embedding')(word_input)
        char_embedding = Embedding(self.char_vocab_size,self.char_embed_size,name='char_embedding')(char_input) #[batch,sentence,word,dim of char embedding]
        if self.dropout_prob:
            word_embedding = Dropout(self.dropout_prob)(word_embedding)
            char_embedding = Dropout(self.dropout_prob)(char_embedding)

        #char_embedding maxpooling part
        char_embedding_shape = K.int_shape(char_embedding)#[batch,sentence,word,dim]
        char_embedding_reshaped = K.reshape(char_embedding,shape=[-1,char_embedding_shape[-2],self.char_embed_size]) #[batch*sentence,word,dim of char embedding]
        char_lstm = Bidirectional(CuDNNLSTM(self.hidden_size//2,return_sequences=True,name='char_lstm_layer'))(char_embedding_reshaped)
        char_maxpool = GlobalMaxPooling1D(char_lstm) #[batch*sentence,hidden_size]
        char_embedding = K.reshape(char_maxpool,shape=[-1,char_embedding_shape[1],self.hidden_size]) #[batch,sentence,hidden_size]
        embedding = Concatenate(axis=-1)([word_embedding,char_embedding])

        #part1 , entity_pred
        lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer0'))(embedding)
        #multi_lstm_layers
        if self.multi_layers >= 2:
            for i in range(self.multi_layers -1 ):
                lstm = Bidirectional(CuDNNLSTM(self.hidden_size//2,return_sequences=True,name='lstm_layer{}'.format(i+1)))(lstm)

        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_size)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = multiply([lstm, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))([lstm, attention]) #[这里考虑下用相加的方法，以及门控相加]
        entity_pred = Dense(self.num_classes_part1,activation='softmax')(lstm_attention)
        entity_model = Model([word_input],[entity_pred])

        ###############################################################################################################################
        #part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([lstm,entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2,activation='softmax')(h)
        relation_model = Model([word_input],[multi_head_selection_pred])
        train_model = Model([word_input,outputs_part1,outputs_part2],[multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1,entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2,multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss+part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model,relation_model,train_model

class word_char_model_2():
    def __init__(self, hidden_size, word_embed_size, char_embed_size, word_vocab_size,char_vocab_size, multi_layers, maxlen, maxlen_word,
                 num_classes_part1,num_classes_part2, learning_rate=5e-5, dropout_prob=0.1, optmizer='adam'):
        """
        try the concatenate way for word_char_embedding，char_embedding from the char-level of word attention

        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param dropout_prob:
        """
        self.num_classes_part1 = num_classes_part1
        self.num_classes_part2 = num_classes_part2
        self.hidden_size = hidden_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.maxlen = maxlen
        self.maxlen_word = maxlen_word
        self.multi_layers = multi_layers
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>LSTM*2=>attention=>dense=>outputs
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(self.maxlen,))  # [batch_size,sentence]
        char_input = Input(shape=(self.maxlen, self.maxlen_word,))  # [batch_size,sentence,word]
        outputs_part1 = Input(shape=(self.maxlen,))
        outputs_part2 = Input(shape=(self.maxlen,))
        # word_embedding_layer
        word_embedding = Embedding(self.word_vocab_size, self.word_embed_size, name='word_embedding')(word_input)
        char_embedding = Embedding(self.char_vocab_size, self.char_embed_size, name='char_embedding')(
            char_input)  # [batch,sentence,word,dim of char embedding]
        if self.dropout_prob:
            word_embedding = Dropout(self.dropout_prob)(word_embedding)
            char_embedding = Dropout(self.dropout_prob)(char_embedding)

        # char_embedding maxpooling part
        char_embedding_shape = K.int_shape(char_embedding)  # [batch,sentence,word,dim]
        char_embedding_reshaped = K.reshape(char_embedding, shape=[-1, char_embedding_shape[-2],
                                                                   self.char_embed_size])  # [batch*sentence,word,dim of char embedding]
        char_lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='char_lstm_layer'))(
            char_embedding_reshaped)
        # char_maxpool = GlobalMaxPooling1D(char_lstm)  # [batch*sentence,hidden_size]
        char_att = Attention_Layer()(char_lstm) #[batch*sentence,hidden_size]
        # char_embedding = K.reshape(char_maxpool, shape=[-1, char_embedding_shape[1],
        #                                                 self.hidden_size])  # [batch,sentence,hidden_size]
        char_embedding = K.reshape(char_att,shape=[-1,char_embedding_shape[-1],self.hidden_size])#[batch,sentence,hidden_size]
        embedding = Concatenate(axis=-1)([word_embedding, char_embedding])

        # part1 , entity_pred
        lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer0'))(embedding)
        # multi_lstm_layers
        if self.multi_layers >= 2:
            for i in range(self.multi_layers - 1):
                lstm = Bidirectional(
                    CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer{}'.format(i + 1)))(lstm)

        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_size)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = multiply([lstm, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))(
            [lstm, attention])  # [这里考虑下用相加的方法，以及门控相加]
        entity_pred = Dense(self.num_classes_part1, activation='softmax')(lstm_attention)
        entity_model = Model([word_input], [entity_pred])

        ###############################################################################################################################
        # part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([lstm, entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2, activation='softmax')(h)
        relation_model = Model([word_input], [multi_head_selection_pred])
        train_model = Model([word_input, outputs_part1, outputs_part2], [multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1, entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2, multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss + part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model, relation_model, train_model

class word_char_model_3():


    def __init__(self, hidden_size, word_embed_size, char_embed_size, word_vocab_size,char_vocab_size, multi_layers, maxlen, maxlen_word,
                 num_classes_part1,num_classes_part2, learning_rate=5e-5, dropout_prob=0.1, optmizer='adam'):
        """
        try the add way for word_char_embedding, char_embedding from the char-leve of word maxpool

        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param dropout_prob:
        """
        self.num_classes_part1 = num_classes_part1
        self.num_classes_part2 = num_classes_part2
        self.hidden_size = hidden_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.maxlen = maxlen
        self.maxlen_word = maxlen_word
        self.multi_layers = multi_layers
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>LSTM*2=>attention=>dense=>outputs
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(self.maxlen,))  # [batch_size,sentence]
        char_input = Input(shape=(self.maxlen, self.maxlen_word,))  # [batch_size,sentence,word]
        outputs_part1 = Input(shape=(self.maxlen,))
        outputs_part2 = Input(shape=(self.maxlen,))
        # word_embedding_layer
        word_embedding = Embedding(self.word_vocab_size, self.word_embed_size, name='word_embedding')(word_input)
        char_embedding = Embedding(self.char_vocab_size, self.char_embed_size, name='char_embedding')(char_input)  # [batch,sentence,word,dim of char embedding]
        if self.dropout_prob:
            word_embedding = Dropout(self.dropout_prob)(word_embedding)
            char_embedding = Dropout(self.dropout_prob)(char_embedding)

        # char_embedding maxpooling part
        char_embedding_shape = K.int_shape(char_embedding)  # [batch,sentence,word,dim]
        char_embedding_reshaped = K.reshape(char_embedding, shape=[-1, char_embedding_shape[-2],
                                                                   self.char_embed_size])  # [batch*sentence,word,dim of char embedding]
        char_lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='char_lstm_layer'))(
            char_embedding_reshaped)
        char_maxpool = GlobalMaxPooling1D(char_lstm)  # [batch*sentence,hidden_size]
        # char_att = Attention_Layer()(char_lstm) #[batch*sentence,hidden_size]
        char_embedding = K.reshape(char_maxpool, shape=[-1, char_embedding_shape[1],
                                                        self.hidden_size])  # [batch,sentence,hidden_size]
        # char_embedding = K.reshape(char_att,shape=[-1,char_embedding_shape[-1],self.hidden_size])#[batch,sentence,hidden_size]
        # embedding = Concatenate(axis=-1)([word_embedding, char_embedding])
        embedding = Gate_Add_Lyaer()([word_embedding,char_embedding]) #gate add ==> embedding
        # part1 , entity_pred
        lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer0'))(embedding)
        # multi_lstm_layers
        if self.multi_layers >= 2:
            for i in range(self.multi_layers - 1):
                lstm = Bidirectional(
                    CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer{}'.format(i + 1)))(lstm)

        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_size)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = multiply([lstm, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))(
            [lstm, attention])  # [这里考虑下用相加的方法，以及门控相加]
        entity_pred = Dense(self.num_classes_part1, activation='softmax')(lstm_attention)
        entity_model = Model([word_input], [entity_pred])

        ###############################################################################################################################
        # part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([lstm, entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2, activation='softmax')(h)
        relation_model = Model([word_input], [multi_head_selection_pred])
        train_model = Model([word_input, outputs_part1, outputs_part2], [multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1, entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2, multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss + part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model, relation_model, train_model

class word_char_model_4():


    def __init__(self, hidden_size, word_embed_size, char_embed_size, word_vocab_size,char_vocab_size, multi_layers, maxlen, maxlen_word,
                 num_classes_part1,num_classes_part2, learning_rate=5e-5, dropout_prob=0.1, optmizer='adam'):
        """
        try the gate add way for word_char_embedding, char_embedding from the char-leve of word attention

        :param hidden_size:
        :param embed_size:
        :param vocab_size:
        :param dropout_prob:
        """
        self.num_classes_part1 = num_classes_part1
        self.num_classes_part2 = num_classes_part2
        self.hidden_size = hidden_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.maxlen = maxlen
        self.maxlen_word = maxlen_word
        self.multi_layers = multi_layers
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>LSTM*2=>attention=>dense=>outputs
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(self.maxlen,))  # [batch_size,sentence]
        char_input = Input(shape=(self.maxlen, self.maxlen_word,))  # [batch_size,sentence,word]
        outputs_part1 = Input(shape=(self.maxlen,))
        outputs_part2 = Input(shape=(self.maxlen,))
        # word_embedding_layer
        word_embedding = Embedding(self.word_vocab_size, self.word_embed_size, name='word_embedding')(word_input)
        char_embedding = Embedding(self.char_vocab_size, self.char_embed_size, name='char_embedding')(char_input)  # [batch,sentence,word,dim of char embedding]
        if self.dropout_prob:
            word_embedding = Dropout(self.dropout_prob)(word_embedding)
            char_embedding = Dropout(self.dropout_prob)(char_embedding)

        # char_embedding maxpooling part
        char_embedding_shape = K.int_shape(char_embedding)  # [batch,sentence,word,dim]
        char_embedding_reshaped = K.reshape(char_embedding, shape=[-1, char_embedding_shape[-2],
                                                                   self.char_embed_size])  # [batch*sentence,word,dim of char embedding]
        char_lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='char_lstm_layer'))(
            char_embedding_reshaped)
        # char_maxpool = GlobalMaxPooling1D(char_lstm)  # [batch*sentence,hidden_size]
        char_att = Attention_Layer()(char_lstm) #[batch*sentence,hidden_size]
        # char_embedding = K.reshape(char_maxpool, shape=[-1, char_embedding_shape[1],
        #                                                 self.hidden_size])  # [batch,sentence,hidden_size]
        char_embedding = K.reshape(char_att,shape=[-1,char_embedding_shape[-1],self.hidden_size])#[batch,sentence,hidden_size]
        # embedding = Concatenate(axis=-1)([word_embedding, char_embedding])
        embedding = Gate_Add_Lyaer()([word_embedding,char_embedding])
        # part1 , entity_pred
        lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer0'))(embedding)
        # multi_lstm_layers
        if self.multi_layers >= 2:
            for i in range(self.multi_layers - 1):
                lstm = Bidirectional(
                    CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer{}'.format(i + 1)))(lstm)

        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_size)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = multiply([lstm, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))(
            [lstm, attention])  # [这里考虑下用相加的方法，以及门控相加]
        entity_pred = Dense(self.num_classes_part1, activation='softmax')(lstm_attention)
        entity_model = Model([word_input], [entity_pred])

        ###############################################################################################################################
        # part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([lstm, entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2, activation='softmax')(h)
        relation_model = Model([word_input], [multi_head_selection_pred])
        train_model = Model([word_input, outputs_part1, outputs_part2], [multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1, entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2, multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss + part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model, relation_model, train_model

class self_attention_base_model():

    def __init__(self,hidden_size,embed_size,vocab_size,multi_layers,maxlen,maxlen_word,num_classes_part1,
                 num_classes_part2,learning_rate = 5e-5,dropout_prob=0.1,optmizer='adam'):
        """
        try the gate add way for word_char_embedding and self-attention for part1 without word_char embedding

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
        self.multi_layers = multi_layers
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>multi self attention layer
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(self.maxlen,))
        outputs_part1 = Input(shape=(self.maxlen,))
        outputs_part2 = Input(shape=(self.maxlen,))
        # word_embedding_layer
        word_embedding = Embedding(self.vocab_size, self.embed_size, name='word_embedding')(word_input)

        if self.dropout_prob:
            word_embedding = Dropout(self.dropout_prob)(word_embedding)

        self_attention = Self_Attention_Layer(self.multi_layers)(word_embedding)
        entity_pred = Dense(self.num_classes_part1, activation='softmax')(self_attention)
        entity_model = Model([word_input], [entity_pred])
        ###############################################################################################################################
        # part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([self_attention, entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2, activation='softmax')(h)
        relation_model = Model([word_input], [multi_head_selection_pred])
        train_model = Model([word_input, outputs_part1, outputs_part2], [multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1, entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2, multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss + part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model, relation_model, train_model

class self_attention_word_char_model():
    def __init__(self, hidden_size, word_embed_size, char_embed_size, word_vocab_size,char_vocab_size, multi_layers, maxlen, maxlen_word,
                 num_classes_part1,num_classes_part2, learning_rate=5e-5, dropout_prob=0.1, optmizer='adam'):
        """
        try the gate add way for word_char_embedding, char_embedding from the char-leve of word attention in the self-attention structure

        :param hidden_size:
        :param word_embed_size:
        :param char_embed_size:
        :param word_vocab_size:
        :param char_vocab_size:
        :param multi_layers:
        :param maxlen:
        :param maxlen_word:
        :param num_classes_part1:
        :param num_classes_part2:
        :param learning_rate:
        :param dropout_prob:
        :param optmizer:
        """

        self.num_classes_part1 = num_classes_part1
        self.num_classes_part2 = num_classes_part2
        self.hidden_size = hidden_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.maxlen = maxlen
        self.maxlen_word = maxlen_word
        self.multi_layers = multi_layers
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>LSTM*2=>attention=>dense=>outputs
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(self.maxlen,))  # [batch_size,sentence]
        char_input = Input(shape=(self.maxlen, self.maxlen_word,))  # [batch_size,sentence,word]
        outputs_part1 = Input(shape=(self.maxlen,))
        outputs_part2 = Input(shape=(self.maxlen,))
        # word_embedding_layer
        word_embedding = Embedding(self.word_vocab_size, self.word_embed_size, name='word_embedding')(word_input)
        char_embedding = Embedding(self.char_vocab_size, self.char_embed_size, name='char_embedding')(char_input)  # [batch,sentence,word,dim of char embedding]
        if self.dropout_prob:
            word_embedding = Dropout(self.dropout_prob)(word_embedding)
            char_embedding = Dropout(self.dropout_prob)(char_embedding)

        # char_embedding maxpooling part
        char_embedding_shape = K.int_shape(char_embedding)  # [batch,sentence,word,dim]
        char_embedding_reshaped = K.reshape(char_embedding, shape=[-1, char_embedding_shape[-2],
                                                                   self.char_embed_size])  # [batch*sentence,word,dim of char embedding]
        char_lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='char_lstm_layer'))(
            char_embedding_reshaped)
        # char_maxpool = GlobalMaxPooling1D(char_lstm)  # [batch*sentence,hidden_size]
        char_att = Attention_Layer()(char_lstm) #[batch*sentence,hidden_size]
        # char_embedding = K.reshape(char_maxpool, shape=[-1, char_embedding_shape[1],
        #                                                 self.hidden_size])  # [batch,sentence,hidden_size]
        char_embedding = K.reshape(char_att,shape=[-1,char_embedding_shape[-1],self.hidden_size])#[batch,sentence,hidden_size]
        # embedding = Concatenate(axis=-1)([word_embedding, char_embedding])
        embedding = Gate_Add_Lyaer()([word_embedding,char_embedding])
        # part1 , self-attention-entity_pred
        self_attention = Self_Attention_Layer(self.multi_layers)([embedding])
        entity_pred = Dense(self.num_classes_part1, activation='softmax')(self_attention)
        entity_model = Model([word_input], [entity_pred])

        ###############################################################################################################################
        # part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([self_attention, entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2, activation='softmax')(h)
        relation_model = Model([word_input], [multi_head_selection_pred])
        train_model = Model([word_input, outputs_part1, outputs_part2], [multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1, entity_pred)
        part2_loss = K.sparse_categorical_crossentropy(outputs_part2, multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss + part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model, relation_model, train_model