import keras
import numpy as np
from keras.layers import *
from keras.models import Model
from layers import Position_Embedding,Attention_Layer,Self_Attention_Layer,Gate_Add_Lyaer,seq_and_vec,MaskedConv1D,MaskedLSTM,MaskFlatten,MaskPermute,MaskRepeatVector

class word_char_lstm_model():

    def __init__(self, hidden_size, word_embed_size, char_embed_size, word_vocab_size,char_vocab_size, multi_layers, maxlen, maxlen_word,
                 num_classes_part1,num_classes_part2, learning_rate=5e-5, embedding_dropout_prob=0.1,nn_dropout_prob = 0.1, optmizer='adam'):
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
        self.embedding_dropout_prob = embedding_dropout_prob
        self.nn_dropout_prob = nn_dropout_prob

    def model(self):
        """
        后面加上mask
        part1 : word_embed=>LSTM*2=>attention=>dense=>outputs
        part2 : outputs_concat LSTM => dense=>outputs
        :return:
        """
        word_input = Input(shape=(None,))  # [batch_size,sentence]
        char_input = Input(shape=(None,None,))  # [batch_size,sentence,word]
        outputs_part1 = Input(shape=(None,))
        outputs_part2 = Input(shape=(None,None,))  #[batch,sentence,sentence*rel_counts]
        # word_embedding_layer
        word_embedding = Embedding(self.word_vocab_size, self.word_embed_size, name='word_embedding')(word_input)
        char_embedding = Embedding(self.char_vocab_size, self.char_embed_size, name='char_embedding')(char_input)  # [batch,sentence,word,dim of char embedding]
        if self.embedding_dropout_prob:
            word_embedding = Dropout(self.embedding_dropout_prob)(word_embedding)
            char_embedding = Dropout(self.embedding_dropout_prob)(char_embedding)

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
        if self.nn_dropout_prob:
            lstm = Dropout(self.nn_dropout_prob)(lstm)
        # multi_lstm_layers
        if self.multi_layers >= 2:
            for i in range(self.multi_layers - 1):
                lstm = Bidirectional(
                    CuDNNLSTM(self.hidden_size // 2, return_sequences=True, name='lstm_layer{}'.format(i + 1)))(lstm)
                if self.nn_dropout_prob:
                    lstm = Dropout(self.nn_dropout_prob)(lstm)

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

        # part2 multi-head selection for relation classfication
        h = Concatenate(axis=-1)([lstm, entity_pred])
        multi_head_selection_pred = Dense(self.num_classes_part2, activation='sigmoid')(h) #[batch_size,sentence,]
        relation_model = Model([word_input], [multi_head_selection_pred])
        train_model = Model([word_input, outputs_part1, outputs_part2], [multi_head_selection_pred])

        part1_loss = K.sparse_categorical_crossentropy(outputs_part1, entity_pred)
        part2_loss = K.binary_crossentropy(outputs_part2, multi_head_selection_pred)
        part1_loss = K.mean(part1_loss)
        part2_loss = K.mean(part2_loss)

        train_model.add_loss(part1_loss + part2_loss)
        train_model.compile(keras.optimizers.adam(lr=5e-5))

        return entity_model, relation_model, train_model


class lstm_attention_model_ner_part():
    def __init__(self,embedding_martrix,hidden_size,
                 nb_head,word_embed_size,char_embed_size,word_vocab_size,char_vocab_size,multi_layers,num_classes
                 ,maxlen_sentence,maxlen_word,word_char_embed_mode='add',learning_rate = 5e-5,embedding_dropout_prob=0.1,nn_dropout_prob=0.1,optmizer='adam',
                 is_use_char_embedding=False):
        """
        测试一下self-attention在ner上的效果
        """
        self.embedding_martrix = embedding_martrix
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.nb_head = nb_head
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        # self.pos_embed_size = pos_embed_size #use the add position_embedding
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        # self.maxlen = maxlen
        self.multi_layers = multi_layers
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.word_char_embed_mode=  word_char_embed_mode
        self.learning_rate = learning_rate
        self.embedding_dropout_prob = embedding_dropout_prob
        self.nn_dropout_prob = nn_dropout_prob
        self.is_use_char_embedding = is_use_char_embedding
        print(multi_layers)

    #char_embedding_shape [batch,sentence,word,dim]
    def reshape_layer_1(self, char_embedding,char_embedding_shape):
        def reshape(char_embedding):
            return K.reshape(char_embedding, shape=(-1, char_embedding_shape[-2], self.char_embed_size)) #[batch*sentence,word,dim]
        return Lambda(reshape)(char_embedding)

    def reshape_layer_2(self, char_embedding,char_embedding_shape):
        def reshape(char_embedding):
            return K.reshape(char_embedding, shape=(-1, char_embedding_shape[1], self.char_embed_size)) #[batch,sentence,dim]
        return Lambda(reshape)(char_embedding)

    def model(self):
        word_input = Input(shape=(self.maxlen_sentence,)) #[batch,sentencen]
        char_input = Input(shape=(self.maxlen_sentence,self.maxlen_word,)) #[batch,word,char]
        ner_label = Input(shape=(self.maxlen_sentence,))

        mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(word_input)

        word_embedding = Embedding(self.word_vocab_size, self.word_embed_size,weights=[self.embedding_martrix],name='word_embedding',trainable=True)(word_input) #[batch,word,embed]
        char_embedding = Embedding(self.char_vocab_size,self.char_embed_size,name='char_embedding',trainable=True)(char_input) #[batch,word,char,embedd]

        if self.embedding_dropout_prob:
            word_embedding = Dropout(self.embedding_dropout_prob)(word_embedding)
            char_embedding = Dropout(self.embedding_dropout_prob)(char_embedding)

        if self.is_use_char_embedding:
            # char_embedding maxpooling part
            char_embedding_shape = K.int_shape(char_embedding)  # [batch,sentence,word,dim]
            # char_embedding_reshaped = K.reshape(char_embedding, shape=(-1, char_embedding_shape[-2],self.char_embed_size))  # [batch*sentence,word,dim of char embedding]
            char_embedding_reshaped = self.reshape_layer_1(char_embedding,char_embedding_shape)
            char_lstm = Bidirectional(MaskedLSTM(units=self.char_embed_size // 2, return_sequences=True, name='char_lstm_layer'))(
                char_embedding_reshaped)

            attention = TimeDistributed(Dense(1, activation='tanh'))(char_lstm)
            attention = MaskFlatten()(attention)
            attention = Activation('softmax')(attention)
            attention = MaskRepeatVector(self.char_embed_size)(attention)
            attention = MaskPermute([2, 1])(attention)
            sent_representation = multiply([char_lstm, attention])
            attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

            # char_maxpool = GlobalMaxPooling1D(char_lstm)  # [batch*sentence,hidden_size]
            # char_att = Attention_Layer()(char_lstm)  # [batch*sentence,hidden_size]
            # char_embedding = K.reshape(char_maxpool, shape=[-1, char_embedding_shape[1],
            #                                                 self.hidden_size])  # [batch,sentence,hidden_size]
            # char_embedding = K.reshape(attention, shape=[-1, char_embedding_shape[-1], self.char_embed_size])  # [batch,sentence,hidden_size]
            char_embedding = self.reshape_layer_2(attention,char_embedding_shape)
            if  self.word_char_embed_mode == 'concate':
                embedding = Concatenate(axis=-1)([word_embedding,char_embedding])
            else :
                embedding = Gate_Add_Lyaer()([word_embedding,char_embedding])
                # pass
        else:
            embedding = word_embedding
        #multi-layers self-attention for ner pred
        if self.embedding_dropout_prob:
            embedding = Dropout(self.embedding_dropout_prob)(embedding)

        # part1 , multi-self-attentionblock, (CNN/LSTM/FNN+self-attention)
        lstm = Bidirectional(MaskedLSTM(units=self.hidden_size // 2, return_sequences=True), name='lstm_layer0')(embedding)
        if self.nn_dropout_prob:
            lstm = Dropout(self.nn_dropout_prob)(lstm)
        # # multi_lstm_layers
        # if self.multi_layers >= 2:
        #     for i in range(self.multi_layers - 1):
        #         i+=1
        #         lstm = Bidirectional(CuDNNLSTM(self.hidden_size // 2, return_sequences=True), name='lstm_layer{}'.format(i))(lstm)
        #         if self.nn_dropout_prob:
        #             lstm = Dropout(self.nn_dropout_prob)(lstm)

        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        #
        attention = MaskFlatten()(attention)
        attention = Activation('softmax')(attention)
        attention = MaskRepeatVector(self.hidden_size)(attention)
        attention = MaskPermute([2, 1])(attention)
        sent_representation = multiply([lstm, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))(
            [lstm, attention])  # [这里考虑下用相加的方法，以及门控相加]
        lstm_attention = MaskedConv1D(filters=self.hidden_size,kernel_size=3,activation='relu',padding='same')(lstm_attention)

        bio_pred = Dense(self.num_classes, activation='softmax')(lstm_attention)
        pred_model =Model([word_input, char_input], bio_pred)
        train_model = Model([word_input, char_input, ner_label], bio_pred)

        loss = K.sparse_categorical_crossentropy(ner_label, bio_pred)
        loss = K.sum(loss * mask[:, :, 0]) / K.sum(mask)

        loss = K.sum(loss * mask) / K.sum(mask)
        train_model.summary()
        train_model.add_loss(loss)
        train_model.compile(keras.optimizers.adam(lr=self.learning_rate))

        return train_model,pred_model
