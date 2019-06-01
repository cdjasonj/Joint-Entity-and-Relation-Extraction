from layers import Position_Embedding,Attention_Layer,Self_Attention_Layer,Gate_Add_Lyaer,seq_and_vec,MaskedConv1D,MaskedLSTM,MaskFlatten,MaskPermute,MaskRepeatVector
from keras.models import Model
from keras.layers import *
import keras
from keras_contrib.layers import CRF
from keras_multi_head import MultiHead,MultiHeadAttention
from keras_self_attention import SeqSelfAttention as self_attention
from keras_pos_embd  import TrigPosEmbedding
from keras_position_wise_feed_forward import FeedForward

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


class lstm_model_ner_part():
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
        bio_pred = Dense(self.num_classes, activation='softmax')(lstm)
        pred_model =Model([word_input, char_input], bio_pred)
        train_model = Model([word_input, char_input, ner_label], bio_pred)

        loss = K.sparse_categorical_crossentropy(ner_label, bio_pred)
        loss = K.sum(loss * mask[:, :, 0]) / K.sum(mask)

        loss = K.sum(loss * mask) / K.sum(mask)
        train_model.summary()
        train_model.add_loss(loss)
        train_model.compile(keras.optimizers.adam(lr=self.learning_rate))

        return train_model,pred_model


