import keras
from keras import backend as K
from keras.layers import *

class Attention_Layer(keras.layers.Layer):

    """
    dot attention for word_char_embedding
    q,v,k for define the attention
    score = softmax(dot(q,v))
    attention = sum(score*k)=
    """
    # def __init__(self):


class Gate_Add_Lyaer(keras.layers.Layer):
    """
    gate add mechanism for word_char embedding
    z =  sigmoid(W(1)tanh(W(2)word_embedding + W(3)char_att))
    word_char_embedding = z*word_embedding + (1-z)char_att

    """
    def __init__(self,**kwargs):
        """

        :param word_embedding:  shape [batch,sentence,dim of word_embedding]
        :param char_att:  shape [batch,sentence,dim of char_embedding]
        :param kwargs:
        """
        super(Gate_Add_Lyaer,self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[0][2] == input_shape[1][2]

        self.W1 = self.add_weight(name='W1',shape=(input_shape[0][-1],input_shape[0][-1]),initializer='glorot_normal') #[dim,dim]
        self.W2 = self.add_weight(name='W2',shape=(input_shape[0][-1],input_shape[0][-1]),initializers='glorot_normal')
        self.W3 = self.add_weight(name='W3',shape=(input_shape[0][-1],input_shape[0][-1]),initializers='glorot_normal')

        super(Gate_Add_Lyaer, self).build(input_shape)

    def call(self,inputs,mask=None):
        # inputs[0]:word_embedding ,inputs[1]:char_embedding
        word_embedding_shape = K.int_shape(inputs[0]) #[batch,sentence,dim of word embedding]
        char_embedding_shape = K.int_shape(inputs[1]) #[batch,sentence,dim of char embedding]
        word_embedding_reshaped = K.reshape(inputs[0],shape=(-1,word_embedding_shape[-1])) #[batch*sentence,dim of word embedding]
        char_embedding_reshaped = K.reshape(inputs[1],shape=(-1,char_embedding_shape[-1])) #[batch*sentence, dim of char embedding]
        z = K.sigmoid(K.dot(K.tanh(K.dot(word_embedding_reshaped,self.W1) + K.dot(char_embedding_shape,self.W2)),self.W3))
        embedding = z*word_embedding_reshaped + (1-z)*char_embedding_reshaped  #[batch*sentence,]
        embedding = K.reshape(embedding,shape=(-1,word_embedding_reshaped[1],word_embedding_reshaped[-1]))# [batch,sentecen,dim]

        return embedding

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2])


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x,mask=None):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Self_Attention_Layer(Layer):

    def __init__(self, nb_head, size_per_head, mask_right=False, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.mask_right = mask_right
        super(Self_Attention_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Self_Attention_Layer, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x,mask=None):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e12
            A = A - mask
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)