import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.layers import Activation

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight((input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True,
                                 name='{}_W_q'.format(self.name))
        self.W_k = self.add_weight((input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True,
                                 name='{}_W_k'.format(self.name))
        self.W_v = self.add_weight((input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True,
                                 name='{}_W_c'.format(self.name))
        super(SelfAttention, self).build(input_shape)

    def call(self, x, mask=None):
        q = K.dot(x, self.W_q)
        k = K.dot(x, self.W_k)
        v = K.dot(x, self.W_v)

        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = K.batch_dot(q, k, axes=[2, 2]) / temper
        attn = Activation('softmax')(attn)
        attn_weight = K.sum(attn, axis=1)
        output = K.batch_dot(attn, v)
        output = K.sum(output, axis=1)
        return [output, attn_weight]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1])]

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        return base_config
