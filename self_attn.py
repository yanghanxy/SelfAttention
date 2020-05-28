import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.layers import Activation

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
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
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform', trainable=True,
                                   name='{}_W_q'.format(self.name))
        self.W_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform', trainable=True,
                                   name='{}_W_k'.format(self.name))
        self.W_v = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform', trainable=True,
                                   name='{}_W_c'.format(self.name))
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            input_shape = K.int_shape(inputs)
            broadcast_shape = [-1, input_shape[1], 1]
            mask = K.reshape(mask, broadcast_shape)
            inputs *= mask

        q = K.dot(inputs, self.W_q)
        k = K.dot(inputs, self.W_k)
        v = K.dot(inputs, self.W_v)

        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = K.batch_dot(q, k, axes=[2, 2]) / temper
        if mask is not None:
            attn *= mask
        attn = K.softmax(attn)
        attn_weight = K.mean(attn, axis=1)
        output = K.batch_dot(attn, v)
        if mask is not None:
            output *= mask
            attn_weight *= mask
        return [output, attn_weight]

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return [mask, mask[:-1]]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1], input_shape[2]), (input_shape[0], input_shape[1])]

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        return base_config

