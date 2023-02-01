import tensorflow as tf 
from tensorflow.keras.layers import (
    Conv2D, 
    BatchNormalization, 
    ReLU, 
    ZeroPadding2D,
    concatenate
)


def CBR(input_layer, filters_shape, downsample=False, activate=True, bn=True, training=True):
    '''
        filters_shape = (k, s, c_in, c_out)
    '''
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        s = 2
        p = 'valid'
    else:
        s = 1
        p = 'same'

    c_out = filters_shape[-1]
    k = filters_shape[0]

    conv = Conv2D(filters=c_out, kernel_size=k, strides=s, padding=p,
                    use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: 
        #TODO SET TRAINING=TRUE/FALSE FOR BATCHNORMALIZATION!!
        conv = BatchNormalization()(conv, training=training)        
    if activate:
        conv = ReLU()(conv)

    return conv

def SPP(input_layer, kernels=(5,9,13), s=1, padding="SAME"):
    p1 = tf.nn.max_pool(input_layer, kernels[0], strides=s, padding=padding)
    p2 = tf.nn.max_pool(input_layer, kernels[1], strides=s, padding=padding)
    p3 = tf.nn.max_pool(input_layer, kernels[2], strides=s, padding=padding)    
    return concatenate([input_layer, p1, p2, p3])


# def bottleneck1(self):
#     pass # not define yet

# def bottleneck2(self, input_layer, input_channel, output_channel):
#     x = self.convolutional(input_layer, (1, 1, input_channel, output_channel), activate=False, bn=False)
#     x = self.convolutional(x       , (3, 1, input_channel,   output_channel), activate=False, bn=False)
#     return x
