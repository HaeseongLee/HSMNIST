import tensorflow as tf 
from tensorflow.keras import Input, Model
from tensorflow.keras import applications as tfa
from tensorflow.keras.layers import UpSampling2D, Add
import numpy as np

from utils import decode
from constant import WIDTH, HEIGHT, NUM_CLASS
from layer_modules import CBR, SPP

class HSMNIST():
    def __init__(self, input_size, n_class=1):
        self.input = Input(input_size)
        self.n_class = n_class
        self.backbone = tfa.vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=self.input) 

        for l in self.backbone.layers:
            layer_type = l.name.split("_")[1]
            # fine convolusion layers, then apply l2 regularization
            if layer_type[:-1] == "conv":
                l.kernel_regularizer = tf.keras.regularizers.l2(0.001),


    def build(self, training=True):
        p3 = self.backbone.get_layer("block3_conv3").output # (104x104x256)
        p4 = self.backbone.get_layer("block4_conv3").output # (52x52x512)
        p5 = self.backbone.get_layer("block5_conv3").output # (26x26x512)
        
        #################################### NECK ####################################
                     #k, s, in , out
        p5 = SPP(p5)
        p5 = CBR(p5, (3, 1, 512, 512),training=training)
        conv = UpSampling2D(size=(2,2), interpolation="bilinear")(p5)
        
        p4 = CBR(p4, (3, 1, 512, 512),training=training)        
        p4 = Add()([p4, conv])

        conv = UpSampling2D(size=(2,2), interpolation="bilinear")(p4)
        p3 = CBR(p3, (3, 1, 256, 512),training=training)
        p3 = Add()([p3, conv])

        p5 = CBR(p5, (3, 3, 512, 512),training=training)
        p4 = CBR(p4, (3, 3, 512, 512),training=training)
        p3 = CBR(p3, (3, 3, 512, 512),training=training)

        conv = CBR(p3, (3, 3, 512, 512), downsample=True,training=training)
        p4 = Add()([p4, conv])

        conv = CBR(p4, (3, 3, 512, 512), downsample=True,training=training)
        p5 = Add()([p5, conv])
        

        #################################### HEAD ####################################
        p5 = CBR(p5, (3, 1, 512, 256),training=training)
        p5 = CBR(p5, (3, 1, 256, 128),training=training)
        p5 = CBR(p5, (3, 1, 128, 3*(5 + self.n_class)), activate=False, bn=False)

        p4 = CBR(p4, (3, 1, 512, 256),training=training)
        p4 = CBR(p4, (3, 1, 256, 128),training=training)
        p4 = CBR(p4, (3, 1, 128, 3*(5 + self.n_class)), activate=False, bn=False)

        p3 = CBR(p3, (3, 1, 512, 256),training=training)
        p3 = CBR(p3, (3, 1, 256, 128),training=training)
        p3 = CBR(p3, (3, 1, 128, 3*(5 + self.n_class)), activate=False, bn=False)


        model = Model(self.input, outputs=[p3, p4, p5])
        return model




if __name__ == "__main__":
    model = HSMNIST((WIDTH, HEIGHT, 3), NUM_CLASS).build()
    # pred = Model(inputs=model.input, outputs=decode(model.output))
    pred = Model(inputs=model.input, outputs=model.output)
    tf.keras.utils.plot_model(pred, to_file="test.png", show_shapes=True, show_layer_names=True)

