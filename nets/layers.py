from keras.layers import Layer,Input,Dense,Conv2D,Flatten,Droupout
import keras

class PrimaryCapsLayer(Layer):
    def __init__(self,*kwargs):
        super(PrimaryCapsLayer,self).__init__(*kwargs)
    def call(self,inputs):
        pass
    def compute_output_shape(self,input_shape):
        pass
class CapsLayer(Layer):
    def __init__(self,*kwargs):
        super(CapsLayer,self).__init__(*kwargs)
    def call(self,inputs):
        pass
    def compute_output_shape(self,input_shape):
        pass
class LengthLayer(Layer):
    def __init__(self,*kwargs):
        super(LengthLayer,self).__init__(*kwargs)
    def call(self,inputs):
        pass
    def compute_output_shape(self,input_shape):
        pass