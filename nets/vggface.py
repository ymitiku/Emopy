from nets.base import NeuralNet
from keras.layers import Flatten,Dense
from keras_vggface.vggface import VGGFace


class VGGFaceEmopyNet(NeuralNet):
    def __init__(self,input_shape,preprocessor = None,logger=None,train=True):
        NeuralNet.__init__(self,input_shape,preprocessor,logger,train)
    def build(self):
        # x = VGGFace(include_top=False, input_shape=self.input_shape)
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(32, activation='relu', name='fc6')(x)
        x = Dense(32, activation='relu', name='fc7')(x)
        print("VGG")
        x.summary()