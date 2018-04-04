from nets.base import NeuralNet
from keras.layers import Flatten,Dense,Input,Dropout
from keras_vggface.vggface import VGGFace
from keras.models import Model


class VGGFaceEmopyNet(NeuralNet):
    def __init__(self,input_shape,preprocessor = None,logger=None,train=True,model_architecture="vgg16"):
        self.model_architecture = model_architecture
        NeuralNet.__init__(self,input_shape,preprocessor,logger,train)
   
    def build(self):
        # x = VGGFace(include_top=False, input_shape=self.input_shape)
        vgg_model = VGGFace(include_top=False, input_shape=(48, 48, 3),model=self.model_architecture)
        # for layer in vgg_model.layers:
        #     layer.trainable = False
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(128, activation='relu', name='fc6')(x)
        x = Dense(252, activation='relu', name='fc7')(x)
        x = Dense(self.preprocessor.classifier.get_num_class(),activation="softmax",name="output")(x)
        self.model = Model(vgg_model.input,x)

        self.built = True
        print self.model.summary()
        return self.model