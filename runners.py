from config import SESSION
from nets.base import NeuralNet
from constants import IMG_SIZE
def run():
    if SESSION == 0:
        run_train()
    elif SESSION == 1:
        run_test()
def run_train():
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)
    neuralNet = NeuralNet(input_shape)
    neuralNet.train()
def run_test():
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)
    neuralNet = NeuralNet(input_shape)
    neuralNet.train()