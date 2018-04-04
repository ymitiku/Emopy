from config import IMG_SIZE
BATCH_SIZE = 1 # Batch sized used for traing.
EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_PATH = "models/model"
DATA_SET_DIR = "/home/mtk/iCog/projects/emopy/dataset/all"
LOG_DIR = "logs"
STEPS_PER_EPOCH = 1000   
NETWORK_TYPE = "face" 
AUGMENTATION = True
INPUT_SHAPE = (IMG_SIZE[0],IMG_SIZE[1],1) 
