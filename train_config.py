from config import IMG_SIZE
BATCH_SIZE = 1 # Batch sized used for traing.
EPOCHS = 100
LEARNING_RATE = 1e-3
PATH2SAVE_MODELS = "models"
DATA_SET_DIR = "/home/mtk/iCog/projects/emopy/dataset/all"
LOG_DIR = "logs"
STEPS_PER_EPOCH = 400   
NETWORK_TYPE = "drnn" # mi for multi input or si for single input
AUGMENTATION = True
INPUT_SHAPE = (IMG_SIZE[0],IMG_SIZE[1],1) 