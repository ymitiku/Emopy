import argparse
from config import EMOTION_CLASSIFICATION

def get_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n","--net",default="face",type=str)
    parser.add_argument("-c","--emotions",default=EMOTION_CLASSIFICATION,type=str)
    parser.add_argument("-m","--model_name",default="models/model",type=str)
    parser.add_argument("-x","--sequence_length",default=71,type=int)

    args = parser.parse_args()
    return args