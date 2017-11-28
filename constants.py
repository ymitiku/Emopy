IMG_SIZE = (48,48)
"""
dictionary to represent all seven basic emotions
"""

EMOTIONS = {
    0 : "anger",
    1 : "disgust",
    2 : "fear",
    3 : "happy",
    4 : "sad",
    5 : "surprise",
    6 : "neutral"
}
"""dictionary to represent if emotion is either positive, negetive or neutral.

can be used for networks which classify emotion into pos,neg and neutral.
"""
EMOTION_STATES = {
    'anger'     : 2, # 2 negetive
    'disgust'   : 2,
    'fear'      : 2,
    'happy'     : 1, # 1 positive
    'sad'       : 2,
    'surprise'  : 1,
    'neutral'   : 0, # 0 neutral
}