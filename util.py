class Classifier(object):
    def get_num_class(self):
        raise Exception("Not implemented yet")
    def get_string(self,emotion):
        raise Exception("Not implemented yet")
    def get_class(self,string):
        raise Exception("Not implemented yet")

class SevenEmotionsClassifier(Classifier):
    def __init__(self):
        self.EMOTIONS = {
            0 : "anger",
            1 : "disgust",
            2 : "fear",
            3 : "happy",
            4 : "sad",
            5 : "surprise",
            6 : "neutral"
        }
    def get_num_class(self):
        return len(self.EMOTIONS)
    def get_string(self,emotion):
        return self.EMOTIONS[emotion]
    def get_class(self,string):
        for emotion in self.EMOTIONS:
            if self.EMOTIONS[emotion] == string:
                return emotion
        raise Exception("Emotion "+str(string) +" is not recognized")


class PositiveNeutralClassifier(Classifier):
    def __init__(self):
        self.EMOTION_STATE = {
            "happy" : 1,
            "surprise": 1,
            "neutral": 0
        }
    def get_num_class(self):
        return 2
    def get_string(self,emotion):
        assert emotion in [0,1],"Emotion value must be either 0 or 1 for Positive neutral classifier"
        if emotion == 0:
            return "neutral"
        elif emotion == 1:
            return "positive"
    def get_class(self,string):
        assert string in ["happy","neutral","surprise"],"Emotion must be either happy, neutral or surprise for Positive neutral classifier"
        return self.EMOTION_STATE[string]
    
class PositiveNegetiveClassifier(Classifier):
    def __init__(self):
        self.EMOTION_STATE = {
                "happy" : 1,
                "surprise": 1,
                "neutral": 0,
                "anger": 2,
                "disgust":2,
                "sad": 2,
                "fear":2
            }
    def get_num_class(self):
        return 3
    def get_string(self,emotion):
        assert emotion in [0,1,2],"Emotion value must be either 0,1 or 2 for Positive negative classifier"
        if emotion == 0:
            return "neutral"
        elif emotion == 1:
            return "positive"
        elif emotion == 2:
            return "negative"
    def get_class(self,string):
        assert string in ["happy","neutral","surprise","anger","disgust","fear","sad"],"Emotion must be either happy, neutral,sad, fear,disgust,sad or surprise for Positive negative classifier"
        return self.EMOTION_STATE[string]
    