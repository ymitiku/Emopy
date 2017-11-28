import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        pass
        
    def extract(self,images):
        raise NotImplementedError('abstract method extract should be implemented!')

class ImageFeatureExtractor(FeatureExtractor):
    def __init__(self,**kwargs):
        FeatureExtractor.__init__(self,**kwargs)

    def extract(self,images):
        images = np.array(images).astype(float)/255
        return images

class DlibFeatureExtractor(FeatureExtractor):
    def __init__(self, detector, predictor, **kwargs):
        FeatureExtractor.__init__(self,**kwargs)
        self.detector = detector
        self.predictor = predictor


    def extract(self,images):
        raise NotImplementedError("Not implmented yet")
