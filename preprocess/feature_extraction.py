import numpy as np
import dlib 
from config import IMG_SIZE


class FeatureExtractor(object):
    """Base class for Feature extactors
    """
    
    def __init__(self):
        pass
    def extract(self,images):
        raise NotImplementedError('abstract method extract should be implemented!')

class ImageFeatureExtractor(FeatureExtractor):
    def __init__(self,**kwargs):
        FeatureExtractor.__init__(self,**kwargs)

    def extract(self,images):
        new_images = np.array(images).astype(float)/255
        return new_images

class DlibFeatureExtractor(FeatureExtractor):
    def __init__(self,predictor, **kwargs):
        FeatureExtractor.__init__(self,**kwargs)
        self.predictor = predictor
    def get_dlib_points(self,image):
        face = dlib.rectangle(0,0,image.shape[1]-1,image.shape[0]-1)
        img = image.reshape(IMG_SIZE[0],IMG_SIZE[1])
        shapes = self.predictor(img,face)
        parts = shapes.parts()
        output = np.zeros((68,2))
        for i,point in enumerate(parts):
            output[i]=[point.x,point.y]
        output = np.array(output).reshape((1,68,2))
        return output
    def to_dlib_points(self,images):
        output = np.zeros((len(images),1,68,2))
        centroids = np.zeros((len(images),2))
        for i in range(len(images)):
            dlib_points = self.get_dlib_points(images[i])[0]
            centroid = np.mean(dlib_points,axis=0)
            centroids[i] = centroid
            output[i][0] = dlib_points
        return output,centroids
            
    def get_distances_angles(self,all_dlib_points,centroids):
        all_distances = np.zeros((len(all_dlib_points),1,68,1))
        all_angles = np.zeros((len(all_dlib_points),1,68,1))
        for i in range(len(all_dlib_points)):
            dists = np.linalg.norm(centroids[i]-all_dlib_points[i][0],axis=1)
            angles = self.get_angles(all_dlib_points[i][0],centroids[i])
            all_distances[i][0] = dists.reshape(1,68,1)
            all_angles[i][0] = angles.reshape(1,68,1)
        return all_distances,all_angles
    def angle_between(self,p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return (ang1 - ang2) % (2 * np.pi)
    def get_angles(self,dlib_points,centroid):
        output = np.zeros((68))
        for i in range(68):
            angle = self.angle_between(dlib_points[i],centroid)
            output[i] = angle
        return output

    def extract(self,images):


        dlib_points,centroids = self.to_dlib_points(images)

        distances,angles = self.get_distances_angles(dlib_points,centroids)

        IMAGE_CENTER = np.array(IMG_SIZE)/2
        IMG_WIDTH = IMG_SIZE[1]
        # normalize
        dlib_points = (dlib_points - IMAGE_CENTER)/IMG_WIDTH
        dlib_points = dlib_points.reshape((-1,1,68,2))
        

        distances /= 50.0;
        distances = distances.reshape(-1,1,68,1)
        

        angles /= (2 * np.pi)
        angles = angles.reshape(-1,1,68,1)
        images = images.astype(np.float32)/255

        return images,dlib_points,distances,angles
