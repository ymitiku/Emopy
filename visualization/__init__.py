from keras.models import model_from_json,Model
import cv2
import numpy as np
import os
import json
from visualization.json_helpers import JsonLayer
import pandas as pd
import codecs, json 
from keras import backend as K

def load_model(json_path,h5_path):
    with open(json_path) as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(h5_path)
        return model
def get_feature_images(layer_features):
    if len(layer_features.shape)==2:
        if layer_features.shape[1]<100:
            print layer_features
        output = np.tile(layer_features,(50,1))
        
        output = (output*255).astype(np.uint8)
        if output.shape[1]<100:
            print output.shape
            output = cv2.resize(output,(50,100))
        return output.reshape(1,50,-1)
    elif len(layer_features.shape)==3:
        print "layer is 3D"
    elif len(layer_features.shape)==4:
        clf_shape = layer_features.shape
        if clf_shape[1]> 30:
            output_images = np.zeros((clf_shape[3],clf_shape[1],clf_shape[2]))
        else:
            output_images = np.zeros((clf_shape[3],30,30))
        for i in range(clf_shape[3]):
            current_image = layer_features[:,:,:,i]
            current_image_shape = current_image.shape
            current_image = current_image.reshape(current_image_shape[1],current_image_shape[2])
            current_image = (current_image*255).astype(np.uint8)
            if current_image.shape[0]<30:
                current_image = cv2.resize(current_image,(30,30))
            output_images[i] = current_image
        output_images = output_images.tolist()
        return output_images
    else:
        raise Exception("get feature images not implmented for layer with shape: "+str(layer_features.shape))
def generate_layer_features(input_layer,current_layer,image):
    current_layer_model = Model(inputs=input_layer,outputs=current_layer.output)
    current_layer_features = current_layer_model.predict(image)
    output_images = get_feature_images(current_layer_features)
    return output_images
def replace_non_alphanumeric(string,c):
    output = ""
    for ch in string:
        if (ch >= '0' and ch <= '9') or (ch>='A' and ch<='Z') or (ch>='a' and ch<='z'):
            output+=ch
        else:
            output+=c
    return output
def generate_features(model,image,output_dir):
    input_layer = model.input
    image_shape = image.shape
    image = image.reshape(-1,image_shape[0],image_shape[1],1)
    json_layers = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    for i in range(len(model.layers)):
        layer = model.layers[i]
        layer_name = layer.name 
        layer_features = generate_layer_features(input_layer,layer,image)
        layer_name = replace_non_alphanumeric(layer_name,"_")
        if not os.path.exists(os.path.join(output_dir,layer_name)):
            os.makedirs(os.path.join(output_dir,layer_name))
        if not layer_features is None:
            layer_image_paths = []
            for i in range(len(layer_features)):
                img = layer_features[i]
                img = np.array(img)
                layer_image_paths.append("f"+str(i)+".png")
                cv2.imwrite(os.path.join(output_dir,layer_name,"f"+str(i)+".png"),img)
            json_layers.append(JsonLayer(layer_name,layer_image_paths))




    json_string = json.dumps([ob.__dict__ for ob in json_layers])
    with open("/home/mtk/iCog/projects/visualization/nodejs/emopy/test.json","w+") as test_file:
        test_file.write(json_string)
        

    # json.dump(json_layers, codecs.open("test.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
    # print layer1_features.shape
