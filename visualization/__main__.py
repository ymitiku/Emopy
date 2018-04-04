from visualization import load_model,generate_features
import cv2
import numpy as np

model = load_model("models/nn/nn-14.json","models/nn/nn-14.h5")
# model = load_model("models/vgg/vgg-0.json","models/vgg/vgg-0.h5")
# model = load_model("/home/mtk/iCog/projects/emopy-models/all-neut-pos.json","/home/mtk/iCog/projects/emopy-models/all-neut-pos.h5");

# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/test/happy/S026_006_00000012.png")

# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/train/anger/S011_004_00000018.png")
# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/test/disgust/S011_005_00000019.png")
# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/test/fear/S011_003_00000013.png")
# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/test/happy/S011_006_00000012.png")
# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/test/sad/S011_002_00000019.png")
# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/test/surprise/S011_001_00000013.png")
# img = cv2.imread("/home/mtk/iCog/projects/emopy/dataset/ck/test/neutral/S011_004_00000001.png")



# print help(output.eval)
# model.summary()

emg_emotions = {"/home/mtk/iCog/projects/emopy/dataset/ck/train/anger/S011_004_00000018.png":"anger",
                "/home/mtk/iCog/projects/emopy/dataset/ck/test/disgust/S011_005_00000019.png":"disgust",
                "/home/mtk/iCog/projects/emopy/dataset/ck/test/fear/S011_003_00000013.png":"fear",
                "/home/mtk/iCog/projects/emopy/dataset/ck/test/happy/S011_006_00000012.png":"happy",
                "/home/mtk/iCog/projects/emopy/dataset/ck/test/sad/S011_002_00000019.png":"sad",
                "/home/mtk/iCog/projects/emopy/dataset/ck/test/surprise/S011_001_00000013.png":"surprise",
                "/home/mtk/iCog/projects/emopy/dataset/ck/test/neutral/S011_004_00000001.png":"neutral"

                }
for img_file in emg_emotions:
    print img_file
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(48,48))
    img = img.astype(np.float32)/255

    generate_features(model,img,"/home/mtk/iCog/projects/visualization/nodejs/emopy/"+emg_emotions[img_file])