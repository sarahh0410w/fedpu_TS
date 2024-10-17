import os
import numpy as np
import pandas as pd
import pathlib
import imageio
import random
import cv2
from sklearn.utils import class_weight
# import seaborn as sns
import gc
import torchvision.transforms as T
import torch
from sklearn.utils import class_weight
from PIL import Image

def get_chest():
# Exploring dataset 
    base_dir = '/home/jmw7289/fedpu/fake/chest_xray/'

    train_pneumonia_dir = base_dir+'train/PNEUMONIA/'
    train_normal_dir=base_dir+'train/NORMAL/'

    test_pneumonia_dir = base_dir+'test/PNEUMONIA/'
    test_normal_dir = base_dir+'test/NORMAL/'

    val_normal_dir= base_dir+'val/NORMAL/'
    val_pnrumonia_dir= base_dir+'val/PNEUMONIA/'

    train_pn = [train_pneumonia_dir+"{}".format(i) for i in os.listdir(train_pneumonia_dir) ]
    train_normal = [train_normal_dir+"{}".format(i) for i in os.listdir(train_normal_dir) ]

    test_normal = [test_normal_dir+"{}".format(i) for i in os.listdir(test_normal_dir)]
    test_pn = [test_pneumonia_dir+"{}".format(i) for i in os.listdir(test_pneumonia_dir)]

    val_pn= [val_pnrumonia_dir+"{}".format(i) for i in os.listdir(val_pnrumonia_dir) ]
    val_normal= [val_normal_dir+"{}".format(i) for i in os.listdir(val_normal_dir) ]

    print ("Total images:",len(train_pn+train_normal+test_normal+test_pn+val_pn+val_normal))
    print ("Total pneumonia images:",len(train_pn+test_pn+val_pn))
    print ("Total Nomral images:",len(train_normal+test_normal+val_normal))

    # Gathering all pneumina and normal chest X-ray in two python list
    pn = train_pn + test_pn + val_pn
    normal = train_normal + test_normal + val_normal

# Spliting dataset in train set,test set and validation set.

    train_imgs = pn[:3418]+ normal[:1224]  # 80% of 4273 Pneumonia and normal chest X-ray are 3418 and 1224 respectively.
    test_imgs = pn[3418:4059]+ normal[1224:1502]
    val_imgs = pn[4059:] + normal[1502:]

    print("Total Train Images %s containing %s pneumonia and %s normal images" 
      % (len(train_imgs),len(pn[:3418]),len(normal[:1224])))
    print("Total Test Images %s containing %s pneumonia and %s normal images"
      % (len(test_imgs),len(pn[3418:4059]),len(normal[1224:1502])))
    print("Total validation Images %s containing %s pneumonia and %s normal images" 
      % (len(val_imgs),len(pn[4059:]),len(normal[1502:])))

    random.shuffle(train_imgs)
    random.shuffle(test_imgs)
    random.shuffle(val_imgs)
    
    #Loading each image and their label into array

    X, y = preprocess_image(train_imgs)

    # get the labels for test set

    P, t = preprocess_image(test_imgs)

    # get the labels for validation set

    K, m = preprocess_image(val_imgs)

    #class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
   # class_weights = dict(enumerate(class_weights))
   # print(class_weights)

    train_imgs = train_pn[:3875]+ train_normal[:1341]
    del train_imgs
    gc.collect()

    X_train = np.array(X)
    X_train = np.moveaxis(X_train,-1,1)
    y_train = np.array(y)
    X_test = np.array(P)
    X_test = np.moveaxis(X_test,-1,1)
    y_test = np.array(t)
    X_val = np.array(K)
    X_val = np.moveaxis(X_val,-1,1)
    y_val = np.array(m)

    print('x_train: ',X_train.shape) #x_train:  (4642, 224, 224, 3)
    print(y_train.shape) #(4642,)
    print(X_test.shape) #(919,224,224,3)
    print(y_test.shape) #(919,)
    print(X_val.shape)  #(295,224,224,3)
    print(y_val.shape)

    #data transforms
    #preprocess = T.Compose([T.ToPILImage(),
     #   T.RandomCrop(224),
      #  T.RandomHorizontalFlip(),
      #  T.ToTensor(),
      #  T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

   # Xtrain_tfm = preprocess(X_train)
   # Xtest_tfm = preprocess(X_test)
   # Xval_tfm = preprocess(X_val)

   # print('X_train transform: ', Xtrain_tfm.shape)
    print('checking for nan...')
    print('1')
    assert not np.any(np.isnan(X_train))
    print('2')
    assert not np.any(np.isnan(y_train))
    print('3')
    assert not np.any(np.isnan(X_test))
    print('4')
    assert not np.any(np.isnan(y_test))

    print('calculating weights...')
    weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train
                                    )

    print('weights: ',weights)


    return X_train, y_train, X_test, y_test



def preprocess_image(image_list):
    img_size = 224

    X = [] # images
    print('image list: ', len(image_list))
    y = [] #labels (0 for Normal or 1 for Pneumonia)
    count=0

    for image in image_list:

        try:

            img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
           # print('img: ', img)

            img=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_CUBIC)

            #convert image to 2D to 3D
            img = np.dstack([img, img, img])

            #convrt greyscale image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalalize Image
            img = img.astype(np.float32)/255.

            count=count+1

            X.append(img)

        except:
            continue
        #get the labels
        if 'NORMAL' in image:
            y.append(0)

        elif 'IM' in image:
            y.append(0)

        elif 'virus' or 'bacteria' in image:
            y.append(1)


    return X, y


if __name__ == '__main__':

    x1, y1, x2,y2 = get_chest()
