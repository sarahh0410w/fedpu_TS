import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import pandas as pd
from PIL import Image
from sklearn.utils import class_weight
def get_isic():
    print('getting ISIC...')
    df = pd.read_csv("/home/jmw7289/fedpu/fake/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
    df_test = pd.read_csv("/home/jmw7289/fedpu/fake/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")
    target_to_label = list(df.columns.values)[1:]
    arr = np.array(df[target_to_label])
    targets = list(arr.argmax(axis=1))
    img_names = list(df['image'])

    target_to_label_test = list(df_test.columns.values)[1:]
    arr_test = np.array(df_test[target_to_label_test])
    targets_test = list(arr_test.argmax(axis=1))
    img_names_test = list(df_test['image'])
    #print('test target_name: [0]', targets_test[0])

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    trainIMG_dir = '/home/jmw7289/fedpu/fake/ISIC2018_Task3_Training_Input/'
    for i in range(len(img_names)):
        img_path = os.path.join(trainIMG_dir, img_names[i] + '.jpg')
        image = np.asarray(Image.open(img_path))
        target = targets[i]
        x_train.append(image)
        y_train.append(target)
    
    print('coverting training data...')
    x_train = np.array(x_train)
    x_train = np.moveaxis(x_train,-1,1)
    y_train = np.array(y_train)
    #print('x_train: ', x_train.shape)
    #print(y_train)

    testIMG_dir = '/home/jmw7289/fedpu/fake/ISIC2018_Task3_Test_Input/'
    for i in range(len(img_names_test)):
        img_path = os.path.join(testIMG_dir, img_names_test[i] + '.jpg')
        image = np.asarray(Image.open(img_path))
        target = targets_test[i]
        x_test.append(image)
        y_test.append(target)

    
    x_test = np.array(x_test)
    print('converting test data...')
    x_test = np.moveaxis(x_test,-1,1)
    y_test = np.array(y_test)
    #print('x_test: ', x_test.shape)
    #print(y_test.shape)

    print('checking for nan values...')
    print('- in x train')
    assert not np.any(np.isnan(x_train))
    print('- in y train')
    assert not np.any(np.isnan(y_train))
    print('- in x test')
    assert not np.any(np.isnan(x_test))
    print('- in y test')
    assert not np.any(np.isnan(y_test))
    #print('y test', y_test)

    del df
    del df_test
    del target_to_label
    del target_to_label_test
    del arr_test
    del arr
    del targets
    del targets_test
    del img_names
    del img_names_test

    weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train
                                    )

    print('weights: ', weights)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    get_isic()



