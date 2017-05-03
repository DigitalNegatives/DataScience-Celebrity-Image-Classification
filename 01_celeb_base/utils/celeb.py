import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.transform import resize
from utils.imgs import *

def get_dataframe(path, num_entities=4, num_samples=25):

    entities = os.listdir(path)
    assert(num_entities <= len(entities))

    celebs = np.random.choice(entities, num_entities, replace=False)

    imgs = []

    for name in celebs:

        samples = os.listdir(path + name)

        assert(num_samples <= len(samples))

        images = np.random.choice(samples, num_samples, replace=False)

        samples_list = [name]*len(images)
        pair = list(zip(samples_list, images))
        imgs = imgs + pair

    df = pd.DataFrame(imgs, columns=['entities', 'images'])

    return df

def get_all_images(df, path):

    all_images = []
    all_labels = []

    for entity in list(set(df.entities)):

        Xs, Ys = get_images(df, entity, path)

        all_images = all_images + list(Xs)
        all_labels = all_labels + Ys

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    return all_images, all_labels

def get_images(df, entity, path):

    this_entity = []
    images_list = df[df.entities == entity].images

    for image in images_list:

        img = imread(path + entity + '/' + image)

        squared_img = imcrop_tosquare(img)
        resized_img = resize(squared_img, (100, 100))

        this_entity.append(resized_img)

    Xs = np.array(this_entity)
    Ys = [entity] * len(images_list)

    montage(Xs, saveto='./'+entity+'.png')

    return Xs, Ys
