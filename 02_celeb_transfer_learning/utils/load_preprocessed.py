from skimage.io import imread
import numpy as np
import os
import re

def load_preprocessed_data(corpus_dir, bottles_dir):

    entities = [f for f in os.listdir(corpus_dir) if re.match(r'm\.*', f)]
    entities.sort()
    bottles = [f for f in os.listdir(bottles_dir) if re.match(r'm\.*', f)]
    bottles.sort()
    assert(entities == bottles), ("The Entities dir and Bottles dir do not match")
    del bottles

    labels_accumulated = []
    image_np_list = []
    bottle_np_list = []
    for entity in entities:
        entity_path = os.path.join(corpus_dir, entity)
        images = [f for f in os.listdir(entity_path) if re.search(r'\.jpg$', f)]
        images.sort()
        #print("images:", images)

        # load bottle
        for image in images:
            image_np_list.append(imread(os.path.join(entity_path, image)))
            bottle_path = os.path.join(bottles_dir, entity, image+'.txt')
            bottle_np_list.append(np.loadtxt(bottle_path, delimiter=','))

        # labels
        labels_accumulated += [entity]*len(images)

    Xs = np.array(image_np_list)
    Ys = np.array(labels_accumulated)
    Zs = np.array(bottle_np_list)
    return Xs, Ys, Zs
