import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

# Taken from lab3


def montage(images, saveto='montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.
    Also saves the file to the destination specified by `saveto`.
    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    plt.imsave(arr=m, fname=saveto)
    return m

def imcrop_tosquare(img):
    """Make any image a square image.
    Parameters
    ----------
    img : np.ndarray
        Input image to crop, assumed at least 2d.
    Returns
    -------
    crop : np.ndarray
        Cropped image.
    """
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop

# UNTESTED in the utils file
def build_gif(imgs, interval=0.1, dpi=72,
              save_gif=True, saveto='animation.gif',
              show_gif=False, cmap=None):
    """Take an array or list of images and create a GIF.
    Parameters
    ----------
    imgs : np.ndarray or list
        List of images to create a GIF of
    interval : float, optional
        Spacing in seconds between successive images.
    dpi : int, optional
        Dots per inch.
    save_gif : bool, optional
        Whether or not to save the GIF.
    saveto : str, optional
        Filename of GIF to save.
    show_gif : bool, optional
        Whether or not to render the GIF using plt.
    cmap : None, optional
        Optional colormap to apply to the images.
    Returns
    -------
    ani : matplotlib.animation.ArtistAnimation
        The artist animation from matplotlib.  Likely not useful.
    """

    imgs = np.asarray(imgs)
    h, w, *c = imgs[0].shape
    fig, ax = plt.subplots(figsize=(np.round(w / dpi), np.round(h/dpi)))
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    ax.set_axis_off()

    if cmap is not None:
        axs = list(map(lambda x: [ax.imshow(x, cmap=cmap)], imgs))
    else:
        axs = list(map(lambda x: [ax.imshow(x)], imgs))
    ani = animation.ArtistAnimation(fig, axs, interval=interval*1000,
                                    repeat_delay=0, blit=False)
    if save_gif:
        ani.save(saveto, writer='imagemagick', dpi=dpi)
    if show_gif:
        plt.show()
    return ani


def montage_filters(W):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.
    Parameters
    ----------
    W : Tensor
        Input tensor to create montage of.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m

    
####
# Functions to normalise and denormalise images from Dataset:
####


def normalization(img, ds):
    norm_img = (img - ds.mean()) / ds.std()
    return norm_img


def denormalization(norm_img, ds):
    img = norm_img * ds.std() + ds.mean()
    return img


import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.transform import resize
from utils.imgs import *
from utils.celeb import *


def qualify_crop(entity, images_list, data_dir, img_dims):
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    face_images = []
    entity_list = []
    num_face = 0
    num_eyes = 0
    num_qual = 0

    #entity = df.entities[0]
    # images_list = os.listdir(data_dir + '/' + entity)
    #print(entity)
        
    for image in images_list:
        #print(image)
        image_name = data_dir + '/' + entity + '/' + image
        #print(image_name)
        img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) <= 1:
            for (x,y,w,h) in faces:
                    
                roi_gray = gray[y:y+h, x:x+w]
                #roi_color = img_c[y:y+h, x:x+w]
                    
                num_face = num_face + 1

                eyes = eye_cascade.detectMultiScale(roi_gray)
                if (len(eyes) != 2): 
                    continue
                        
                ex1,ey1,ew1,eh1 = eyes[0]
                ex2,ey2,ew2,eh2 = eyes[1]
                if np.abs( ey1 - ey2 ) > eh1 or np.abs( ex1 - ex2 ) < (ew1/2):
                    continue
                num_eyes+=1
                new_img = img_c.copy()

                x,y,w,h = faces[0] 
                border = 0
                    
                new_img = cv2.resize(new_img[y-border:y+h-1+border, x-border:x+w-1+border],
                                         (img_dims, img_dims), interpolation = cv2.INTER_CUBIC)

                #crop_converted = crop_img
                face_images.append( new_img )
                entity_list.append( image )
    print("face", num_face)
    print("eyes", num_eyes)
    return face_images, entity_list
