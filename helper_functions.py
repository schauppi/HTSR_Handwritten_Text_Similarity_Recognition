import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from paths import dataset_paths

from PIL import Image, ImageOps, ImageDraw
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K


def resize_and_keep_ratio(path, height):
    """
    Randomly select an image out if given path, resizes it to specific height and keeping the same aspect ratio
    
    Arguments:
        path: List with Paths to the images
        height: int value with desired image height
        
    Returns:
        Resized image in an PIL Image format
    """
        
    #select random folder
    #folder = random.choice(path)
    folder = path[0]
    #select random images
    images = random.sample(os.listdir(folder), 1)

    image = Image.open(folder + "/" + images[0]).convert("L")
    if image.size[0] > 500:
        #Get a copy of the image for plotting
        image_before = image

        #Resize the image to specific height and keeping the same aspect ratio
        height_precent = (height / float(image.size[1]))
        width = int((float(image.size[0]) * float(height_precent)))
        image = image.resize((width, height), Image.NEAREST)

        return(image)
    else:
        return image == 0
    
def crop_image(image, crop_width):
    """
    Crops an random part out of an image and display the cropped area
    
    Arguments:
        image: image in an numpy array format
        width: int value width desired crop width
        
    Return:
        Cropped image in an numpy array format
    """
    #Convert PIL Image object into numpy array
    img_array = np.array(image)
    #select random point on x-axis and crop the image
    x_max = img_array.shape[1] - crop_width
    x = np.random.randint(0, x_max)
    crop = img_array[0:crop_width, x: x + crop_width]

    return crop

def data_generator(batch_size=256):
    """
    Data Generator for training process in the .fit function. Yields x and y pair with batch size every epoch
    
    Arguments:
        batch_size: desired batch size in int format
        
    Returns:
        Batch of siamese image pairs x with their label y
    """
    height, width = 224, 224
    path = dataset_paths
    while True:
        x, y = create_batch(batch_size, height, width, path)
        yield x, y

def create_list(folder):
    """
    Creates an list with an item inside
    
    Arguments:
        folder: Desired item to get into list
        
    Returns:
        List with the folder object inside
    """
    folder_list = []
    folder_list.append(folder)
    return folder_list

def create_batch(batch_size, height, width, path):
    """
    Function to create siamese pairs. Randomly takes genuine and opposite pairs from data directory, scales it, croppe out an image piece and normalizes it.
    
    Arguments:
        batch_size: desired batch size in int format
        height: desired height of the cropped images
        width: desired width of the cropped images
        path: path to the data directories
        
    Returns:
        Three shuffled arrays in format ((x_1, x_2), y). x_1 and x_2 are the cropped images and y is the labels array
    """
    #batch size / 2 bc. we have two pairs per iteration
    batch_size = int(batch_size / 2)
    
    #create empty arrays
    x_genuine = np.zeros([batch_size, 2, 1, height, width])
    y_genuine = np.zeros([batch_size, 1])
    x_opposite = np.zeros([batch_size, 2, 1, height, width])
    y_opposite = np.zeros([batch_size, 1])
    
    i = 0
    
    while i < batch_size:
        
        #Select random folders for genuine and opposite pairs and save the paths to lists
        list_genuine = create_list(random.choice(dataset_paths))
        folder_opposite_1, folder_opposite_2 = random.sample(dataset_paths, 2)
        list_opposite_1 = create_list(folder_opposite_1)
        list_opposite_2 = create_list(folder_opposite_2)
        
        #load the genuine and opposite images from disk and preprocess it
        image_1_genuine = resize_and_keep_ratio(list_genuine, height)
        image_2_genuine = resize_and_keep_ratio(list_genuine, height)
        image_1_opposite = resize_and_keep_ratio(list_opposite_1, height)
        image_2_opposite = resize_and_keep_ratio(list_opposite_2, height)
        
        #new cycle if one of the 4 loaded images is 0
        if image_1_genuine == 0 or image_2_genuine == 0 or image_1_opposite == 0 or image_2_opposite == 0:
            pass
        else:
            #crop and standardize the loaded images and save it in the arrays
            x_genuine[i, 0, 0, :, :] = (crop_image(image_1_genuine, width)) / 255.
            x_genuine[i, 1, 0, :, :] = (crop_image(image_2_genuine, width)) / 255.
            y_genuine[i] = 1
            
            x_opposite[i, 0, 0, :, :] = (crop_image(image_1_opposite, width)) / 255.
            x_opposite[i, 1, 0, :, :] = (crop_image(image_2_opposite, width)) / 255.
            y_opposite[i] = 0
        
            i += 1
    #concatenate the arrays
    x = np.concatenate([x_genuine, x_opposite], axis=0)
    y = np.concatenate([y_genuine, y_opposite], axis=0)
    #change the shape of array
    x = np.einsum("abcde->abdec", x)
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    
    randomize = np.arange(len(x_1))
    np.random.shuffle(randomize)
    y = y[randomize]
    x_1 = x_1[randomize]
    x_2 = x_2[randomize]

    return [x_1, x_2], y 

def euclidean_distance(vectors):
    """
    Calculates the euclidian distance between two vectors
    
    Arguments:
        vectors: List containing two tensors of same length
    
    Return:
        Tensor containing euclidian distance between vectors
    """
    (featsA, featsB) = vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
    """
    Calculates the contrastive loss
    
    Arguments:
        y: List of labels
        preds: List of predicted labels with same length as y
        margin: Intergervalue, defines the baseline distance for which pairs should be classified as dissimilar
    
    Returns:
        A tensor containing constrastive loss
    """
    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss