import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from paths import dataset_paths

from PIL import Image, ImageOps, ImageDraw
import numpy as np
from numpy import load

import tensorflow as tf
from tensorflow.keras import backend as K

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def resize_and_keep_ratio(path, height, rgb=False):
    """
    Randomly select an image out if given path, resizes it to specific height and keeping the same aspect ratio
    
    Arguments:
        path: List with Paths to the images
        height: int value with desired image height
        rgb: Images should be loaded in grascale or RGB - default is False, means grayscale is default
    Returns:
        Resized image in an PIL Image format
    """
        
    #select random folder
    #folder = random.choice(path)
    folder = path[0]
    #select random images
    images = random.sample(os.listdir(folder), 1)

    if rgb == True:
        image = Image.open(folder + "/" + images[0])
    else:
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
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1):
    """
    Calculates the contrastive loss
    
    Arguments:
        y_true: List of labels
        y_pred: List of predicted labels with same length as y
        margin: Intergervalue, defines the baseline distance for which pairs should be classified as dissimilar
    
    Returns:
        A tensor containing constrastive loss
    """
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean( (1 - y_true) * square_pred + (y_true) * margin_square)
    

def load_arrays(path1, path2):
    """
    Function for load .npz files from disc
    
    Arguments
        path1: Path to file 1 in string format (x data)
        path2: Path to file 2 in string format (y data)
        
    Returns:
        x and y data in numpy array format
    """
    dict_data_x = load(path1)
    dict_data_y = load(path2)
    x = dict_data_x['arr_0']
    y = dict_data_y['arr_0']
    return x, y

def plot_training(H):
    """
    Function for plotting model training
    
    Arguments
        H: training history of the tensorflow model.fit function
        
    Returns:
        Plot of the training loss
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    #plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    #plt.ylabel("Loss")
    plt.legend(loc="lower left")
    
def load_and_split_data(path_x, path_y, split_size, batch_size):
    """
    Function for loading and splitting the. The validation size is fixed to 5% of the whole dataset.
    
    Arguments:
        path_x: Path to x data
        path_y: Path to y data
        split_size: Split size for train split in integer, eg. 80
        
    Returns:
        Returns 3 tf.data.dataset - train, test and val
    """
    
    x, y = load_arrays(path_x, path_y)
    
    #calculate length of the splits
    len_train_data = int(len(x[0]) * (split_size/100))
    len_val_data = int(len(x[0]) * (split_size/100)*0.05)
    len_test_data = int(len(x[0]) - len_train_data - len_val_data)
    
    #index the arrays to split the data
    x_train_0 = x[0][0 :len_train_data]
    x_train_1 = x[1][0 :len_train_data]
    y_train = y[:len_train_data]
    
    x_test_0 = x[0][len_train_data:len_train_data+len_test_data] 
    x_test_1 = x[1][len_train_data:len_train_data+len_test_data]
    y_test = y[len_train_data:len_train_data+len_test_data]
    
    x_val_0 = x[0][len_train_data+len_test_data:]
    x_val_1 = x[1][len_train_data+len_test_data:]
    y_val = y[len_train_data+len_test_data:]
    
    train_dataset = tf.data.Dataset.from_tensor_slices(((x_train_0, x_train_1), y_train)).shuffle(100).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(((x_test_0, x_test_1), y_test)).shuffle(100).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(((x_val_0, x_val_1), y_val)).shuffle(100).batch(batch_size)
    
    return train_dataset, test_dataset, val_dataset

def get_model_predictions(dataset, model):
    """
    Function for predictions by keras model
    
    Arguments:
        dataset: tf.data.dataset format dataset
        model: the trained keras model
    Returns:
        predictions and labels in numpy format
    """
    
    data = dataset.take(1)
    preds = tf.round(model.predict(data))
    preds = preds.numpy()
    for images, labels in data: 
        labels = labels.numpy()

    preds = np.squeeze(preds)
    labels = np.squeeze(labels)
    
    return preds, labels
    

def plot_confusion_matrix(preds, labels):
    """
    Function for plotting an confusion matrix for binary classification
    
    Arguments:
        preds: predictions in numpy format
        labels: labels in numpy format
    Returns:
        Seaborn plot of confusion matrix
    """

    cf_matrix = confusion_matrix(labels, preds)
    
    group_names = ['True Neg','False_Pos','False_Neg','True_Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
def plot_roc_curve(preds, labels):
    """
    Function for plotting an roc curve for binary classification
    
    Arguments:
        preds: predictions in numpy format
        labels: labels in numpy format
    Returns:
        Matplotlib plot of roc curve
    """

    fpr, tpr, _ = roc_curve(labels, preds)
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0,)
    plt.ylim(0,)
    plt.show()
    
def plot_prec_rec_curve(preds, labels):
    precision, recall, _ = precision_recall_curve(labels, preds)
    plt.plot(recall, precision)
    plt.title('PR curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0,)
    plt.ylim(0,)
    plt.show()

    
    
