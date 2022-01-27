import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from paths import dataset_paths
from paths import path_matrix

from PIL import Image, ImageOps, ImageDraw
import numpy as np
from numpy import load

import tensorflow as tf
from tensorflow.keras import backend as K

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.manifold import TSNE

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    Crops an random part out of an image
    
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

def crop_image_triplet_loss(image, crop_width):
    """
    Crops two random parts out of an image
    
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
    x_1 = np.random.randint(0, x_max)
    x_2 = np.random.randint(0, x_max)
    crop_1 = img_array[0:crop_width, x_1: x_1 + crop_width]
    crop_2 = img_array[0:crop_width, x_2: x_2 + crop_width]

    return crop_1, crop_2

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
    
def triplet_loss_l2(alpha, emb_size):
    """
    Calculates the triplet loss with eucledian distance function
    
    Arguments:
        alpha: bias margin for distance calculation
        embedding: embedded output from siamese model
        
    Returns:
        A tensor containing triplet loss
    """
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
        distance1 = tf.sqrt(tf.reduce_sum(tf.pow(anc - pos, 2), 1, keepdims=True))
        distance2 = tf.sqrt(tf.reduce_sum(tf.pow(anc - neg, 2), 1, keepdims=True))
        return tf.reduce_mean(tf.maximum(distance1 - distance2 + alpha, 0.))
    return loss

def triplet_loss_cosine(alpha, emb_size):
    """
    Calculates the triplet loss with cosine distance function
    
    Arguments:
        alpha: bias margin for distance calculation
        embedding: embedded output from siamese model
        
    Returns:
        A tensor containing triplet loss
    """
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
        distance1 = tf.keras.losses.cosine_similarity(anc, pos)
        distance2 = tf.keras.losses.cosine_similarity(anc, neg)
        return tf.keras.backend.clip(distance1 - distance2 + alpha, 0., None)
    return loss

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
    
def plot_triplet_training(H):
    """
    Function for plotting model training with triplet loss
    
    Arguments
        H: training history of the tensorflow model.fit function
        
    Returns:
        Plot of the training loss
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    
def load_and_split_data(path_x, path_y, split_size, batch_size, triplet=False):
    """
    Function for loading and splitting the dataset. The validation size is fixed to 5% of the whole dataset.
    
    Arguments:
        path_x: Path to x data
        path_y: Path to y data
        split_size: Split size for train split in integer, eg. 80
        
    Returns:
        Returns 3 tf.data.dataset - train, test and val
    """
    
    x, y = load_arrays(path_x, path_y)
    
    if triplet == True:
        #calculate length of the splits
        len_train_data = int(len(x[0]) * (split_size/100))
        len_val_data = int(len(x[0]) * (split_size/100)*0.05)
        len_test_data = int(len(x[0]) - len_train_data - len_val_data)

        #index the arrays to split the data
        x_train_0 = x[0][0 :len_train_data]
        x_train_1 = x[1][0 :len_train_data]
        x_train_2 = x[2][0 :len_train_data]
        y_train = y[:len_train_data]

        x_test_0 = x[0][len_train_data:len_train_data+len_test_data] 
        x_test_1 = x[1][len_train_data:len_train_data+len_test_data]
        x_test_2 = x[2][len_train_data:len_train_data+len_test_data]
        y_test = y[len_train_data:len_train_data+len_test_data]

        x_val_0 = x[0][len_train_data+len_test_data:]
        x_val_1 = x[1][len_train_data+len_test_data:]
        x_val_2 = x[2][len_train_data+len_test_data:]
        y_val = y[len_train_data+len_test_data:]
        
        train_dataset = tf.data.Dataset.from_tensor_slices(((x_train_0, x_train_1, x_train_2), y_train)).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices(((x_test_0, x_test_1, x_test_2), y_test)).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices(((x_val_0, x_val_1, x_val_2), y_val)).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    else:
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
    
        train_dataset = tf.data.Dataset.from_tensor_slices(((x_train_0, x_train_1), y_train)).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices(((x_test_0, x_test_1), y_test)).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices(((x_val_0, x_val_1), y_val)).shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
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

    
def recall_f(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_f(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_f(y_true, y_pred)
    recall = recall_f(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def evaluate_preds(model, data):
    loss, accuracy, f1_score, precision, recall = model.evaluate(data)
    return {"loss": loss,
           "accuracy": accuracy,
           "f1-score": f1_score,
           "precision": precision,
           "recall": recall}

def plot_triplet_roc_curve(model, dataset, model_name, emb_size):
    """
    Function for plotting the roc curve of siamese model trained on triplet loss
    
    Arguments:
        model: trained keras classifier
        dataset: evaluation dataset
        model_name: model name in string format
        emb_size: embedding size of the model output vectors
        
    Returns:
        Matplotlib plot of the roc curve
    """
    #get model predictions
    results = model.predict(dataset)
    y_score = []
    y_true = []
    #iterate over all triplets and get distances between anchor/positive, anchor/negative
    #append labels to list y_true - 1 if anchor/positive, else 0
    for i in range(len(results)):
        anchor = results[i][:emb_size]
        positive = results[i][emb_size:emb_size*2]
        negative = results[i][emb_size*2:]
        distance1 = tf.sqrt(tf.reduce_sum(tf.pow(anchor - positive, 2), axis=-1)).numpy()
        y_true.append(1)
        y_score.append((distance1 * -1))
        distance2 = tf.sqrt(tf.reduce_sum(tf.pow(anchor - negative, 2), axis=-1)).numpy()
        y_true.append(0)
        y_score.append((distance2 * -1))
    
    #get list in numpy array format
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    
    #predict fpr, tpr and auc score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    
    #calculate best threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)

    #plot the roc curve
    plt.plot(fpr, tpr, label=f"{model_name} ,auc: {auc_score}")
    plt.plot([0,1], [0,1], linestyle="--", label=f"No Skill")
    plt.scatter(fpr[ix], tpr[ix], marker="o", color="black", label="Best")
    plt.title(f'ROC curve, \nBest threshold: {thresholds[ix]}')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0,)
    plt.ylim(0,)
    plt.legend()
    plt.show()
    

def plot_embeddings(height, width, path_matrix, model, emb_size):
    """
    Function for plotting triplet image embedding - 42 plots - 1 scribe against all other scribes in the dataset
    
    Arguments:
        height: height of the image - on which is the model trained on
        width: width of the image - on which is the model trained on
        path_matrix: list of the data directories - scribe[0], all other scribes[1]
        model: trained keras model on triplet loss
    
    Returns:
        42 Matplotlib scatter subplots containing the embedded images
    """
    tsne = TSNE(n_components=2, n_iter=15000, metric="cosine")
    
    #Reference image
    img_1 = resize_and_keep_ratio(path_matrix[0][0], height, rgb=True)
    crop_1, crop_2 = crop_image_triplet_loss(img_1, width)
    crop_1, crop_2 = crop_1 / 255. , crop_2 / 255.
    img_1 = np.expand_dims(crop_1, 0)
    img_2 = np.expand_dims(crop_2, 0)

    #all other image
    img_list = []
    for i in range(len(path_matrix[1])):
        i = 0
        #check if image not 0 
        while i < 1:
            img = resize_and_keep_ratio(path_matrix[1][i], height, rgb=True)
            if img == 0:
                pass
            else:
                i = 1
        crop = crop_image(img, width) / 255.
        img = np.expand_dims(crop, 0)
        img_list.append(img)
        
    embeddings = []

    for i in range(len(img_list)):
        cache_embedding_list = []
        results = model.predict([img_1, img_2, img_list[i]])
        emb_size = emb_size
        anchor = results[0][:emb_size]
        positive = results[0][emb_size:emb_size*2]
        negative = results[0][emb_size*2:]
        anchor_emb = tsne.fit_transform(anchor.reshape(-1,1))
        positive_emb = tsne.fit_transform(positive.reshape(-1,1))
        negative_emb = tsne.fit_transform(negative.reshape(-1,1))
        cache_embedding_list.append(anchor_emb)
        cache_embedding_list.append(positive_emb)
        cache_embedding_list.append(negative_emb)
        embeddings.append(cache_embedding_list)
        
    plt.rcParams["figure.figsize"] = (15,10)
    fig, ax = plt.subplots(nrows=6, ncols=7)
    count = 0
    for row in ax:
        for col in row:
            col.scatter(embeddings[count][0][:,0], embeddings[count][0][:,1], color="red")
            col.scatter(embeddings[count][1][:,0], embeddings[count][1][:,1], color="green")
            col.scatter(embeddings[count][2][:,0], embeddings[count][2][:,1], color="blue")
            #plt.title("Red: Anchor, Green: Positives, Blue: Negatives")
            count += 1

    fig.tight_layout()
    plt.show()
    
    
    
def plot_single_embedding(height, width, dataset_paths, model, emb_size):
    """
    Function for plotting image embedding from two scribes
    
    
    Arguments:
        height: height of the image - on which is the model trained on
        width: width of the image - on which is the model trained on
        path_matrix: list of the data directories - scribe[0], all other scribes[1]
        model: trained keras model on triplet loss
    
    Returns:
        Scatterplots containing embedded images
    """
    
    tsne = TSNE(n_components=2, n_iter=15000, metric="cosine")
    
    #select randomly two different scribes
    folder_1, folder_2 = random.sample(dataset_paths, 2)
    
    folder_list_1 = []
    folder_list_1.append(folder_1)
    folder_list_2 = []
    folder_list_2.append(folder_2)
    
    img_1 = resize_and_keep_ratio(folder_list_1, height, rgb=True)
    img_2 = resize_and_keep_ratio(folder_list_1, height, rgb=True)
    img_3 = resize_and_keep_ratio(folder_list_2, height, rgb=True)
    
    img_1 = crop_image(img_1, width) / 255.
    img_2 = crop_image(img_2, width) / 255.
    img_3 = crop_image(img_3, width) / 255.
    
    img_1 = np.expand_dims(img_1, 0)
    img_2 = np.expand_dims(img_2, 0)
    img_3 = np.expand_dims(img_3, 0)
    
    results = model.predict([img_1, img_2, img_3])
    
    anchor = results[0][:emb_size]
    positive = results[0][emb_size:emb_size*2]
    negative = results[0][emb_size*2:]
    
    anchor_emb = tsne.fit_transform(anchor.reshape(-1,1))
    positive_emb = tsne.fit_transform(positive.reshape(-1,1))
    negative_emb = tsne.fit_transform(negative.reshape(-1,1))
    
    plt.scatter(anchor_emb[:,0], anchor_emb[:,1], color="red")
    plt.scatter(positive_emb[:,0], positive_emb[:,1], color="green")
    plt.scatter(negative_emb[:,0], negative_emb[:,1], color="blue")
    plt.title("Red: Anchor, Green: Positives, Blue: Negatives")
    
    
def create_tf_data_datasets(anchor_images_path, positive_images_path, height, width, batch_size):
    """
    Function for creating tf data datasets input pipeline for triplet loss. The target shape must be changed in the "preprocess_image()" function!
    
    Arguments:
        anchor_images_path: path to anchor dir in string_format
        positive_images_path: path to positive dir in string_format
        height: target height of image
        width: target width of image
        batch_size: desired batch size for training
        
    Returns:
        train and val and test datasets
    """
    
    #list and sort the data in folders
    anchor_images = sorted([str(anchor_images_path + "/" + f) for f in os.listdir(anchor_images_path)])
    positive_images = sorted([str(positive_images_path + "/" + f) for f in os.listdir(positive_images_path)])
    image_count = len(anchor_images)
    
    #create anchor and positives dataset
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
    
    #randomize list for generating corresponding negative images
    rng = np.random.RandomState(seed=42)
    rng.shuffle(anchor_images)
    rng.shuffle(positive_images)
    negative_images = anchor_images + positive_images
    np.random.RandomState(seed=32).shuffle(negative_images)
    
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
    negative_dataset = negative_dataset.shuffle(buffer_size=4096)
    
    #concaneta the datasets together and preprocess it
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)
    
    #split the dataset
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))
    val_dataset = val_dataset.take(round(image_count * 0.75))
    test_dataset = val_dataset.skip(round(image_count * 0.75))
    
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def preprocess_image(filename, target_shape=(224,224)):
    """
    Function for loading the images, preprocess is and reshape it to target shape
    
    Argumemnts:
        filename: filepath in string_format
    
    Returns:
        Preprocessed image
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Function for loading and preprocessing triplet pairs
    
    Arguments:
        anchor: path to anchor image
        positive: path to positive image
        anchor: path to negative image
        
    Returns:
        Preprocessed image triplets
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

def visualize_triplets_tf_data_dataset(anchor, positive, negative):
    """
    Function for visualizing image triplets from batches
    
    Arguments:
        anchor: negative image from tf data dataset
        positive: positive image from td data dataset
        negative: negative image from tf data dataset 
        
    Returns:
        3x3 Matplotlib subplots
    """

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])