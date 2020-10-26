"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
from scipy import stats
import string
import itertools


def reduce_dimensions(feature_vectors_full, model):
    """Reduce dimensions using principal components analysis

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    if 'fvectors' in model:
        v = np.array(model['fvectors'])
    else:
        #Principal Components Analysis implemented from lab code
        covx = np.cov(feature_vectors_full, rowvar=0)
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
        v = np.fliplr(v)
        model['fvectors'] = v.tolist()
    pca_train = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)
    return pca_train[:,0:10]


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.
    (including loading in the words for the dictionary)

    Params:
    train_page_names - list of training page names
    """

    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()

    with open('ListOfOneHundredThousandWords.txt') as word_file:
        words_list = [words.replace(" ", "").strip('\n').upper() for words in word_file]
    model_data['words'] = words_list

    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """Classifier uses K-Nearest Neighbour classification to receive better classification
    at higher levels of "noise"

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """

    labels_train = np.array(model['labels_train']) #pass through compiling
    train = np.array(model['fvectors_train']) #this too so they're both fine

    x = np.dot(page, train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance

    """Simple Nearest Neighbour output below"""
    #nearest = np.argmax(dist, axis=1)
    #label = labels_train[nearest]

    k = 15
    # make distance negative to get descending sort of knn
    knn = np.argsort(-dist, axis=1)
    # narrow down for your k nearest
    knn = knn[:,:k]

    # find most occuring value
    label = stats.mode(labels_train[knn],axis=1)[0]

    return np.transpose(label)[0]

def string_difference(str1, str2):
    """This takes in 2 strings, we zip them together so we can check whether each letter matches
    or not and return how many don't, useful for replacements later on if classifications are wrong

    params:
    str1 - first string input
    str2 - second string input"""

    assert len(str1) == len(str2)
    return sum(a != b for a, b in zip(str1,str2))

def remove_punctuation(label):
    """This function will remove the punctuation in the labels so it doesn't cause anomalies/
    partition certain words

    params:
    label - output classification label for each feature vector"""

    word = "".join(c for c in word if c not in ('!','.',':',',','.','?','\''))
    return word

def correct_errors(page, labels, bboxes, model):
    """Error correction is based off taking the labels, removing the unneccessary information,
    making the labels into words, checking if those words are real and exist via comparison with text file,
    then make amends to words that don't exist by way of changing a letter or two. After changes are made to 
    individual words, add the new words in, split the data into labels again, add in unneccessary information back,
    check for percentage correctness 

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage

    In checking the bbox coordinates for each character represented, you can see for some coordinates
    that the difference between the bbox[:2] and the first coordinate of the next bbox is quite large in
    some cases but normally quite small. This is where the whitespace occurs and how we can infer the spacing
    of boxes for words. We can now have a way of showing when the words end. """

    # end_of_word = []

    # take every bbox and check the width between them
    # total_bboxes = len(bboxes)
    # for n in range(total_bboxes-1):
    #     if (space_between(n,n+1,bboxes)):
    #         end_of_word.append(n+1)

    # now we can make words by joining the characters where there isn't whitespace
    # so from the start until the index of the first space we should join together the characters

    # char_to_word = []
    # for x in range(len(end_of_word)):
    #     if (x==0):
    #         char_to_word.append(''.join(labels[x:(end_of_word[x])]).upper())
    #     else:
    #         char_to_word.append(''.join(labels[(end_of_word[x-1]):(end_of_word[x])]).upper())

    # We want to remove all the punctuation as it gets in the way of matching words
    # however we still want to store it's index in the labels for later.
    # punctuation_index = []
    # punctuation_value = []

    # for x in range(len(labels)):
    #     if labels[x] in ('!','.',':',',',';','?','\''):
    #         punctuation_index.append(labels.tolist().index(labels[x]))
    #         punctuation_value.append(labels[x])
    #     labels[x] = remove_punctuation(labels[x])

    # Get the words that don't need changing out of the way (they already exist)
    # no_change_words = set(char_to_word).intersection(model['words'])

    # now we want to take all words that don't appear in both lists of words and change those with a few
    # misclassified chars to words that are similar in the list of actual words.

    # words_to_correct = [w for w in char_to_word if w not in no_change_words]

    # We'll need the index of the words so we can put them back.
    # wtc_index = []
    # for i in range(len(words_to_correct)):
    #     wtc_index.append(char_to_word.index(words_to_correct[i]))

    # Start looking for similar words of the same length for a given word and input the similar word instead.
    # for y in range(len(words_to_correct)):
    #     word = words_to_correct[y]
    #     # filter words of the same length
    #     close_word = filter(lambda x: len(x) == len(word), model['words'])
    #     # replace current word with one real word with only a letters difference
    #     for x, similar in enumerate(close_word):
    #         if (string_difference(similar, word) == 1):
    #             words_to_correct[y] = similar

    # using the stored index, put the words back where they need to be
    # for i in range(len(wtc_index)):
    #     for n, j in enumerate(char_to_word):
    #         if (n == wtc_index[i]):
    #             char_to_word[n] = words_to_correct[i]

    # start splitting the words up again so that they're in the necessary label shape
    # possible_labels = []
    # for i in range(len(char_to_word)):
    #     split_words = [x for x in char_to_word[i].lower()]
    #     possible_labels.append(split_words)
    #     possible_labels1 = list(itertools.chain.from_iterable(possible_labels))

    # add the punctuation back in and you should get the full amount of labels to return and test on
    # for i in range(len(punctuation_index)):
    #     while len(possible_labels1) != len(labels):
    #         possible_labels1.insert(punctuation_index[i],punctuation_value[i])

    return labels


def space_between(first_value, second_value, bboxes):
    """As defined in check errors we will use this to check for whitespace between words so we know where
    they end.
    params:
    value1 - character in first box
    value2 - character in second box
    bboxes - the character box of 4 coordinate representation"""
    bbox1 = bboxes[second_value][0]
    bbox2 = bboxes[first_value][2]
    space_betw = bbox1 - bbox2
    
    if (space_betw > 6):
        return True
    return False