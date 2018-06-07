import os
import random
import math
import numpy as np
from scipy.misc import imread, imresize

# Random seed for splitting data consistently
SEED = 231

# H x W that the images should be reshaped to
IMG_SIZE = (128, 128)
BAM_MEAN = 131.5598075494837
BAM_STD = 81.77253959615437

# Directory where the media/emotion subdirectories are located
DATA_DIR = "data/"
# SQLite file needs to go at "data/bam.sqlite"
BAM = "bam.sqlite"
# Names for the saved numpy arrays that include the reshaped and formated data
INPUT_FILE = "input.npz"
MEDIA_LABEL_FILE = "media_label.npz"
EMOTION_LABEL_FILE = "emotion_label.npz"

# List of media label names
MEDIA_LABELS = [
        "media_3d_graphics",
        "media_comic",
        "media_graphite",
        "media_oilpaint",
        "media_pen_ink",
        "media_vectorart",
        "media_watercolor"]
# List of emotion label names
EMOTION_LABELS = [
        "emotion_gloomy",
        "emotion_happy",
        "emotion_peaceful",
        "emotion_scary"]


def split_data(dev_test_ratio=.1):
    """
    Splits the directory struction in the DATA_DIR to have three folders
    (train, dev, test). Each folder contains the same subdirectories as are
    created using get_data.sh. Data is split so that the dev and test datasets
    are DEV_TEST_RATIO of the original dataset.

    Input:
        dev_test_ratio - proportion of the full dataset that should be used
                         for the dev set and for the test set
    """
    # Prevent splitting when data is already split by checking if
    # DATA_DIR/train exists.
    print("Checking if data needs to be split")
    if os.access(os.path.join(DATA_DIR, "train"), os.F_OK):
        return

    # Move all files to train/dev/test randomly using a set seed
    print("Splitting data into train/dev/test sets")
    seed = SEED
    random.seed(seed)

    for media_dir in os.scandir(DATA_DIR):
        if not media_dir.is_dir():
            continue
        for path, _, filenames in os.walk(media_dir):
            if len(path.split('/')) <= 2:
                continue

            _, media_type, emotion_type = path.split("/")

            # If no images in these categories then remove the folders
            if len(filenames) == 0:
                os.removedirs(os.path.join(DATA_DIR, media_type, emotion_type))
                continue

            # Shuffle images and then save into train/dev/test guaranteeing
            # that at least one image is in the test set from each category
            random.shuffle(filenames)
            set_size = int(math.ceil(len(filenames) * dev_test_ratio))

            train_path = os.path.join(DATA_DIR, "train", media_type, emotion_type)
            for fn in filenames[:-2*set_size]:
                os.renames(os.path.join(path, fn), os.path.join(train_path, fn))

            dev_path = os.path.join(DATA_DIR, "dev", media_type, emotion_type)
            for fn in filenames[-2*set_size:-set_size]:
                os.renames(os.path.join(path, fn), os.path.join(dev_path, fn))

            test_path = os.path.join(DATA_DIR, "test", media_type, emotion_type)
            for fn in filenames[-set_size:]:
                os.renames(os.path.join(path, fn), os.path.join(test_path, fn))
    print("Data successfully split\n")


def load_data(update=False, remove_broken=False):
    """
    Load BAM dataset images from disk and return as numpy arrays

    Input:
        update - flag to specify if data should be recreated from the source
                 images or loaded from saved numpy arrays
        remove_broken - flag to specify if images that cannot be loaded from
                        disk should be removed (some BAM images are corrupt)
    """

    # Dictionaries used to create labels
    media_dict = dict((k, v) for v, k in enumerate(MEDIA_LABELS))
    num_media = len(media_dict)
    emotion_dict = dict((k, v) for v, k in enumerate(EMOTION_LABELS))
    num_emotion = len(emotion_dict)

    # Load saved numpy arrays if the data doesn't need to be regathered
    if not update:
        print("Attempting to load saved numpy arrays")
        try:
            X = np.load(os.path.join(DATA_DIR, INPUT_FILE))
            y_media = np.load(os.path.join(DATA_DIR, MEDIA_LABEL_FILE))
            y_emotion = np.load(os.path.join(DATA_DIR, EMOTION_LABEL_FILE))
            print("Completed loading all data")
            print(f"Input shape: {X['train'].shape}")
            print(f"Media labels shape: {y_media['train'].shape}")
            print(f"Emotion labels shape: {y_emotion['train'].shape}")
            return X, y_media, y_emotion
        except:
            print("Could not reload saved numpy arrays")

    # Create dictionaries for the input and labels depending on the set_type
    X = {}
    y_media = {}
    y_emotion = {}
    for set_type in ["train", "dev", "test"]:
        X[set_type] = []
        y_media[set_type] = []
        y_emotion[set_type] = []

    # Scan for images in the data directory
    print(f"Scanning '{DATA_DIR}'")
    for media_dir in os.scandir(DATA_DIR):
        if not media_dir.is_dir():
            continue
        for path, _, filenames in os.walk(media_dir):
            if len(path.split('/')) <= 3 or len(filenames) == 0:
                continue
            _, set_type, media_type, emotion_type = path.split("/")

            # Create the training labels as binary ndarrays
            media_label = ((np.arange(num_media) == media_dict[media_type])
                           * np.ones(num_media))
            emotion_label = ((np.arange(num_emotion) == emotion_dict[emotion_type])
                             * np.ones(num_emotion))

            # Attempt to open each image to add to training data
            for img_name in filenames:
                img_path = os.path.join(path, img_name)
                # If image cannot be opened then check if it should be removed
                try:
                    if os.stat(img_path).st_size == 0:
                        print(f"Could not open: {img_path}")
                        raise OSError
                    mid = img_name.split('.')[0]
                    img = imread(img_path)
                    img = imresize(img, IMG_SIZE)

                    # Add the reshaped img and its labels
                    if (img.shape[2] != 3):
                        continue
                    #print(img.shape, media_label.shape, emotion_label.shape)
                    X[set_type].append(np.expand_dims(img, axis=0))
                    y_media[set_type].append(np.expand_dims(media_label, axis=0))
                    y_emotion[set_type].append(np.expand_dims(emotion_label, axis=0))

                except (OSError, IndexError, KeyError, ValueError):
                    print(f"Could not open: {img_path}")
                    if remove_broken: os.remove(img_path)

    # Concatenate all the data and randomly sort using set seed
    seed = SEED
    np.random.seed(seed)
    for set_type in ["train", "dev", "test"]:
        X[set_type] = np.concatenate(X[set_type], axis=0).astype('float32')
        y_media[set_type] = np.concatenate(y_media[set_type], axis=0)
        y_emotion[set_type] = np.concatenate(y_emotion[set_type], axis=0)

        order = np.random.permutation(X[set_type].shape[0])
        X[set_type] = X[set_type][order]
        y_media[set_type] = y_media[set_type][order]
        y_emotion[set_type] = y_emotion[set_type][order]

    print("Preprocessing data")
    mean_pixel = BAM_MEAN #np.mean(X["train"])
    std_pixel = BAM_STD #np.std(X["train"])
    print(f'\ttrain mean: {mean_pixel}')
    print(f'\ttrain std: {std_pixel}')

    # TODO: cast everything as float32 for operation
    for set_type in ["train", "dev", "test"]:
        X[set_type] -= mean_pixel
        X[set_type] /= std_pixel

    # Save each dictionary as an npz file
    print("Completed loading all data")
    print(f"Input train shape: {X['train'].shape}")
    print(f"Media labels train shape: {y_media['train'].shape}")
    print(f"Emotion labels train shape: {y_emotion['train'].shape}")
    np.savez(os.path.join(DATA_DIR, INPUT_FILE), **X)
    np.savez(os.path.join(DATA_DIR, MEDIA_LABEL_FILE), **y_media)
    np.savez(os.path.join(DATA_DIR, EMOTION_LABEL_FILE), **y_emotion)

    return X, y_media, y_emotion

if __name__ == "__main__":
    # Split the data into train/dev/test sets
    split_data()

    # Load the data and reshape for training and evaluation
    X, y_media, y_emotion = load_data(update=False)

    for set_type in ["train", "dev", "test"]:
        total_media = np.sum(y_media[set_type], axis=0)
        total_emotion = np.sum(y_emotion[set_type], axis=0)

        print(f"Total images for each media category in {set_type} set:")
        for v, k in enumerate(MEDIA_LABELS):
            print(f"\t{k}: {total_media[v]}")
        print(f"Total images for each emotion category in {set_type} set:")
        for v, k in enumerate(EMOTION_LABELS):
            print(f"\t{k}: {total_emotion[v]}")
