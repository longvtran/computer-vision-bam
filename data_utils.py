import os
import numpy as np
from scipy.misc import imread, imresize

# H x W that the images should be reshaped to
IMG_SIZE = (128, 128)

# Directory where the media/emotion subdirectories are located
DATA_DIR = "data/"
# SQLite file needs to go at "data/bam.sqlite"
BAM = "bam.sqlite"
# Names for the saved numpy arrays that include the reshaped and formated data
INPUT_FILE = "input.npy"
MEDIA_LABEL_FILE = "media_label.npy"
EMOTION_LABEL_FILE = "emotion_label.npy"

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
            print(f"Input shape: {X.shape}")
            print(f"Media labels shape: {y_media.shape}")
            print(f"Emotion labels shape: {y_emotion.shape}")
            return X, y_media, y_emotion
        except:
            print("Could not reload saved numpy arrays")

    X = []
    y_media = []
    y_emotion = []

    # Scan for images in the data directory
    print(f"Scanning '{DATA_DIR}'")
    for media_dir in os.scandir(DATA_DIR):
        if not media_dir.is_dir():
            continue
        for path, _, filenames in os.walk(media_dir):
            if len(path.split('/')) <= 2 or len(filenames) == 0:
                continue
            _, media_type, emotion_type = path.split("/")

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
                    X.append(np.expand_dims(img, axis=0))
                    y_media.append(np.expand_dims(media_label, axis=0))
                    y_emotion.append(np.expand_dims(emotion_label, axis=0))

                except (OSError, IndexError, KeyError):
                    print(f"Could not open: {img_path}")
                    if remove_broken: os.remove(img_path)

    # Concatenate all the data
    X = np.concatenate(X, axis=0)
    y_media = np.concatenate(y_media, axis=0)
    y_emotion = np.concatenate(y_emotion, axis=0)

    # Randomly shuffle the data and labels
    order = np.random.permutation(X.shape[0])
    X = X[order]
    y_media = y_media[order]
    y_emotion = y_emotion[order]

    print("Completed loading all data")
    print(f"Input shape: {X.shape}")
    print(f"Media labels shape: {y_media.shape}")
    print(f"Emotion labels shape: {y_emotion.shape}")
    np.save(os.path.join(DATA_DIR, INPUT_FILE), X)
    np.save(os.path.join(DATA_DIR, MEDIA_LABEL_FILE), y_media)
    np.save(os.path.join(DATA_DIR, EMOTION_LABEL_FILE), y_emotion)
    return X, y_media, y_emotion

if __name__ == "__main__":
    X, y_media, y_emotion = load_data(update=True)
    total_media = np.sum(y_media, axis=0)
    total_emotion = np.sum(y_emotion, axis=0)

    print("Total images for each media category:")
    for v, k in enumerate(MEDIA_LABELS):
        print(f"\t{k}: {total_media[v]}")
    print("Total images for each emotion category:")
    for v, k in enumerate(EMOTION_LABELS):
        print(f"\t{k}: {total_emotion[v]}")
