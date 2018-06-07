import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave

from data_utils import DATA_DIR, INPUT_FILE, MEDIA_LABEL_FILE, EMOTION_LABEL_FILE
from preprocessing import load_data
from preprocessing import preprocess as our_preprocess
from squeezenet import SqueezeNet
#from model6 import ConvNet
from ourmodel import OurModel
from style_config import StyleConfig

TEST_DIR = "tests/"

# Constants to normalize images for style transfer with squeezenet
SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Our model input image size
IMG_SIZE = (128, 128)

# Constants to normalize images for style transfer with our models
BAM_MEAN = 131.5598075494837
BAM_STD = 81.77253959615437

def our_load_image(filename):
    """
    Load an image and resize to the specified size for use in our trained
    models. Return in a format for preprocess to work.
    """
    img = imread(filename)
    img = imresize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    return img

def our_preprocess(img):
    return (img.astype(np.float32) - BAM_MEAN) / BAM_STD

def squeezenet_load(filename, size=None):
    """
    Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img

def squeezenet_preprocess(img):
    """
    Preprocess an image for squeezenet.
    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


class StyleNetwork:
    """
    Style network class that takes in a configuration class, a model that has
    and "extract_features" function and the string name of the model type
    (i.e. "ours" or "squeezenet")
    """
    def __init__(self, config, model, model_type):
        self.config = config
        self.model = model
        self.model_type = model_type

    def gram_matrix(self, features, normalize=True):
        C = tf.shape(features)[-1]
        feature_map = tf.reshape(features, [-1, C])
        gram = tf.transpose(feature_map) @ feature_map
        if normalize:
            gram /= tf.reduce_prod(tf.cast(tf.shape(features), tf.float32))
        return gram

    def content_loss(self, content_current, content_target):
        # calculate the loss from content
        diff_squared = (content_current - content_target)**2
        loss = self.config.val["content_weight"] * tf.reduce_sum(diff_squared)
        return loss

    def style_loss(self, features, style_targets):
        # calculate the loss from style
        loss = tf.constant(0, dtype=tf.float32)
        for i, l in enumerate(self.config.val["style_layers"]):
            G = self.gram_matrix(features[l])
            A = style_targets[i]
            loss += self.config.val["style_weights"][i] * tf.reduce_sum((G - A)**2)
        return loss

    def tv_loss(self, transfer_img):
        # calculate total variational loss
        L = transfer_img[..., :-1, :]
        R = transfer_img[..., 1:, :]
        U = transfer_img[:, :-1, ...]
        D = transfer_img[:, 1:, ...]
        loss = self.config.val["tv_weight"] * (tf.reduce_sum((L - R)**2)
                                        + tf.reduce_sum((U - D)**2))
        return loss

    def loss(self, features, transfer_img, content_target, style_targets):
        # Sum up all of the losses
        style_loss_tensor = self.style_loss(features, style_targets)
        content_current = features[self.config.val["content_layer"]]
        content_loss_tensor = self.content_loss(content_current, content_target)
        tv_loss_tensor = self.tv_loss(transfer_img)
        return style_loss_tensor + content_loss_tensor + tv_loss_tensor

    def transfer(self, content_image, style_image, output_folder=""):
        # Create placeholder for the input image that will be modified
        if self.model_type == "ours":
            input_image = tf.placeholder('float', shape=[None,128,128,3],
                                         name='input_image')
        elif self.model_type == "squeezenet":
            input_image = tf.placeholder('float', shape=[None,None,None,3],
                                         name='input_image')

        # Extract features from the content image
        if self.model_type == "ours":
            train_data, _a, _b = load_data(DATA_DIR, INPUT_FILE, 
                                           MEDIA_LABEL_FILE,
                                           EMOTION_LABEL_FILE)
            _, val_datagen = our_preprocess(train_data)
            content_img = val_datagen.flow(our_load_image(content_image),
                                           batch_size=1)
            content_img = np.squeeze(content_img[0])
        elif self.model_type == "squeezenet":
            content_img = squeezenet_preprocess(squeezenet_load(content_image,
                                                                self.config.val["image_size"]))
        features = self.model.extract_features(input_image)
        content_target = sess.run(features[self.config.val["content_layer"]],
                                  {input_image: content_img[None]})

        # Extract features from the style image
        if self.model_type == "ours":
            style_img = val_datagen.flow(our_load_image(style_image),
                                         batch_size=1)
            style_img = np.squeeze(style_img[0])
        elif self.model_type == "squeezenet":
            style_img = squeezenet_preprocess(squeezenet_load(style_image,
                                                              self.config.val["style_size"]))
        style_target_vars = [self.gram_matrix(features[l])
                             for l in self.config.val["style_layers"]]
        style_targets = sess.run(style_target_vars,
                                 {input_image: style_img[None]})

        # Initialize generated image to content image and compute loss
        img_var = tf.Variable(content_img[None], name="image")
        features = self.model.extract_features(img_var)
        loss = self.loss(features, img_var, content_target, style_targets)

        # Create and initialize the Adam optimizer
        lr_var = tf.Variable(self.config.val["initial_lr"], name="lr")
        # Create train_op that updates the generated image when run
        with tf.variable_scope("optimizer") as opt_scope:
            train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
        # Initialize the generated image and optimization variables
        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
        sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))
        # Create an op that will clamp the image values when run
        clamp_image_op = tf.assign(
                img_var,
                tf.clip_by_value(img_var,
                                 -self.config.val["clamp_value"],
                                 self.config.val["clamp_value"])
            )

        # Transfer update loop
        test_number = self.config.test_number
        for t in range(self.config.val["num_iters"] + 1):
            # Take an optimization step to update img_var
            sess.run(train_op)
            if t < self.config.val["decay_lr_at"]:
                sess.run(clamp_image_op)
            if t == self.config.val["decay_lr_at"]:
                sess.run(tf.assign(lr_var, self.config.val["decayed_lr"]))
            if t % 50 == 0:
                print(f"Iteration {t}")
                img = sess.run(img_var)
                file_name = f"style_{test_number}_{t}.png"
                imsave(os.path.join(output_folder, file_name), np.squeeze(img))


if __name__ == "__main__":
    """
    Do style transfer with the specified saved model and images
    """
    # Number of different randomized parameter configurations to test
    num_tests = 10
    test_folder = TEST_DIR

    # Load a configuration file to use
    config = StyleConfig()

    # Select a model
    #model_type = "ours"
    model_type = "squeezenet"
    #model_path = 'weights/media_ckpt_best.h5'
    weight_path = "weights/squeezenet.ckpt"

    # Select content and style images
    #content_image = "data/train/media_oilpaint/emotion_peaceful/20164695.jpg"
    content_image = "styles/tubingen.jpg"
    #style_image = "data/train/media_oilpaint/emotion_scary/42142915.jpg"
    style_image = "styles/composition_vii.jpg"

    # Test loop
    for i in range(num_tests):
        # Call the update function to get new random values for the randomized
        # weights and other randomized config settings.
        config.update(i)
        config.save_settings(test_folder)

        # Load the model and do the style transfer
        with tf.Session() as sess:
            #model = OurModel(save_path=model_path, sess=sess)
            model = SqueezeNet(save_path=weight_path, sess=sess)
            style_transfer = StyleNetwork(config, model, model_type)
            style_transfer.transfer(content_image, style_image, test_folder)

        # Clear the graph between each test
        tf.reset_default_graph()

