import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave

from data_utils import DATA_DIR, INPUT_FILE, MEDIA_LABEL_FILE, EMOTION_LABEL_FILE
from preprocessing import load_data
from preprocessing import preprocess as our_preprocess
from squeezenet import SqueezeNet
#from model6 import ConvNet
from ourmodel import OurModel

IMG_SIZE = (128, 128)

def our_load(filename):
    """
    Load an image and resize to the specified size for use in our trained
    models. Return in a format for preprocess to work.
    """
    img = imread(filename)
    img = imresize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    return (img, )

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

# import from config file
class StyleConfig:
    """
    Class used to store the settings for a style transfer network
    """
    def __init__(self):
        # image sizes (for squeezenet)
        self.image_size = 192
        self.style_size = 512

        # layers used for the loss and how much to weight the loss
        self.content_layer = 3
        self.content_weight = 5e-2
        self.style_layers = [1, 4, 6, 7]
        self.style_weights = [200000, 500, 12, 1]
        self.tv_weight = 5e-2

        # transfer settings
        self.initial_lr = 3.0
        self.decayed_lr = 0.1
        self.decay_lr_at = 180
        self.num_iters = 200

# class takes in following inputs
class StyleNetwork:
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
        loss = self.config.content_weight * tf.reduce_sum(diff_squared)
        return loss

    def style_loss(self, features, style_targets):
        # calculate the loss from style
        loss = tf.constant(0, dtype=tf.float32)
        for i, l in enumerate(self.config.style_layers):
            G = self.gram_matrix(features[l])
            A = style_targets[i]
            loss += self.config.style_weights[i] * tf.reduce_sum((G - A)**2)
        return loss

    def tv_loss(self, transfer_img):
        # calculate total variational loss
        L = transfer_img[..., :-1, :]
        R = transfer_img[..., 1:, :]
        U = transfer_img[:, :-1, ...]
        D = transfer_img[:, 1:, ...]
        loss = self.config.tv_weight * (tf.reduce_sum((L - R)**2)
                                 + tf.reduce_sum((U - D)**2))
        return loss

    def loss(self, features, transfer_img, content_target, style_targets):
        # Sum up all of the losses
        style_loss_tensor = self.style_loss(features, style_targets)
        content_current = features[self.config.content_layer]
        content_loss_tensor = self.content_loss(content_current, content_target)
        tv_loss_tensor = self.tv_loss(transfer_img)
        return style_loss_tensor + content_loss_tensor + tv_loss_tensor

    def transfer(self, content_image, style_image):
        # Create placeholder for the input image that will be modified
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
            print(content_img.shape)
            content_img = np.squeeze(content_img)
        elif self.model_type == "squeezenet":
            content_img = squeezenet_preprocess(squeezenet_load(content_image,
                                                                self.config.image_size))
        features = self.model.extract_features(input_image)
        content_target = sess.run(features[self.config.content_layer],
                                  {input_image: content_img[None]})

        # Extract features from the style image
        if self.model_type == "ours":
            style_img = val_datagen.flow(our_load_image(style_image),
                                         batch_size=1)
            print(style_img.shape)
            style_img = np.squeeze(style_img)
        elif self.model_type == "squeezenet":
            style_img = squeezenet_preprocess(squeezenet_load(style_image,
                                                              self.config.style_size))
        style_target_vars = [self.gram_matrix(features[l])
                             for l in self.config.style_layers]
        style_targets = sess.run(style_target_vars,
                                 {input_image: style_img[None]})

        # Initialize generated image to content image and compute loss
        img_var = tf.Variable(content_img[None], name="image")
        features = self.model.extract_features(img_var)
        loss = self.loss(features, img_var, content_target, style_targets)

        # Create and initialize the Adam optimizer
        lr_var = tf.Variable(self.config.initial_lr, name="lr")
        # Create train_op that updates the generated image when run
        with tf.variable_scope("optimizer") as opt_scope:
            train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
        # Initialize the generated image and optimization variables
        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
        sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))
        # Create an op that will clamp the image values when run
        clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))

        # Transfer update loop
        for t in range(self.config.num_iters + 1):
            # Take an optimization step to update img_var
            sess.run(train_op)
            if t < self.config.decay_lr_at:
                sess.run(clamp_image_op)
            if t == self.config.decay_lr_at:
                sess.run(tf.assign(lr_var, self.config.decayed_lr))
            if t % 50 == 0:
                print(f"Iteration {t}")
                img = sess.run(img_var)
                imsave(f"style_{t}.png", np.squeeze(img))


if __name__ == "__main__":
    config = StyleConfig()
    model_type = "ours"
    model_path = 'weights/media_ckpt_best.h5'
    #weight_path = "weights/squeezenet.ckpt"
    content_image = "data/train/media_oilpaint/emotion_happy/119420229.jpg"
    #content_image = "styles/tubingen.jpg"
    style_image = "data/train/media_watercolor/emotion_happy/151337707.jpg"
    #style_image = "styles/composition_vii.jpg"
    with tf.Session() as sess:
        model = OurModel(save_path=model_path, sess=sess)
        #model = SqueezeNet(save_path=weight_path, sess=sess)
        # TODO: update model calls to model.model
        style_transfer = StyleNetwork(config, model, model_type)
        style_transfer.transfer(content_image, style_image)

