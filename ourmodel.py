import pdb
import tensorflow as tf
from model6 import ConvNet as Model6

class OurModel():
    def __init__(self, save_path):
        self.model = tf.keras.models.load_model(save_path)

    def extract_features():
        layers = self.model.get_weights()
        pdb.set_trace()
        return layers

class OurModel2(tf.keras.Model):
    def extract_features(self, inputs=None, reuse=True):
        if inputs is None:
            inputs = self.image

        initializer = tf.variance_scaling_initializer(scale=2.0)
        layers = []

        # First layer
        x = tf.keras.layers.Conv2D(
                filters=64, kernel_size=(5,5), padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(1e-3)
            )(inputs)
        layers.append(x.output)

        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Media layers
        x_media = tf.keras.layers.Conv2D(
                filters=32, kernel_size=(3,3), padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(1e-3)
            )(x)
        layers.append(x_media.output)

        x_media = tf.keras.layers.MaxPooling2D(2, 2)(x_media)
        x_media = tf.keras.layers.BatchNormalization()(x_media)
        x_media = tf.keras.layers.Conv2D(
                filters=16, kernel_size=(3,3), padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(1e-3)
            )(x_media)
        layers.append(x_media.output)

        x_media = tf.keras.layers.MaxPooling2D(2, 2)(x_media)
        x_media = tf.keras.layers.Flatten()(x_media)
        x_media = tf.keras.layers.Dense(
                units=128,
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                activation=tf.nn.relu
            )(x_media)

        x_media = tf.keras.layers.Dropout(rate=0.65)(x_media)
        output_media = tf.keras.layers.Dense(
                self.num_classes_media,
                kernel_initializer=initializer,
                activation='softmax',
                name='output_media'
            )(x_media)

        # Emotion layers
        x_emotion = tf.keras.layers.Conv2D(
                filters=32, kernel_size=(3,3), padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(1e-3)
            )(x)
        layers.append(x_emotion.output)

        x_emotion = tf.keras.layers.MaxPooling2D(2, 2)(x_emotion)
        x_emotion = tf.keras.layers.BatchNormalization()(x_emotion)
        x_emotion = tf.keras.layers.Conv2D(
                filters=16, kernel_size=(3,3), padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(1e-3)
            )(x_emotion)
        layers.append(x_emotion.output)

        x_emotion = tf.keras.layers.MaxPooling2D(2, 2)(x_emotion)
        x_emotion = tf.keras.layers.Flatten()(x_emotion)
        x_emotion = tf.keras.layers.Dense(
                units=128,
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                activation=tf.nn.relu
            )(x_emotion)

        x_emotion = tf.keras.layers.Dropout(rate=0.65)(x_emotion)
        output_emotion = tf.keras.layers.Dense(
                self.num_classes_emotion,
                kernel_initializer=initializer,
                activation='softmax',
                name='output_emotion'
            )(x_emotion)

        return layers, [output_media, output_emotion]

    def __init__(self, num_classes_media=7, num_classes_emotion=4,
                 save_path=None, sess=None):
        """
        Build a convolutional network using Keras Functional API. This network reads
        the image image arrays as input and produces two prediction outputs on the
        media or emotion class. The architecture consists of two sequences that share
        weights in the first Conv2D and MaxPooling2D layer

        Inputs:
            - num_classes_media: the number of media classes
            - num_classes_emotion: the number of emotion classes
            - training: a boolean that indicates whether the model is currently in
            training phase or not

        Returns:
            - a keras model instance
        """
        self.num_classes_media = num_classes_media
        self.num_classes_emotion = num_classes_emotion
        self.layers = []
        self.outputs = []

        # Set up the input, layers and output
        self.image = tf.placeholder(
                'float',
                shape=[None, 128, 128, 3],
                name='input_image'
            )
        self.layers, self.outputs = self.extract_features(self.image)
        super().__init__(inputs=self.image, outputs=self.outputs)

        # Load the saved weights if available
        if save_path is not None:
            self.load_weights(save_path)

    #def call(self, inputs, training=False):
        #x = self.dense1(inputs)
        #if training:
            #x = self.dropout(x, training=training)
        #return self.dense2(x)
