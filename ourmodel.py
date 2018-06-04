import tensorflow as tf
from model6 import ConvNet

class OurModel():
    """
    This class only works with the model6 neural network structure. To add
    additional network structures you will have to rewrite them in this style.
    """
    def extract_features(self, inputs=None, reuse=True):
        if inputs is None:
            inputs = self.image
        inputs = tf.keras.layers.Input(tensor=inputs)

        initializer = tf.variance_scaling_initializer(scale=2.0)
        layers = []

        with tf.variable_scope('features', reuse=reuse):
            """
            Original model structure follows (replace below with alternate
            model to adapt this class for other model structures)
            """
            # First layer
            x = tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(5,5), padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3)
                )(inputs)
            layers.append(x)

            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
            x = tf.keras.layers.BatchNormalization()(x)

            # Media layers
            x_media = tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3,3), padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3)
                )(x)
            layers.append(x_media)

            x_media = tf.keras.layers.MaxPooling2D(2, 2)(x_media)
            x_media = tf.keras.layers.BatchNormalization()(x_media)
            x_media = tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(3,3), padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3)
                )(x_media)
            layers.append(x_media)

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
            layers.append(x_emotion)

            x_emotion = tf.keras.layers.MaxPooling2D(2, 2)(x_emotion)
            x_emotion = tf.keras.layers.BatchNormalization()(x_emotion)
            x_emotion = tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(3,3), padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3)
                )(x_emotion)
            layers.append(x_emotion)

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

        self.model = tf.keras.Model(inputs=inputs,
                                    outputs=[output_media, output_emotion])

        return layers

    def __init__(self, save_path=None, sess=None):
        self.num_classes_media = 7
        self.num_classes_emotion = 4
        self.layers = []
        self.outputs = []

        self.image = tf.placeholder(
                'float',
                shape=[None, 128, 128, 3],
                name='input_image'
            )

        self.layers = self.extract_features(self.image, reuse=False)

        # Load the saved weights if available
        if save_path is not None:
            saved_model = tf.keras.models.load_model(save_path)
            self.model.set_weights(saved_model.get_weights()) 

class OurModelBad():
    """
    This model currently does not work because I was unable to replace the
    input layer with the image variable for style transfer.
    """
    def __init__(self, save_path=None, sess=None):
        if save_path is None:
            self.model = ConvNet()
        else:
            self.model = tf.keras.models.load_model(save_path)

    def extract_features(self, input_image=None):
        if input_image is None:
            input_image = self.model.input
        layers = []
        input_layer = tf.keras.Input(tensor=input_image)
        self.model.layers.pop(0)
        output_layer = self.model(input_layer)
        our_model = tf.keras.Model(input_layer, output_layer)
        for l in our_model.layers[1].layers:
            if l.name[:6] == "conv2d":
                layers.append(l.output)
        return layers

