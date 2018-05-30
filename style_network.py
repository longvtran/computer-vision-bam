import tensorflow as tf

# import from config file
class StyleConfig:
    """
    Class used to store the settings for a style transfer network
    """
    def __init__(self):
        # paths to the images used and their size (must be square dimensions)
        self.content_img
        self.style_img
        self.image_size
        self.style_size

        # layers used for the loss and how much to weight the loss
        self.content_layer
        self.content_weight
        self.style_layers
        self.style_weights
        self.tv_weight

        # transfer settings
        self.num_iters

# class takes in following inputs
class StyleNetwork:
    def __init__(self, config):
        self.config = config

    def load_weights(self):
        # load the saved weights into the model
        # TODO:
        pass

    def gram_matrix(self, features, normalize=True):
        feature_map = tf.expand_dims(features, 0)
        gram = tf.transpose(feature_map) @ feature_map
        if normalize:
            gram /= tf.reduce_prod(tf.cast(tf.shape(features), tf.float32))
        return gram

    def style_loss(self, style_targets):
        # calculate the loss from style
        loss = tf.constant(0, dtype=tf.float32)
        for i, l in enumerate(self.style_layers):
            G = self.gram_matrix(self.model.extract_features()[l])
            A = style_targets[i]
            loss += self.style_weights[i] * tf.reduce_sum((G - A)**2)
        return loss

        style_loss = 0
        for i in range(len(self.style_layers)):
            G = style_targets[i]
            A = self.gram_matrix(self.model.extract_features()[self.style_layers[i]])
            style_loss += self.style_weights[i] * tf.reduce_sum((G-A)**2)
        return style_loss

    def content_loss(self, transfer_img):
        # calculate the loss from content
        diff_squared = (transfer_img - self.content_img)**2
        loss = self.content_weight * tf.reduce_sum(diff_squared)
        return loss

    def tv_loss(self, transfer_img):
        # calculate total variational loss
        L = transfer_img[..., :-1, :]
        R = transfer_img[..., 1:, :]
        U = transfer_img[:, :-1, ...]
        D = transfer_img[:, 1:, ...]
        loss = self.tv_weight * (tf.reduce_sum((L - R)**2)
                                 + tf.reduce_sum((U - D)**2))
        return loss

    def loss(self):
        # loss from style
        features = self.model.extract_features()
        style_target_vars = []
        for l in style_layers:
            style_target_vars.append(self.gram_matrix(features[l]))
        # TODO: fix feed dictionary
        style_targets = sess.run(style_target_vars,
                                 {model.image: self.style_img})
        style_loss_tensor = self.style_loss(features, style_targets)

        # loss from content
        # TODO: content_img_test what?

        # tv loss
        # TODO: how to feed in transfer_img
        tv_loss_tensor = tv_loss(model.image)

        # combined loss
        return style_loss_tensor + content_loss_tensor + tv_loss_tensor

    def train(self):
        # train the style network, save images for each iteration?
        # TODO:
        pass

