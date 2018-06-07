import os
import numpy as np


class SqueezeStyleConfig:
    """
    Class used to store the settings for a style transfer network
    """
    def __init__(self):
        # Dictionary to translate a parameter name to its value
        self.val = dict()
        # List of parameters that should be saved in plain text when exported
        self.exportable = []
        # List of paramters that will have random values that can be updated
        self.randomized = []

        # Non-randomized parameters
        # Image sizes (for squeezenet)
        self.val["image_size"] = 192
        self.val["style_size"] = 512

        # Transfer iterations
        self.val["decay_lr_at"] = 180
        self.val["num_iters"] = 200

        # Transfer Layers
        self.val["content_layer"] = 3
        self.val["style_layers"] = [1, 4, 6, 7]

        self.exportable = list(self.val.keys())

        # Randomized parameters
        # Transfer weights
        self._add_random_value("content_weight", 1e-4, 1)
        self._add_random_value("style_weights", [10000, 50, 4, .5],
                                                [1000000, 2000, 25, 3])
        self._add_random_value("tv_weight", 1e-4, 1)
        self._add_random_value("clamp_value", 0.5, 4)

        # Transfer settings
        self._add_random_value("initial_lr", 0.5, 5.0)
        self._add_random_value("decayed_lr", 0.05, 0.5)
        self.update(0)

    def _add_random_value(self, name, start, end):
        # Add a random parameter that can be in the range [start, end)
        if (start > end):
            raise ValueError(f"parameter '{name}' start value greater than end value")
        self.randomized.append(name)
        self.val[f"{name}_start"] = start
        self.val[f"{name}_end"] = end
        self.exportable.append(name)

    def _update_value(self, name):
        # Updates the value of a randomized paramter to be in range [start, end)
        start = self.val[f"{name}_start"]
        end = self.val[f"{name}_end"]
        if name == "style_weights":
            self.val[name] = [((np.random.rand() * (b - a)) + a)
                              for a, b in zip(start, end)]
        else:
            self.val[name] = (np.random.rand() * (end - start)) + start

    def update(self, test_number):
        # Update all randomized parameters with new values
        for name in self.randomized:
            self._update_value(name)

        # Set the test number for backup
        self.test_number = test_number

    def save_settings(self, folder_name):
        # Print out all of the parameters
        file_name = f"config_{self.test_number}.txt"
        with open(os.path.join(folder_name, file_name), 'w') as f:
            for name in self.exportable:
                f.write(f"{name}: {self.val[name]}\n")

class OurStyleConfig:
    """
    Class used to store the settings for a style transfer network
    """
    def __init__(self):
        # Dictionary to translate a parameter name to its value
        self.val = dict()
        # List of parameters that should be saved in plain text when exported
        self.exportable = []
        # List of paramters that will have random values that can be updated
        self.randomized = []

        # Non-randomized parameters
        # Image sizes (for squeezenet)
        self.val["image_size"] = 192
        self.val["style_size"] = 512

        # Transfer iterations
        self.val["decay_lr_at"] = 180
        self.val["num_iters"] = 200

        # Transfer Layers
        self.val["content_layer"] = 1
        self.val["style_layers"] = [0, 2, 3, 4]

        self.exportable = list(self.val.keys())

        # Randomized parameters
        # Transfer weights
        self._add_random_value("content_weight", 1e-4, 1)
        self._add_random_value("style_weights", [1000, .5, 50, .5],
                                                [1000000, 10, 2000, 10])
        self._add_random_value("tv_weight", 1e-4, 1)
        self._add_random_value("clamp_value", 0.5, 4)

        # Transfer settings
        self._add_random_value("initial_lr", 0.5, 5.0)
        self._add_random_value("decayed_lr", 0.05, 0.5)
        self.update(0)

    def _add_random_value(self, name, start, end):
        # Add a random parameter that can be in the range [start, end)
        if (start > end):
            raise ValueError(f"parameter '{name}' start value greater than end value")
        self.randomized.append(name)
        self.val[f"{name}_start"] = start
        self.val[f"{name}_end"] = end
        self.exportable.append(name)

    def _update_value(self, name):
        # Updates the value of a randomized paramter to be in range [start, end)
        start = self.val[f"{name}_start"]
        end = self.val[f"{name}_end"]
        if name == "style_weights":
            self.val[name] = [((np.random.rand() * (b - a)) + a)
                              for a, b in zip(start, end)]
        else:
            self.val[name] = (np.random.rand() * (end - start)) + start

    def update(self, test_number):
        # Update all randomized parameters with new values
        for name in self.randomized:
            self._update_value(name)

        # Set the test number for backup
        self.test_number = test_number

    def save_settings(self, folder_name):
        # Print out all of the parameters
        file_name = f"config_{self.test_number}.txt"
        with open(os.path.join(folder_name, file_name), 'w') as f:
            for name in self.exportable:
                f.write(f"{name}: {self.val[name]}\n")

