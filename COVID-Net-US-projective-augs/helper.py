import json
import numpy as np

import torch
from torchvision import transforms

# Set default device as gpu, if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_hyperparameter(hp_path="hyperparameters.json"):
    with open(hp_path, "r") as f:
        hp = json.load(f)

    return hp


def ensure_image_mode(image, expected_mode="int"):
    if expected_mode != "int":
        if "int" in image.dtype.name:
            image = image.astype(np.float32)
            image = image / 255
    else:
        if "int" not in image.dtype.name:
            image = image * 255
            image = image.astype(int)
    return image


def custom_transform(transform_fn, expected_mode=None):
    def get_transforms_lambda(**kwargs):
        def custom_transform_function(image):
            np_image = np.array(image)
            if expected_mode is not None:
                np_image = ensure_image_mode(np_image, expected_mode)
            transformed_img = transform_fn(np_image, **kwargs)
            return transformed_img

        return transforms.Lambda(custom_transform_function)

    return get_transforms_lambda


def vary_value(mean, distribution="uniform", plus_minus=0, scale=1.0):
    if distribution == "uniform":
        return np.random.uniform(mean - plus_minus, mean + plus_minus)
    elif distribution == "normal":
        return np.random.normal(mean, scale=scale)
