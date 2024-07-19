import numpy as np
import pathlib


def get_data():

    dataset_path = pathlib.Path(__file__).parent / "MNIST" / "mnist.npz"

    with np.load(dataset_path) as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels
