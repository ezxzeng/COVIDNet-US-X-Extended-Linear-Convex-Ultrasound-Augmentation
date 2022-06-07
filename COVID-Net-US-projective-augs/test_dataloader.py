import os
import json
from dataset import COVID_US_Datset, get_dataloaders
from random_warp import WarpProjective
from matplotlib import pyplot as plt
from skimage.io import imread, imshow

if __name__ == "__main__":
    with open("hyperparameters.json", "r") as f:
        hp = json.load(f)

    hp["data_params"]["batch_size"] = 1
    dataloaders, _ = get_dataloaders(**hp["data_params"])

    train_dataloader = dataloaders["train"]

    for train_features, train_labels in train_dataloader:
        # Display image and label.
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")
