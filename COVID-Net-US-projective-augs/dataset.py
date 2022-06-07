import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.io import read_image

from helper import custom_transform
from random_warp import WarpProjective, WarpAffine


def add_channels():
    # return transforms.Lambda(lambda x: x.convert("RGB"))
    return transforms.Lambda(lambda x: x.expand(3, -1, -1))
    # return transforms.Lambda(lambda x: x.repeat(3,1,1))
    

TRANSFORM_NAME_TO_FUNCTION = {
    "ToPILImage": transforms.ToPILImage,
    "Grayscale": transforms.Grayscale,
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "RandomRotation": transforms.RandomRotation,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "CenterCrop": transforms.CenterCrop,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
    "Resize": transforms.Resize,
    "ColorJitter": transforms.ColorJitter,
    "add_channels": add_channels
}
LABEL_TO_VAL = {"positive": 1, "negative": 0}


def target_transform_fn(label):
    return LABEL_TO_VAL[label]


class COVID_US_Datset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=None,
        target_transform=None,
        warp_convex_ultrasound=None,
    ):
        if type(annotations_file) == list:
            self.img_labels = pd.concat([pd.read_csv(file, header=None, sep=" ") for file in annotations_file])
        else:
            self.img_labels = pd.read_csv(annotations_file, header=None, sep=" ")
        self.img_dir = img_dir
        if transform is not None:
            self.transform = transforms.Compose(
                [
                    TRANSFORM_NAME_TO_FUNCTION[transform_item["name"]](
                        **transform_item["params"]
                    )
                    for transform_item in transform
                ]
            )
        else:
            self.transform = None
        if target_transform is not None:
            self.target_transform = target_transform
        else:
            self.target_transform = target_transform_fn

        self.warp_convex_ultrasound = None
        if warp_convex_ultrasound is not None:
            if warp_convex_ultrasound["name"] == "WarpProjective":
                self.warp_convex_ultrasound = WarpProjective(
                    **warp_convex_ultrasound["params"]
                )
            elif warp_convex_ultrasound["name"] == "WarpAffine":
                self.warp_convex_ultrasound = WarpAffine(
                    **warp_convex_ultrasound["params"]
                )
            else:
                print(
                    "convex warping {} not implemented".format(
                        warp_convex_ultrasound["name"]
                    )
                )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, filename)
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.warp_convex_ultrasound is not None:
            vid_id = str(int(self.img_labels.iloc[idx, 0]))
            if vid_id in self.warp_convex_ultrasound.points_info:
                image = TF.to_pil_image(image)
                image = TF.to_grayscale(image)
                image = np.array(image)
                image = self.warp_convex_ultrasound(image, vid_id)
                image = TF.to_tensor(image)
            # else:
                # print(vid_id)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_dataloaders(
    image_directory,
    train_annotations_file,
    valid_annotations_file,
    train_transforms,
    valid_transforms,
    batch_size=64,
    warp_convex_ultrasound=None,
    test_annotations_file=""
):
    # Load data from folders
    dataset = {
        "train": COVID_US_Datset(
            train_annotations_file,
            image_directory,
            train_transforms,
            warp_convex_ultrasound=warp_convex_ultrasound,
        ),
        "valid": COVID_US_Datset(
            valid_annotations_file, image_directory, valid_transforms
        ),
    }

    # Size of train and validation data
    dataset_sizes = {"train": len(dataset["train"]), "valid": len(dataset["valid"])}

    # Create iterators for data loading
    dataloaders = {
        "train": DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
        ),
        "valid": DataLoader(
            dataset["valid"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
        ),
    }

    # Print the train and validation data sizes
    print(
        "Training-set size:",
        dataset_sizes["train"],
        "\nValidation-set size:",
        dataset_sizes["valid"],
    )

    if test_annotations_file:
        dataset["test"] = COVID_US_Datset(
            test_annotations_file, image_directory, valid_transforms
        )
        dataset_sizes["test"] = len(dataset["test"])
        dataloaders["test"] = DataLoader(
            dataset["test"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
        )


    return dataloaders, dataset_sizes


def get_test_dataloader(
    image_directory,
    test_annotations_file,
    test_transforms,
    batch_size=64,
):
    test_dataset = COVID_US_Datset(
        test_annotations_file,
        image_directory,
        test_transforms,
        warp_convex_ultrasound=None,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_count(),
        pin_memory=True,
        drop_last=True,
    )

    return test_dataloader, len(test_dataset)
