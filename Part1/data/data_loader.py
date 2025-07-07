import os
import pandas as pd
import numpy as np
import os
import sys
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


np.random.seed(38)


class Dataset:
    def __init__(self, imgs, objects, transforms=None):
        """Initialize the dataset class"""
        self.imgs = imgs
        self.classes = {}
        for i in range(len(objects)):
            self.classes[objects[i]] = i

        self.transforms = transforms

    def __getitem__(self, idx):
        """Function to prepare the images in the required format for the dataloader"""
        img = self.imgs[idx]
        label = self.imgs[idx].split('/')[-2]
        label = self.classes[label]

        img = Image.open(img)

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


def split():
    """Function to split the data into train, validation and test in the required format"""
    path = '../data/dataset/train/'
    objects = os.listdir(path)
    train_imgs, val_imgs = [], []
    for obj in objects:
        items = os.listdir(path + obj)
        items = [path + obj + '/' + item for item in items]
        train_imgs = train_imgs + items[1000:]
        val_imgs = val_imgs + items[:1000]

    path = '../data/dataset/test/'
    objects = os.listdir(path)
    test_imgs = []
    for obj in objects:
        items = os.listdir(path + obj)
        items = [path + obj + '/' + item for item in items]
        test_imgs = test_imgs + items

    return train_imgs, val_imgs, test_imgs, sorted(objects)


def load_data(batch_size=64):
    """Function to load the data and returning it in the dataloader format"""
    train, val, test, objects = split()
    augmentation = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    preprocessing = transforms.Compose([
        transforms.ToTensor(),  # automatic scale to [0, 1]
        transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768]
        )
    ])

    transform = transforms.Compose([augmentation, preprocessing])
    test_transform = transforms.Compose([preprocessing])

    train_dataset, val_dataset, test_dataset = Dataset(train, objects, transform), Dataset(val, objects, transform), Dataset(test, objects, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def sample():
    """Function for testing the augmentations with a sample image"""
    color_jitter = transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25
    )
    random_affine = transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    )

    path = '../data/dataset/train/'
    objects = os.listdir(path)
    items = os.listdir(path + objects[0])
    items = [path + objects[0] + '/' + item for item in items]
    img = Image.open(items[0])

    augmented_color = [color_jitter(img) for _ in range(10)]
    augmented_affine = [random_affine(img) for _ in range(10)]

    return img, augmented_color, augmented_affine


def get_label_map() -> dict:
    """Function for returning the corresponding object name to the label"""
    path = '../data/dataset/train/'
    objects = sorted(os.listdir(path))
    label_map = {}
    for i in range(len(objects)):
        obj = objects[i]
        label_map[i] = obj

    return label_map


def get_sample_test_img():
    """Function to return a sample test image"""
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768]
        )
    ])
    path = '../data/dataset/test/'
    objects = os.listdir(path)
    items = os.listdir(path + objects[0])
    items = [path + objects[0] + '/' + item for item in items]
    img = Image.open(items[0])

    return preprocessing(img)


def get_test():
    """Function to load the test dataset"""
    train, val, test, objects = split()
    augmentation = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    preprocessing = transforms.Compose([
        transforms.ToTensor(),  # automatic scale to [0, 1]
        transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768]
        )
    ])

    test_transform = transforms.Compose([preprocessing])

    return Dataset(test, objects, test_transform)
