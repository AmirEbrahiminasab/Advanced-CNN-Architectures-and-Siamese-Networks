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


def get_example() -> list:
    """Function to return 3 images from 2 people"""
    path = '../data/dataset/train/'
    celeb = os.listdir(path)

    choice = np.random.choice(celeb, 2, replace=False)
    path_1 = os.listdir(path + choice[0])
    path_2 = os.listdir(path + choice[1])
    while len(path_1) < 3 or len(path_2) < 3:
        choice = np.random.choice(celeb, 2, replace=False)
        path_1 = os.listdir(path + choice[0])
        path_2 = os.listdir(path + choice[1])

    images_1 = np.random.choice(path_1, 3, replace=False)
    images_2 = np.random.choice(path_2, 3, replace=False)
    images_1 = [path + choice[0] + '/' + img for img in images_1]
    images_2 = [path + choice[1] + '/' + img for img in images_2]

    return images_1 + images_2


class Dataset:
    def __init__(self, imgs, transforms=None):
        """Initialize the dataset class"""
        self.imgs = imgs
        self.person_to_img = {}
        for img in self.imgs:
            name = img.split('/')[-2]
            if name not in self.person_to_img:
                self.person_to_img[name] = []

            self.person_to_img[name].append(img)

        self.persons = list(self.person_to_img.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        """Function to prepare the images in the required format for the dataloader"""
        img1, img2, label = None, None, None
        if np.random.random() > 0.5:
            person = np.random.choice(self.persons)
            images = self.person_to_img[person]
            while len(images) < 2:
                person = np.random.choice(self.persons)
                images = self.person_to_img[person]

            img1, img2 = np.random.choice(images, 2, replace=False)
            label = 1
        else:
            person1, person2 = np.random.choice(self.persons, 2, replace=False)
            img1 = np.random.choice(self.person_to_img[person1])
            img2 = np.random.choice(self.person_to_img[person2])
            label = 0

        img1, img2 = Image.open(img1), Image.open(img2)

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        else:
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.imgs)


def split():
    """Function to split the data into train, validation and test in the required format"""
    path = '../data/dataset/train/'
    ls = os.listdir(path)
    train_imgs, val_imgs = [], []

    for i in ls:
        imgs = os.listdir(path + i)
        if len(imgs) > 3:
            sz = len(imgs)
            for j in range(sz//2):
                val_imgs.append(path + i + '/' + imgs[j])

            for j in range(sz//2, sz):
                train_imgs.append(path + i + '/' + imgs[j])
        else:
            for j in range(len(imgs)):
                train_imgs.append(path + i + '/' + imgs[j])

    path = '../data/dataset/val/'
    ls = os.listdir(path)
    test_imgs = []
    for i in ls:
        imgs = os.listdir(path + i)
        for j in imgs:
            test_imgs.append(path + i + '/' + j)

    return train_imgs, val_imgs, test_imgs


def load_data(batch_size=8):
    """Function to load the data and returning it in the dataloader format"""
    train, val, test = split()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset, val_dataset, test_dataset = Dataset(train, transform), Dataset(val, transform), Dataset(test, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_test() -> tuple:
    """Function to return test images and its corresponding path"""
    path = '../data/dataset/val/'
    ls = os.listdir(path)
    test_imgs = []
    for i in ls:
        imgs = os.listdir(path + i)
        for j in imgs:
            test_imgs.append(path + i + '/' + j)

    test_set = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    for image in test_imgs:
        img = Image.open(image)
        img = transform(img)
        test_set.append(img)

    return test_set, test_imgs


def get_sample() -> tuple:
    """Function to return sample for t-SNE visualization"""
    path = '../data/dataset/val/'
    ls = os.listdir(path)
    test_imgs = []
    cnt = 0
    for i in ls:
        imgs = os.listdir(path + i)
        if len(imgs) < 3:
            continue
        cnt += 1
        if cnt > 5:
            break
        for j in imgs[:3]:
            test_imgs.append(path + i + '/' + j)

    test_set = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    for image in test_imgs:
        img = Image.open(image)
        img = transform(img)
        img = img.unsqueeze(0)
        test_set.append(img)

    return test_set, test_imgs
