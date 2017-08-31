# -*- coding: utf-8 -*-
import torch
import argparse
import os
from PIL import Image
import cv2
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.nn import DataParallel
from torch.autograd import Variable


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
kwargs = {'num_workers': 1, 'pin_memory': True}
batch_size=32


def random_choose_data(label_path):
    random.seed(1)
    file = open(label_path)
    lines = file.readlines()
    slice = random.sample(lines, 200000)
    random.shuffle(slice)
    train_label = slice[:150000]
    test_label = slice[150000:200000]
    return train_label, test_label


# def my data loader, return the data and corresponding label
def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(data.Dataset):  # Class inheritance
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        #fh = open(label)
        c = 0
        imgs = []
        class_names = ['regression']
        for line in label:  # label is a list
            cls = line.split()  # cls is a list
            fn = cls.pop(0)
            if os.path.isfile(os.path.join(root, fn)):
                imgs.append((fn, tuple([float(v) for v in cls[len(cls)-2:len(cls)-1]])))
                # access the last label
                # images is the list,and the content is the tuple, every image corresponds to a label
                # despite the label's dimension
                # we can use the append way to append the element for list
            c = c + 1
        print('the total image is',c)
        print(class_names)
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # eventhough the imgs is just a list, it can return the elements of is
        # in a proper way
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes


class AlexNet(nn.Module):

    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 256*256, for the input image size is 256*256*3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # 63*63
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 32*32
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # 32*32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 15*15
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # 15*15
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # 15*15
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 15*15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 7*7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x


cnn = AlexNet()  # a method to use the data parallel
cnn2=DataParallel(cnn.cuda())


mytransform = transforms.Compose([transforms.ToTensor()])  # almost don't do any operation
train_data_root="/home/ying/data/google_streetview_train_test1"
test_data_root="/home/ying/data/google_streetview_train_test1"
data_label="/home/ying/data/google_streetview_train_test1/label.txt"
# test_label="/home/ying/data/google_streetview_train_test1/label.txt"

train_label,test_label = random_choose_data(data_label)
train_loader = torch.utils.data.DataLoader(
         myImageFloder(root=train_data_root, label=train_label, transform=mytransform ),
         batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
         myImageFloder(root=test_data_root, label=test_label, transform=mytransform ),
         batch_size=batch_size, shuffle=True, **kwargs)


criterion = nn.MSELoss()
lr = 0.001

optimizer = torch.optim.Adam(cnn2.parameters(), lr=lr)
for epoch in (range(50)):
    for i, (images, labels) in enumerate(train_loader):
        # run all the image in the dataloader, and the data is all different
        # print(i, images, labels)
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(i)
        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Train_Loss: %.4f" % (epoch + 1, 50, i + 1, 2343, loss.data[0]))
        # test the data
            for i, (test_images, test_labels) in enumerate(test_loader):
                #if (i+1) % 500 == 0:  # test the data
                test_images = Variable(test_images.cuda())
                test_labels = Variable(test_labels.cuda())
                outputs = cnn(test_images)
                loss=criterion(outputs, test_labels)
                print("Epoch [%d/%d], Iter [%d/%d] Test_Loss: %.4f" % (epoch + 1, 50, i + 1, 781, loss.data[0]))
                break
        # Decaying Learning Rate
    if (epoch + 1) % 20 == 0:
        lr /= 3
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

