import numpy as np
import os
import sys
import torch
import scipy.ndimage as ndi

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_transfrom import *

from scipy import linalg

from six.moves import range


class VOC2012(Dataset):
    def __init__(self,data_file,data_dir,label_dir,
                 crop_size=320,zoom_range=[0.5,2],rotation=None,channel_shift = 0.,
                 mirror=True,label_cval=255,ignore_label = 255,nb_class =21,task = 'Train'):

        #self.target_size = target_size
        self.crop_size = crop_size
        self.zoom_range = zoom_range
        self.rotation = rotation

        self.channel_shift = channel_shift
        self.mirror = mirror
        self.label_cval = label_cval
        self.ignore_label = ignore_label
        self.nb_class = nb_class
        self.task = task
        self.data_file = data_file
        self.data_dir = data_dir
        self.label_dir = label_dir
        file = open(self.data_file)
        self.data_list = file.readlines()
        file.close
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_index = self.data_list[index].strip('\n')
        img = Image.open(os.path.join(self.data_dir,data_index)+'.jpg')
        label = Image.open(os.path.join(self.label_dir,data_index)+'.png')
        img,label = self.transform(img,label)
        return img,label
    def transform(self,img,label):
        img_w, img_h = img.size
        #resize
        scale = np.random.uniform(self.zoom_range[0],self.zoom_range[1])
        aug_w = int(scale*img_w)
        aug_h = int(scale*img_h)
        img = img.resize(size=(aug_w,aug_h),resample=Image.BICUBIC)
        label = label.resize(size=(aug_w,aug_h),resample=Image.NEAREST)
        #rotation
        '''
        if self.rotation:
            theta = np.random.uniform(-self.rotation, self.rotation)
        else:
            theta = 0
        img  = img.rotate(theta,resample=Image.BICUBIC,expend=0)
        label = label.rotate(theta,resample=Image.NEAREST,expend=self.label_cval)
        '''
        img = img_to_array(img)
        img = standardlize(img)
        label = img_to_array(label)
        h = img.shape[1]
        w = img.shape[2]
        pad_h = max(self.crop_size-h,0)
        pad_w = max(self.crop_size-w,0)
        img = np.pad(img,((0,0),(pad_h/2,pad_h-pad_h/2),(pad_w/2,pad_w-pad_w/2)),'constant',constant_values=0.)
        label = np.pad(label,((0,0),(pad_h/2,pad_h-pad_h/2),(pad_w/2,pad_w-pad_w/2)),'constant',constant_values=self.label_cval).astype(int)
        if self.channel_shift != 0:
            img = random_channel_shift(img,self.channel_shift,0)
        img, label = pair_random_crop(img, label, self.crop_size)
        if self.mirror:
            if np.random.random() < 0.5:
                img = flip_axis(img,2)
                label = flip_axis(label,2)
        if self.ignore_label:
            label[np.where(label==self.ignore_label)] = -1
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        return img,label
class VOC2012Val(Dataset):
    def __init__(self,data_file,data_dir,label_dir,label_cval = 255,ignore_label = 255,nb_class = 21):
        self.data_file = data_file
        self.data_dir = data_dir
        self.label_dir = label_dir

        self.label_cval = label_cval
        self.ignore_label = ignore_label
        self.nb_class = nb_class

        file = open(self.data_file)
        self.data_list = file.readlines()
        file.close

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_index = self.data_list[index].strip('\n')
        img = Image.open(os.path.join(self.data_dir,data_index)+'.jpg')
        label = Image.open(os.path.join(self.label_dir,data_index)+'.png')

        img_w, img_h = img.size
        img = img_to_array(img)
        label = img_to_array(label).astype(int)

        pad_h = (32 - img_h % 32) % 32
        pad_w = (32 - img_w % 32) % 32

        img = np.lib.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0.)
        label = np.lib.pad(label,((0,0),(0,pad_h),(0,pad_w)),'constant',constant_values=self.label_cval)

        img = standardlize(img)
        if self.ignore_label:
            label[np.where(label == self.ignore_label)] = -1
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        '''
        img = img.float().div(256)
        img[0,:,:] = img[0,:,:].div(0.229)
        img[1,:,:] = img[1,:,:].div(0.224)
        img[2,:,:] = img[2,:,:].div(0.225)
        '''
        return img,label



