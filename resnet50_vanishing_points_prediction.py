import torch
import torch.nn as nn
import math
from torch.nn import DataParallel
import os
from PIL import Image
import random
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.autograd import Variable


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
kwargs = {'num_workers': 1, 'pin_memory': True}
batch_size=32

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# load the data
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
                imgs.append((fn, tuple([float(v) for v in cls[:len(cls)-1]])))
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
        fn, label = self.imgs[index]  # even though the imgs is just a list, it can return the elements of is
        # in a proper way
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes

mytransform = transforms.Compose([transforms.ToTensor()])  # almost don't do any operation
train_data_root="/home/ying/data/google_streetview_train_test1"
test_data_root="/home/ying/data/google_streetview_train_test1"
data_label="/home/ying/data/google_streetview_train_test1/label.txt"
# test_label="/home/ying/data/google_streetview_train_test1/label.txt"
train_label,test_label = random_choose_data(data_label)
train_loader = torch.utils.data.DataLoader(
         myImageFloder(root=train_data_root, label=train_label, transform=mytransform),batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
         myImageFloder(root=test_data_root, label=test_label, transform=mytransform),batch_size=batch_size, shuffle=True, **kwargs)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # decrease the channel, does't change size
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # the size become 1/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # the size become 1/2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        #  block: object, planes: output channel, blocks: the num of blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # the input channel num become 4 times
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
    return model

# cnn = DataParallel(resnet50(pretrained=True))
cnn = resnet50(pretrained=False)  # load the pretrained weight to initialize the weight
#for param in cnn.parameters():
#    param.requires_grad=False
cnn.fc=nn.Linear(2048,9)
cnn.cuda()
# print(cnn)
# print(cnn.fc)
#cnn=DataParallel(cnn.cuda())
# print(cnn2)
# print(cnn2.module.fc)

criterion = nn.MSELoss().cuda()
lr = 0.001
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
print('resnet_no_finetune_predict_vanishing_points')
for epoch in (range(100)):
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
        if (i + 1) % 5 == 0:  # every 5 iteration will output a train loss
            print("Epoch [%d/%d], Iter [%d/%d] Train_Loss: %.4f" % (epoch + 1, 100, i + 1, 2343, loss.data[0]))
        # test the data
            for i, (test_images, test_labels) in enumerate(test_loader):
                    test_images = Variable(test_images.cuda())
                    test_labels = Variable(test_labels.cuda())
                    outputs = cnn(test_images)
                    loss=criterion(outputs, test_labels)
                    print("Epoch [%d/%d], Iter [%d/%d] Test_Loss: %.4f" % (epoch + 1, 100, i + 1, 781, loss.data[0]))
                    break
        # Decaying Learning Rate
    if (epoch + 1) % 20 == 0:
        lr /= 3
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)