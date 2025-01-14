import torch
import torch.nn as nn
import torch.nn.functional as F
from options import *

#for clip
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = FlattenLayer()
        self.linear = nn.Linear(512*block.expansion, args.num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], args = args)


def conv_block(in_ch, out_ch, pool=False):
    '''Represents single convolution block'''
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(out_ch),
              nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()
        # in 32*32*3
        self.conv1 = conv_block(3, 64)  # 32*32*64
        self.conv2 = conv_block(64, 128, True)  # 16*16*128

        self.res1 = nn.Sequential(conv_block(128, 128), # 16*16*128
                                  conv_block(128, 128))
        self.conv3 = conv_block(128, 256, True)  # 8*8*256
        self.conv4 = conv_block(256, 512, True)  # 4*4*512
        self.res2 = nn.Sequential(conv_block(512, 512), # 4*4*512
                                  conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),  # 1*1*512
                                        nn.Flatten(),  # 512
                                        # nn.Dropout(0.2),  # removing 20% of activations
                                        nn.Linear(512, 10)
                                        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


import clip


class ClipModelat(object):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, imgadpy=True, freezepy=True):
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.imgadpy = imgadpy
        self.freezepy = freezepy
        self.device = device

    def get_image_features(image, model, cpreprocess, device='cuda', need_preprocess=False):
        if need_preprocess:
            image = cpreprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        return image_features

    def freeze_param(model):
        for name, param in model.named_parameters():
            param.requires_grad = False

    def initdgatal(self, dataloader):

        for batch in dataloader:
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = self.get_image_features(
                image, self.model, self.preprocess)
            break
        if self.freezepy:
            self.freeze_param(self.model)

        if self.imgadpy:
            self.img_adap = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Tanh(
            ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        self.labels = labels


#dinov2
from copy import deepcopy
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dinov2_vits14)
        self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, args.num_classes))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
