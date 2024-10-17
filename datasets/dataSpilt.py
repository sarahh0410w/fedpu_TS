import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from options import opt
import json
from chestXray_loader import *
from isic_loader import *
from options import args
import collections
from clip_models import *
from PIL import Image
import torchvision.transforms.functional as TF

device = args.device
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# -------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS
# -------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        #if self.inputs
        return self.inputs.shape[0]


class CustomImageDataset_P(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, preprocess=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.preprocess = preprocess

    def __getitem__(self, index):
        #print('index in dataset', index)
        img, label = self.inputs[index], self.labels[index]
        #print('img size: ', img.shape,label)
        #convert image to PIL
        img = TF.to_pil_image(img)

        if self.preprocess is not None:
            img = self.preprocess(img)
        return img, label

    def __len__(self):
        #if self.inputs
        return self.inputs.shape[0]

def get_MNIST():
    print('getting MNIST...')
    dataset_train = datasets.MNIST(root=opt.data_root, train=True, download=True,
                                       transform = get_default_data_transforms(opt.dataset, verbose=False)[0])
    dataset_test = datasets.MNIST(root=opt.data_root, train=False, download=True,
                                      transform = get_default_data_transforms(opt.dataset, verbose=False)[1])
    return dataset_train.train_data.numpy().reshape(-1,1,28,28), dataset_train.train_labels.numpy(), dataset_test.test_data.numpy().reshape(-1,1,28,28), dataset_test.test_labels.numpy()


def get_CIFAR10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = datasets.CIFAR10(root=opt.data_root, train=True, download=True)
    data_test = datasets.CIFAR10(root=opt.data_root, train=False, download=True)
    #
    # x_train, y_train = data_train.train_data.transpose((0, 3, 1, 2)), np.array(data_train.train_labels)
    # x_test, y_test = data_test.test_data.transpose((0, 3, 1, 2)), np.array(data_test.test_labels)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    print('y_train:',y_train)


    return x_train, y_train, x_test, y_test


def get_default_data_transforms(name, train=True, verbose=True):
    transforms_train = {
        # 'MNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'MNIST': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224, 224)),  # Resize to match CLIP's input size
            #transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel image
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'isic':transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
        ]),
        'chest': transforms.Compose([
                   transforms.ToPILImage(),
                   transforms.Resize(256),
                   transforms.CenterCrop(224),
                   transforms.Resize((224,224),interpolation=Image.NEAREST),
                   transforms.RandomHorizontalFlip(),
                   transforms.RandomVerticalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),

        'FashionMNIST': transforms.Compose([
            transforms.ToPILImage(),
           # transforms.Resize((32, 32)),
          #  transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'CIFAR10': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224, 224)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    }
    transforms_eval = {

        'MNIST': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel image
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'isic':transforms.Compose([
             transforms.ToPILImage(),
             transforms.Resize(255),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
             
        ]),
        'chest': transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
            #transforms.Resize((224,224),interpolation=Image.NEAREST),                                       
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'FashionMNIST': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((3，32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'CIFAR10': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return transforms_train[name], transforms_eval[name]


def relabel_K(dataset_train, unlabel_dict):
    #relabel the 'unlabel_dict' in dataset
    count = 0
    for index, label in enumerate(dataset_train.labels):
        if count < len(unlabel_dict) and index == unlabel_dict[count]:
            dataset_train.labels[index] += opt.num_classes
            count += 1
    return dataset_train


def puSpilt_index(dataset, indexlist, samplesize):

    labels = dataset.labels.numpy()
    #print("split: ")
    #print("indexlist: ", indexlist)
    #print('samplesize: ',samplesize)

    labeled_size = 0
    for i in indexlist:
        labeled_size += int(samplesize[i] * opt.positiveRate)
    unlabeled_size = len(labels) - labeled_size

    #l_shard = [i for i in range(int(singleClass * pos_rate))]
    labeled = np.array([], dtype='int64')
    unlabeled = np.array([], dtype='int64')
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    priorlist = []

    # divide to unlabeled
    bias = 0
    for i in range(opt.num_classes):
        if samplesize[i] != 0:
            if i in indexlist and samplesize[i]>=40:
                labeled = np.concatenate(
                    (labeled, idxs[bias : int(bias + opt.positiveRate * samplesize[i])]), axis=0)
                bias += int(opt.positiveRate * samplesize[i])
                unlabeled = np.concatenate(
                    (unlabeled, idxs[bias : int(bias + (1-opt.positiveRate) * samplesize[i])]), axis=0)
                bias += int((1-opt.positiveRate) * samplesize[i])
                priorlist.append(samplesize[i] * (1 - opt.positiveRate) / unlabeled_size)
            else:
                unlabeled = np.concatenate((unlabeled, idxs[bias : bias + samplesize[i]]), axis=0)
                bias += samplesize[i]
                priorlist.append(samplesize[i] / unlabeled_size)
        else:
            priorlist.append(0.0)

    return labeled, unlabeled, priorlist

# -------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS
# -------------------------------------------------------------------------------------------------------
def split_image_data(data, labels, n_clients, classes_per_client, shuffle=False, verbose=True):
    '''
    Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
    different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    '''
    print('classes_per_client:',classes_per_client)
    counter = collections.Counter(labels)
    print('couter in split image data:', counter)
    n_data = len(data)
    n_labels = np.max(labels) + 1
    #print('n_labels: ', n_labels,len(labels))

    data_per_client = [n_data // n_clients] * n_clients
    print('data_per_client ',data_per_client)
    data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients
    print('data_per_client_per_class: ',data_per_client_per_class)

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        #print('label,',label, j)
        data_idcs[label] += [j]

    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    #print('data_idcs: ',len(data_idcs))
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = max(c, 0)
        while budget > 0:
            #print('data_idcs[c]',len(data_idcs[c]))
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]
        #print('labels in split: ', labels[client_idcs])

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)
    return clients_split


def get_data_loaders_v0(verbose=True):
    # for added sampler
    unlbl_dicts = []
    lbl_dicts = []

    x_train, y_train, x_test, y_test = globals()['get_' + opt.dataset]()

    transforms_train, transforms_eval = get_default_data_transforms(opt.dataset, verbose=False)
    #
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval),
                                         batch_size=opt.pu_batchsize, shuffle=True)


    split = split_image_data(x_train, y_train, n_clients=opt.num_clients,
                             classes_per_client=opt.classes_per_client,
                             verbose=verbose)
    # print('num split: ',len(split))

    train_dataset = []
    priorlist = []
    indexlist = []
    count = 0
    randomIndex_num = [4, 4, 3, 3, 2, 2, 1, 1, 1, 1]

    for i, (x, y) in enumerate(split):
        indexList = []
        # dataset = CustomImageDataset(x, y, transforms_train)
        dataset = CustomImageDataset(x, y, transforms_train)
        selectcount = [0 * 1 for i in range(opt.num_classes)]

        # 计算每一类的样本量
        samplesize = [0 * 1 for i in range(opt.num_classes)]
        # print('dataset labels: ', dataset.labels)
        # print('num')
        for l in dataset.labels:
            samplesize[l] += 1
        if opt.P_Index_accordance:  # indexlist长度一致

            for j in range(opt.randomIndex_num):
                k = 0
                while True:
                    index = (count + j + k) % opt.num_classes
                    if (i == (opt.num_clients - 1) or samplesize[index] > 40) and selectcount[
                        index] < opt.randomIndex_num \
                            and (sum(m == 0 for m in selectcount) > (
                            opt.num_classes - opt.classes_per_client) and index not in indexList):
                        indexList.append(index)
                        selectcount[index] += 1
                        break
                    elif k > opt.num_classes:
                        break
                    k += 1
        else:

            # print("i: ",i)
            ####changed index#######originally this has "out-of-index" error
            for j in range(randomIndex_num[i]):  # % opt.num_classes
                k = 0
                while True:
                    index = (count + j + k) % opt.num_classes
                    if samplesize[index] > 40 and selectcount[index] < sum(
                            randomIndex_num) / opt.num_classes and index not in indexList:
                        indexList.append(index)
                        selectcount[index] += 1
                        break
                    elif k > opt.num_classes:
                        break
                    k += 1
        label_dict, unlabel_dict, priorList = puSpilt_index(dataset, indexList, samplesize)
        priorlist.append(priorList)
        # convert to onehot for torch
        li = [0] * opt.num_classes
        for i in indexList:
            li[i] = 1
        indexlist.append(li)

        unlabel_dict = np.sort(unlabel_dict)  # dict序列排序
        # for added sampler
        unlbl_dicts.append(unlabel_dict)
        lbl_dicts.append(label_dict)
        # print('unlabel_dict: ', len(unlabel_dict))
        # print('label_dic: ', len(label_dict))

        ####for checking
        with open('label.txt', 'w') as covertf:
            covertf.write(json.dumps(label_dict.tolist()))
        with open('unlabel.txt', 'w') as f2:
            f2.write(json.dumps(unlabel_dict.tolist()))

        if 'SL' not in opt.method:
            dataset = relabel_K(dataset, unlabel_dict)  # get unlabeled data
            # dataset = save_dataset_with_batch_positive(opt.pu_batchsize,opt.positiveRate, dataset, label_dict, unlabel_dict)
        train_dataset.append(dataset)

        count += len(indexList)

    # print('n: ', n, 'dl: ',list(iter(torch.utils.data.DataLoader(train_dataset[n], batch_sampler=sampler, num_workers=0, shuffle=False))))

    stats = [x.shape for x, y in split]
    # print('stats: ', stats)
    # print('client loaders')

    # original
    # print('original')
    client_loaders = [torch.utils.data.DataLoader(data, batch_size=opt.pu_batchsize, num_workers=16, shuffle=True) for
                      data in train_dataset]

    indexlist = torch.Tensor(indexlist).to(device)
    priorlist = torch.Tensor(priorlist).to(device)

    print('index list: ', indexlist)

    return client_loaders, stats, test_loader, indexlist, priorlist
    # torch.Tensor(indexlist).cuda(), torch.Tensor(priorlist).cuda()


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))
