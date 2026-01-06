# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang


import os

import numpy as np
from collections import defaultdict

import torch
from torchvision import datasets
import torchvision.transforms as transforms
        

def get_global_data(dataset):
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        global_data = datasets.MNIST(root='./dataset/local_dataset/', train=False, download=True, transform=transform)
    elif dataset == 'fashionmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        global_data = datasets.FashionMNIST(root='./dataset/local_dataset/', train=False, download=True, transform=transform)
    elif dataset == 'emnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        global_data = datasets.EMNIST(root='./dataset/local_dataset/', train=False, split="byclass", download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        global_data = datasets.CIFAR10(root='./dataset/local_dataset/', train=False, download=True, transform=transform)
    elif dataset == 'cifar100':
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        global_data = datasets.CIFAR100(root='./dataset/local_dataset/', train=False, download=True, transform=transform)

    return global_data


def get_data(dataset, idx, args, is_train=True):
    if is_train:
        if args.partition == 'pat':
            data_dir = os.path.join('./dataset/formatdata/', dataset, str(args.num_clients), args.niid, args.partition, args.balance, "train/")
        else:
            data_dir = os.path.join('./dataset/formatdata/', dataset, str(args.num_clients), args.niid, args.partition, args.balance, "train_{:}/".format(args.dir_alpha))
    else:
        if args.partition == 'pat':
            data_dir = os.path.join('./dataset/formatdata/', dataset, str(args.num_clients), args.niid, args.partition, args.balance, "test/")
        else:
            data_dir = os.path.join('./dataset/formatdata/', dataset, str(args.num_clients), args.niid, args.partition, args.balance, "test_{:}/".format(args.dir_alpha))

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def get_client_data(dataset, idx, args, is_train=True, few_shot=0):
    data = get_data(dataset, idx, args, is_train)

    data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list


def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]



if __name__ == "__main__":
    dataset = get_global_data(dataset ='cifar10')
    print(len(dataset.targets))


