import numpy as np
import os
import random
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
dir_path = "formatdata/emnist/"


# Allocate data to users
def generate_dataset(dir_path,args):
    num_clients, niid, balance, partition = args.num_clients, args.niid, args.balance, args.partition

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    if partition == 'pat':
        config_path = os.path.join(dir_path, str(num_clients), niid, partition, balance, "config.json")
        train_path = os.path.join(dir_path, str(num_clients), niid, partition, balance, "train/")
        test_path = os.path.join(dir_path, str(num_clients), niid, partition, balance, "test/")
    else:
        config_path = os.path.join(dir_path, str(num_clients), niid, partition, balance, "config_{:}.json".format(args.alpha))
        train_path = os.path.join(dir_path, str(num_clients), niid, partition, balance, "train_{:}/".format(args.alpha))
        test_path = os.path.join(dir_path, str(num_clients), niid, partition, balance, "test_{:}/".format(args.alpha))

    if check(config_path, train_path, test_path, args):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.EMNIST(
        root='local_dataset', train=True, download=True, split="byclass", transform=transform)
    testset = torchvision.datasets.EMNIST(
        root='local_dataset', train=False, download=True, split="byclass", transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_classes, args)

    train_data, test_data = split_data(X, y,args.train_ratio)

    save_file(config_path, train_path, test_path, train_data, test_data, num_classes, statistic, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--niid', type=str, default="iid", help="data distribution is iid/noniid")
    parser.add_argument('--partition', type=str, default="pat", help="data distribution hete(pat/dir/exdir)")
    parser.add_argument('--balance', type=str, default="unbalance", help="data distribution balance")

    parser.add_argument("--num_clients", type=int, default=20, help="number of clients")
    parser.add_argument("--class_per_client", type=int, default=10, help="class number of clients")

    parser.add_argument("--batch_size", type=int, default=16, help="number of clients")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="merge original training set and test set, then split it manually. ")
    parser.add_argument("--alpha", type=float, default=1.0, help="for Dirichlet distribution. 100 for exdir")

    args = parser.parse_args()

    generate_dataset(dir_path, args)