# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang

import argparse
import time

import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from system.utils.result_utils import Metrics
from system.model.models import *
from system.model.resnet import *

# Traditional FL
from system.servers.serveravg import FedAvg
from system.servers.servercspack import FedCSPack


def start_train(args):
    print('--------------------- start ------------------------')
    start = time.time()
    args.model_name = args.model
    metrics = Metrics(args)

    if args.model_name == 'cnn':
        if args.dataset == "mnist":
            args.model = CNNMNIST_LIT().to(args.device)
        elif args.dataset == "fashionmnist":
            args.model = CNNMNIST_BIG().to(args.device)
        elif args.dataset == "emnist" and args.model_type == 'lit':
            args.model = CNNMNIST_BIG(num_classes=62).to(args.device)
        elif args.dataset == "emnist" and args.model_type == 'big':
            args.model = CNNEMNIST(num_classes=62).to(args.device)
        elif args.dataset == "cifar10":
            args.model = CNNCIFAR().to(args.device)
        elif args.dataset == "cifar100":
            args.model = CNNCIFAR(num_classes=100).to(args.device)
        else:
            raise 'only [mnist,fmnist, emnist, cifar10, cifar100]'
    
    elif args.model_name == 'resnet':
        if args.dataset == "cifar100" and args.model_type == 'lit':
            args.model = resnet18(num_classes=100, in_dim=3).to(args.device)
        elif args.dataset == "cifar100" and args.model_type == 'big':
            args.model = resnet34(num_classes=100, in_dim=3).to(args.device)

        else:
            raise 'only EMNIST, cifar-10, cifar-100, tinyimagenet'
    
    else:
        raise "model_name = cnn/resnet"

    print(args.model.parameters())

    # select algorithm
    if args.algorithm == 'FedAvg':
        server = FedAvg(args, metrics)  
    elif args.algorithm == 'FedCSPack':
        server = FedCSPack(args, metrics)
    else:
        print("No related algorithm is provided")


    server.train()
    server.save_global_model()

    metrics.all_time.append(time.time() - start)
    args.model = args.model_name
    metrics.write()
    print(f'\n All done! All Epoch Costs Time: {time.time() - start:.2f} \n')
    

def str2bool(str):
    return True if str.lower() == 'true' else False


if __name__ == '__main__':
    total_start = time.time()
    parser = argparse.ArgumentParser()

    # Base
    parser.add_argument('--algorithm', type=str, default='FedCSPack', help='name of training framework;')
    parser.add_argument('--model', type=str, default='cnn', help='name of model;')
    parser.add_argument('--model_type', type=str, default='big', help='type of model;')
    parser.add_argument('--dataset', type=str, default='mnist', help='name of dataset;')
    parser.add_argument('--num_classes', type=int, default=10, help='classes of dataset;')
    parser.add_argument('--niid', type=str, default="iid", help="data distribution is iid/noniid")
    parser.add_argument('--partition', type=str, default="pat", help="data distribution hete(pat/dir/exdir)")
    parser.add_argument('--balance', type=str, default="balance", help="data distribution balance")
    parser.add_argument('--dir_alpha', type=float, default=0.3, help="Dirichlet distribution ratio")

    # Global
    parser.add_argument('--global_epoch', type=int, default=10, help="number of rounds of global training")
    parser.add_argument('--global_bs', type=int, default=32, help="global batchsize")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--eval_every', type=int, default=1, help='evaluate every ____ rounds;')

    # Client
    parser.add_argument('--num_clients', type=int, default=20, help="number of users: K")
    parser.add_argument('--ran_client_model', type=str, default='fix', help="random client model. [test,random,fix,all]")
    parser.add_argument('--ran_rate', type=float, default=0.3, help="Client participation rate. random and fix")
    parser.add_argument('--local_epoch', type=int, default=3, help="number of rounds of local training")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size")
    parser.add_argument('--local_learn_rate', type=float, default=0.05, help="model learning rate")

    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="Rate for clients that train but drop out")
    parser.add_argument("--train_slow", type=float, default=0.0, help="The rate for slow clients when training locally")
    parser.add_argument("--send_slow", type=float, default=0.0, help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False, help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="The threthold for droping slow clients")
    
    
    # sparsification technology
    parser.add_argument('--spars_name', type=str, default='sim', help='parameter sparsity modes')
    parser.add_argument('--pack_size', type=int, default=1024, help='parameters pack size')
    # Fedcspack Ablation
    parser.add_argument("--spars_type", type=str, default='cskd')


    # Others parameters
    # parser.add_argument('--seed', type=int, default=0, help='seed for randomness;')
    parser.add_argument('--device', type=str, default='cuda', help="device is gpu or cpu")
    parser.add_argument('-fs', "--few_shot", type=int, default=0)


    args = parser.parse_args()
    print("=" * 100)
    if args.device == "cuda" and torch.cuda.is_available():
        print('Using Device is: '.rjust(50), args.device)
        print("Count Cuda Device: ".rjust(50), torch.cuda.device_count())
        print("Using Cuda Device index: ".rjust(50), torch.cuda.current_device())
    else:
        args.device = 'cpu'
        print('Using Device is: '.rjust(50), args.device)

    print("Algorithm: ".rjust(50), args.algorithm)
    print("Model Name: ".rjust(50), args.model)
    print("Model Type: ".rjust(50), args.model_type)
    print("Dataset: ".rjust(50), args.dataset)
    print("dataset distribution type: ".rjust(50), args.niid)
    print('data distribution partition: '.rjust(50), args.partition)
    print('data distribution balance: '.rjust(50), args.balance)
    print('Dirichlet distribution ratio: '.rjust(50), args.dir_alpha)
    print('number of users:'.rjust(50), args.num_clients)
    
    print("Global epoch: ".rjust(50), args.global_epoch)
    print("Total number of clients: ".rjust(50), args.num_clients)
    print("Client random participation probability: ".rjust(50), args.ran_rate)
    print("Local epoch: ".rjust(50), args.local_epoch)
    print("Local batch size: ".rjust(50), args.local_bs)
    print("Local learning rate: ".rjust(50), args.local_learn_rate)
    print("=" * 100)

    print("Set Random Seed: ")
    # random.seed(0 + args.seed)
    # np.random.seed(0 + args.seed)
    # torch.manual_seed(0 + args.seed)
    # torch.cuda.manual_seed(0 + args.seed)

    start_train(args)
