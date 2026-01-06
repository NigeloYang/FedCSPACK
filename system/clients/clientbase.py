# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang


import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class ClientBase(object):
    def __init__(self, args, id, train_datas, test_datas, metrics, **kwargs):
        self.args = args
        self.id = id  # integer
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_epoch = args.global_epoch
        self.model_name = args.model_name
        self.model_type = args.model_type

        self.train_samples = len(train_datas)
        self.test_samples = len(test_datas)

        # local init
        self.local_model = copy.deepcopy(args.model)
        self.pre_global_model = copy.deepcopy(args.model)
        self.delta_model = copy.deepcopy(args.model)

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']

        # Set optimizer for the local updates
        self.local_epoch = args.local_epoch
        self.learn_rate = args.local_learn_rate
        self.local_bs = args.local_bs
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learn_rate)
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        # sparsification
        self.spars_name = args.spars_name

        # metrics
        self.metrics = metrics

        # others
        self.device = args.device

        # get client dataset
        self.local_trainloader, self.local_testloader = self.load_client_dataset(train_datas, test_datas)

        # save model parameters shape
        self.all_params_len, self.layer_params_shape, self.layer_params_num = self.save_model_shape(
            self.local_model)

    ############################# load dataset ##############################
    def load_client_dataset(self, train_data, test_data):
        """
        Returns train, test dataloaders for a given dataset and user indexes.
        """

        trainloader = DataLoader(train_data, self.local_bs, drop_last=True, shuffle=True)
        testloader = DataLoader(test_data, self.local_bs, drop_last=True, shuffle=True)

        return trainloader, testloader

    ############################# model shape ##############################
    def save_model_shape(self, local_model):
        model_all_params_len = 0
        layer_params_shape = {}
        layer_params_num = {}
        for k, param in local_model.named_parameters():
            layer_params_shape[k] = param.shape
            layer_params_num[k] = len(param.data.reshape(-1))
            model_all_params_len += len(param.data.reshape(-1))

        return model_all_params_len, layer_params_shape, layer_params_num
    
    ############################# Model Transformation ##############################
    def weight_interpolation(self, local_train_model):
        for dm, l_t_m, c_g_m in zip(self.delta_model.parameters(), local_train_model.parameters(),
                                    self.pre_global_model.parameters()):
            dm.data = torch.sub(l_t_m.data, c_g_m.data)

    def get_flattened(self, local_model):
        lm_fla_params = torch.tensor([]).to(self.device)
        for l_data in local_model.parameters():
            lm_fla_params = torch.cat((lm_fla_params, l_data.data.reshape(-1)), dim=0)

        return lm_fla_params

    ############################# update local model ##############################
    def update_client_params(self, global_model):
        for client_m, pre_global_m, g_m in zip(self.local_model.parameters(), self.pre_global_model.parameters(),
                                               global_model.parameters()):
            client_m.data = g_m.data.clone()
            pre_global_m.data = g_m.data.clone()

    ############################# metrics ##############################
    def train_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.local_model.eval()

        size, acc = 0.0, 0.0
        losses = []
        with torch.no_grad():
            for images, labels in self.local_trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]

                output = self.local_model(images)
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()

                loss = self.loss(output, labels)
                losses.append(loss.item())

        return acc, sum(losses) / len(losses), size

    def test_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.local_model.eval()

        size, acc = 0.0, 0.0
        losses = []
        with torch.no_grad():
            for images, labels in self.local_testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]

                output = self.local_model(images)
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()

                loss = self.loss(output, labels)
                losses.append(loss.item())

        return acc, sum(losses) / len(losses), size
