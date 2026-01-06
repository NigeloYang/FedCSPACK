# -*- coding: utf-8 -*-            
# @Time : 2025/11/27
# @Author : Yang


import copy
import numpy as np
import time

import torch

from system.clients.clientbase import ClientBase


class clientAvg(ClientBase):
    def __init__(self, args, id, train_datas, test_datas, metrics, **kwargs):
        super().__init__(args, id, train_datas, test_datas, metrics, **kwargs)

    ############################# model training ##############################
    def train(self, client_id, global_round):
        local_trainloader = self.local_trainloader
        self.local_model.train()

        train_time = time.time()

        for local_epoch in range(self.local_epoch):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            acc = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(local_trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                total += len(labels)
                output = self.local_model(images)
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
                
                loss = self.loss(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 10 == 0:
                    print(
                        '| Global Round: {:>3} | Client: {:>3} | Local Epoch: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f}'.format(
                            global_round + 1, client_id, local_epoch + 1,
                            100. * (batch_idx + 1) / len(local_trainloader), 100. * acc / total, loss.item()))

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # get delta_model
        self.weight_interpolation(self.local_model)

        # save train model time cost
        self.metrics.client_train_time[client_id][global_round] = time.time() - train_time

