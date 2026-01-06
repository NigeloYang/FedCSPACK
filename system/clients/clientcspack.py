# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang


import copy
import numpy as np
import time
import random

import torch
import torch.nn.functional as F

from system.clients.clientbase import ClientBase

class clientCSPack(ClientBase):
    def __init__(self, args, id, train_datas, test_datas, metrics, **kwargs):
        super().__init__(args, id, train_datas, test_datas, metrics, **kwargs)
        self.pack_size = args.pack_size
        self.spars_type = args.spars_type

        self.client_mask = []
        self.client_model = []

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

        # Gets the encryption parameter location
        res_mask, pack_delta = self.get_pack_model()
        self.client_mask = res_mask
        self.client_model = pack_delta
        
        # save train model time cost
        self.metrics.client_train_time[client_id][global_round] = time.time() - train_time
        self.metrics.client_pack[client_id].append(len(res_mask) - res_mask.count(0))

    
    ############################# Pick encryption model parameters and indexes ##############################
    def get_pack_model(self):
        batch_num = int(np.ceil(self.all_params_len / self.pack_size))

        # get one-dimensional model parameters
        cur_lm = self.get_flattened(self.local_model)
        pre_gm = self.get_flattened(self.pre_global_model)
        cur_delta_m = self.get_flattened(self.delta_model)

        if self.all_params_len % batch_num != 0:
            padding_num = batch_num * self.pack_size - self.all_params_len
            padd_data = torch.zeros(padding_num).to(self.device)
            cur_lm = torch.cat((cur_lm, padd_data), dim=0)
            pre_gm = torch.cat((pre_gm, padd_data), dim=0)
            cur_delta_m = torch.cat((cur_delta_m, padd_data), dim=0)

        cur_delta_lm_batchs = cur_delta_m.reshape(batch_num, self.pack_size)
        res_mask = [0] * batch_num
        cur_delta = []

        if self.spars_name == 'sim':
            pre_gm_batchs = pre_gm.reshape(batch_num, self.pack_size)
            cur_lm_batchs = cur_lm.reshape(batch_num, self.pack_size)

            sim_batchs1 = []
            sim_batchs2 = []

            # cosine sim
            all_similarity1 = torch.cosine_similarity(cur_lm, pre_gm, dim=0)


            for lm_data, gm_data in zip(cur_lm_batchs, pre_gm_batchs):
                sim_batchs1.append(torch.cosine_similarity(lm_data, gm_data, dim=0))
                sim_batchs2.append(F.kl_div(lm_data.softmax(dim=-1).log(), gm_data.softmax(dim=-1), reduction='sum'))

            for idx, val in enumerate(sim_batchs1):
                if val.item() < all_similarity1.item(): 
                    if self.spars_type == 'cskd':
                        res_mask[idx] = np.abs(sim_batchs2[idx].item() + sim_batchs1[idx].item())
                    elif self.spars_type == 'cs':
                        res_mask[idx] = np.abs(sim_batchs1[idx].item())
                    elif self.spars_type == 'kd':
                        res_mask[idx] = np.abs(sim_batchs2[idx].item())
                    cur_delta.append(cur_delta_lm_batchs[idx])

            return res_mask, cur_delta
       
        elif self.spars_name == 'full':
            res_mask = [1] * batch_num
            for idx in range(batch_num):
                cur_delta.append(cur_delta_lm_batchs[idx])

            return res_mask, cur_delta
        else:
            raise 'spars_name: There are only three parameter sparsity modes. [sim, topk(avgk), randk, full]'
  
    ############################# update model ##############################
    def update_client_params(self, global_model):
        for client_m, pre_global_m, g_m in zip(self.local_model.parameters(), self.pre_global_model.parameters(),
                                                global_model.parameters()):
            client_m.data = g_m.data.clone()
            pre_global_m.data = g_m.data.clone()