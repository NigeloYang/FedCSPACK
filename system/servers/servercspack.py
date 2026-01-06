# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang


import time
import random
import numpy as np
from tqdm import tqdm

import torch

from system.servers.serverbase import ServerBase
from system.clients.clientcspack import clientCSPack

class FedCSPack(ServerBase):
    def __init__(self, args, metrics):
        super().__init__(args, metrics)        
        self.set_slow_clients()
        self.set_clients(args, clientCSPack)
        print(f"total clients: {self.num_clients}")
        print("Finished creating server and clients. \n ")

        self.pack_size = args.pack_size
        self.uploaded_masks = []
        self.agg_mask = []
        self.agg_model = []
    
    ############################# send model to client ##############################
    def send_models(self, epoch):
        send_time = time.time()

        for ids in self.selected_clients:
            if self.clients[ids].send_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            self.clients[ids].update_client_params(self.global_model)
            self.metrics.client_send_time[ids][epoch] = 2 * (time.time() - send_time)

    def train(self):
        for epoch in tqdm(range(self.global_epoch), desc='Processing'):
            print(f'\n--------------- Global training Round: {epoch + 1}th ------------------------')
            epoch_time = time.time()

            # server: select client, sends global model, and client updates local model
            self.selected_clients = self.select_clients_id()
            print(f'selected client: {self.selected_clients} \n')
            self.metrics.per_selected_clients[epoch] = self.selected_clients
            self.send_models(epoch)

            # evaluate model
            if epoch % self.eval_every == 0:
                print("Model is Evaluating")
                self.evaluate(epoch)

            # local iteration train
            for client_id in self.selected_clients:
                self.clients[client_id].train(client_id, epoch)

            ############################# server process / weight process / receive model ##############################
            self.server_process(epoch)

            self.metrics.global_epoch_time.append(time.time() - epoch_time)
            print("Global Training Round: {:>3} | Cost Time: {:>4.4f}".format(epoch + 1, time.time() - epoch_time))

        # all done
        print('\n--------------The model is evaluating for the last time-----------------')
        self.final_evaluate(self.global_epoch)
        self.global_generalization()

    
    ################################# AGGREGATE ##############################
    def receive_client_models(self, g_round):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * len(self.selected_clients)))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_masks = []
        for c_id in active_clients:
            try:
                client_time_cost = self.metrics.client_train_time[c_id][g_round] + self.metrics.client_send_time[c_id][g_round]
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(c_id)
                self.uploaded_weights.append(self.clients[c_id].train_samples)
                self.uploaded_models.append(self.clients[c_id].client_model)
                self.uploaded_masks.append(self.clients[c_id].client_mask)

    def aggregate_model(self):
        self.agg_mask = []
        self.agg_model = []

        agg_model = torch.zeros((int(np.ceil(self.all_params_len / self.pack_size)), self.pack_size)).to(self.device)
        
        self.agg_mask = np.sum(self.uploaded_masks, axis=0)

        for sw, m_idx, client_model in zip(self.uploaded_weights, self.uploaded_masks, self.uploaded_models):
            c_idx = 0
            for idx, val in enumerate(m_idx):
                if val > 0:
                    client_model[c_idx] = torch.multiply(client_model[c_idx], val / self.agg_mask[idx])

                    agg_model[idx] = torch.add(agg_model[idx], client_model[c_idx])
                    c_idx += 1

        flattened = agg_model.reshape(-1)

        temp_len = torch.tensor(0, dtype=torch.int32).to(self.device)
        for name_k, g_params in self.global_model.named_parameters():
            if temp_len == 0:
                tempdata = flattened[:self.layer_params_num[name_k]]
            else:
                tempdata = flattened[temp_len: temp_len + self.layer_params_num[name_k]]
            temp_len += self.layer_params_num[name_k]

            agg_client_param = torch.reshape(tempdata, self.layer_params_shape[name_k])
            g_params.data += agg_client_param.data.clone()


