# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang


import os
import copy
import time
import random
import datetime
import numpy as np


import torch
from torch.utils.data import DataLoader

from system.utils.data_utils import get_client_data, get_global_data


class ServerBase(object):
    def __init__(self, args, metrics):
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.niid = args.niid
        self.partition = args.partition
        self.balance = args.balance
        self.dir_alpha = args.dir_alpha
        self.num_clients = args.num_clients
        self.model_name = args.model_name
        self.model_type = args.model_type

        # global
        self.global_model = copy.deepcopy(args.model)
        self.global_epoch = args.global_epoch
        self.eval_every = args.eval_every

        # client parameter
        self.local_epoch = args.local_epoch
        self.global_bs = args.global_bs
        self.local_bs = args.local_bs
        self.loss = torch.nn.CrossEntropyLoss()
        self.learn_rate = args.local_learn_rate

        # Total number of clients and number of participating clients
        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        self.ran_client_model = args.ran_client_model
        self.ran_rate = args.ran_rate

        # self.times = times
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow
        self.send_slow_rate = args.send_slow
        self.time_threthold = args.time_threthold

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # sparsification
        self.spars_name = args.spars_name

        # others
        self.device = args.device
        self.few_shot = args.few_shot

        # Record parameter
        self.metrics = metrics

        self.global_dataset = get_global_data(self.dataset)

        # save model parameters shape
        self.all_params_len, self.layer_params_shape, self.layer_params_num = self.save_model_shape(
            self.global_model)

    ############################# set / select client ##############################
    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = get_client_data(self.dataset, i, self.args, is_train=True, few_shot=self.few_shot)
            test_data = get_client_data(self.dataset, i, self.args, is_train=False, few_shot=self.few_shot)
            client = clientObj(args, id=i,
                               train_datas=train_data,
                               test_datas=test_data, 
                               metrics=self.metrics,
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)
    
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients
    
    def select_clients_id(self):
        if self.ran_client_model == 'test':
            return [0, 1, 2]
        elif self.ran_client_model == 'random':
            return [i for i in range(self.num_clients) if np.random.random() <= self.ran_rate]
        elif self.ran_client_model == 'fix':
            fix_len = int(self.num_clients * self.ran_rate)
            return [i for i in range(self.num_clients)][:fix_len]
        elif self.ran_client_model == 'all':
            return [i for i in range(self.num_clients)]
        else:
            raise "random client model. [test,random,fix,all]"
        
    ############################# send model to client ##############################
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = get_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = get_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
    
    ############################# send model to client ##############################
    def send_models(self, epoch):
        for ids in self.selected_clients:
            send_time = time.time()
            
            if self.clients[ids].send_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.clients[ids].update_client_params(self.global_model)

            # save send model time cost
            self.metrics.client_send_time[ids][epoch] = 2 * (time.time() - send_time)


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


    ################################# Server AGGREGATE/UPDATE ##############################
    def server_process(self,g_round):
        self.receive_client_models(g_round)
        self.aggregate_model()
    
    def receive_client_models(self, g_round):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * len(self.selected_clients)))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        for c_id in active_clients:
            try:
                client_time_cost = self.metrics.client_train_time[c_id][g_round] + self.metrics.client_send_time[c_id][g_round]
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(c_id)
                self.uploaded_weights.append(self.clients[c_id].train_samples)
                self.uploaded_models.append(self.clients[c_id].delta_model)
                
    def aggregate_model(self):
        client_models = self.uploaded_models
        assert (len(client_models) > 0)

        total_weights = sum(self.uploaded_weights)
        agg_model = copy.deepcopy(client_models[0])
        for params in agg_model.parameters():
            params.data.zero_()

        for sw, client_model in zip(self.uploaded_weights, client_models):
            for a_m, c_m in zip(agg_model.parameters(), client_model.parameters()):
                a_m.data += (c_m.data.clone() * sw) / total_weights

        self.update_global_params(agg_model)

    def update_global_params(self, agg_client_model):
        for global_m, agg_client_m in zip(self.global_model.parameters(), agg_client_model.parameters()):
            global_m.data += agg_client_m.data.clone()


    ############################# metrics / evaluate ##############################
    def test_metrics(self, epoch):
        tot_correct = []
        tot_losses = []
        num_samples = []
        for client_id in self.selected_clients:
            correct, loss, size = self.clients[client_id].test_metrics()
            tot_correct.append((correct * 1.0) / size)
            # tot_correct.append(correct)
            tot_losses.append(loss)
            num_samples.append(size)
            if epoch < self.global_epoch:
                self.metrics.client_test_acc[client_id][epoch] = correct / size
                self.metrics.client_test_loss[client_id][epoch] = loss

        return num_samples, tot_correct, tot_losses

    def train_metrics(self, epoch):
        num_samples = []
        tot_correct = []
        tot_losses = []
        for client_id in self.selected_clients:
            correct, loss, size = self.clients[client_id].train_metrics()
            tot_correct.append((correct * 1.0) / size)
            # tot_correct.append(correct)
            tot_losses.append(loss)
            num_samples.append(size)
            if epoch < self.global_epoch:
                self.metrics.client_train_acc[client_id][epoch] = correct / size
                self.metrics.client_train_loss[client_id][epoch] = loss

        return num_samples, tot_correct, tot_losses

    def evaluate(self, epoch):
        evaluate_time = time.time()
        stats_test = self.test_metrics(epoch)
        stats_train = self.train_metrics(epoch)

        test_acc = sum(stats_test[1]) * 1.0 / len(stats_test[0])
        test_loss = sum(stats_test[2]) * 1.0 / len(stats_test[0])

        train_acc = sum(stats_train[1]) * 1.0 / len(stats_train[0])
        train_loss = sum(stats_train[2]) * 1.0 / len(stats_train[0])

        self.metrics.local_avg_train_acc.append(train_acc)
        self.metrics.local_avg_train_loss.append(train_loss)
        self.metrics.local_avg_test_acc.append(test_acc)
        self.metrics.local_avg_test_loss.append(test_loss)

        print("At Global Round {} Evaluate Model time Cost: {:.4f}".format(epoch + 1, time.time() - evaluate_time))
        print("At Global Round {} Averaged Train Acc: {:.4f}".format(epoch + 1, train_acc))
        print("At Global Round {} Averaged Train Loss: {:.4f}".format(epoch + 1, train_loss))
        print("At Global Round {} Averaged Test Acc: {:.4f}".format(epoch + 1, test_acc))
        print("At Global Round {} Averaged Test Loss: {:.4f}".format(epoch + 1, test_loss))

    def final_evaluate(self, epoch):
        stats_test = self.test_metrics(epoch)
        stats_train = self.train_metrics(epoch)

        test_acc = sum(stats_test[1]) * 1.0 / len(stats_test[0])
        test_loss = sum(stats_test[2]) * 1.0 / len(stats_test[0])

        train_acc = sum(stats_train[1]) * 1.0 / len(stats_train[0])
        train_loss = sum(stats_train[2]) * 1.0 / len(stats_train[0])

        self.metrics.final_avg_train_acc.append(train_acc)
        self.metrics.final_avg_train_loss.append(train_loss)
        self.metrics.final_avg_test_acc.append(test_acc)
        self.metrics.final_avg_test_loss.append(test_loss)

        print("Final Model Evaluation Train Acc: {:.4f}".format(train_acc))
        print("Final Model Evaluation Train Loss: {:.4f}".format(train_loss))
        print("Final Model Evaluation Test Acc: {:.4f}".format(test_acc))
        print("Final Model Evaluation Test Loss: {:.4f}".format(test_loss))

    def global_generalization(self):
        self.global_model.eval()
        test_loader = DataLoader(self.global_dataset, batch_size=self.global_bs, shuffle=True)
        size, acc = 0.0, 0.0
        losses = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]

                output = self.global_model(images)
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()

                loss = self.loss(output, labels)
                losses.append(loss.item())

        self.metrics.final_global_acc.append(acc/size)
        self.metrics.final_global_loss.append(sum(losses)/len(losses))

        print('\n global Generalization Test| Acc: {:.4f} | Loss:{:.4f}'.format(acc/size, sum(losses)/len(losses)))

    ############################# save model ##############################
    def save_global_model(self):
        model_path = os.path.join("./models", str(int(self.ran_rate * 10)), self.dataset, self.args.niid, self.args.partition, self.args.balance)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if self.algorithm == 'FedCSPACK':
            time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            dir_str = str(self.dir_alpha) if int(self.dir_alpha * 10) > 0 else 'no' 
            models_path = os.path.join(model_path, "{}_{}_{}_{}_{}_server.pt".format(self.algorithm, self.spars_name,
                                      str(self.pack_size),dir_str,time))
        else:
            models_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, models_path)