# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang


import datetime
import json
import os


class Metrics(object):
    def __init__(self, args):
        self.args = args
        global_epoch = self.args.global_epoch
        self.clients_rate = []
        self.per_selected_clients = {ge: [0] for ge in range(args.global_epoch)}
        self.client_train_loss = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_train_acc = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_test_acc = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_test_loss = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_train_time = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_send_time = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_en_time = {c: [] for c in range(self.args.num_clients)}
        self.client_de_time = {c: [] for c in range(self.args.num_clients)}
        self.client_pack = {c: [] for c in range(self.args.num_clients)}
        self.epoch_client_bytes = {c: [] for c in range(self.args.num_clients)}
        self.epoch_avg_bytes = []
        self.global_epoch_time = []
        self.local_avg_train_acc = []
        self.local_avg_train_loss = []
        self.local_avg_test_acc = []
        self.local_avg_test_loss = []
        self.final_avg_train_acc = []
        self.final_avg_train_loss = []
        self.final_avg_test_acc = []
        self.final_avg_test_loss = []
        self.final_global_acc = []
        self.final_global_loss = []
        self.attck_psnr_client = {c: [] for c in range(self.args.num_clients)}
        self.attck_psnr_all = []
        self.all_time = []
        self.path = './result/fedcspack/'

    def check_dir(self):
        path = os.path.join(self.path, 'noenc',str(int(self.args.ran_rate * 10)), self.args.dataset, self.args.niid, self.args.partition, self.args.balance)
        print(path)

        if not os.path.exists(path):
            os.makedirs(path)
            print('The directory is created ')
        else:
            print('The directory exists')
        return path

    def write(self):
        # 创建存储文件目录
        self.path = self.check_dir()

        '''write existing history records into a json file'''
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        metrics = {}
        metrics['algorithm'] = self.args.algorithm
        metrics['dataset'] = self.args.dataset
        metrics['model'] = self.args.model
        metrics['dataiid'] = self.args.niid
        metrics['partition'] = self.args.partition
        metrics['balance'] = self.args.balance
        metrics['dir_alpha'] = self.args.dir_alpha

        # global model setting
        metrics['global_epoch'] = self.args.global_epoch

        # local model setting
        metrics['num_clients'] = self.args.num_clients
        metrics['local_epoch'] = self.args.local_epoch
        metrics['local_batchsize'] = self.args.local_bs
        metrics['local_learn_rate'] = self.args.local_learn_rate
        metrics['per_selected_clients'] = self.per_selected_clients
        metrics['random_client_model'] = self.args.ran_client_model
        metrics['random_client_rate'] = self.args.ran_rate
        metrics['clients_rate'] = self.clients_rate

        # sparsification technology
        metrics['spars_name'] = self.args.spars_name
        metrics['pack_size'] = self.args.pack_size
        metrics['client_pack'] = self.client_pack

        # metrics time
        metrics['client_train_time'] = self.client_train_time
        metrics['client_send_time'] = self.client_send_time
        metrics['global_epoch_time'] = self.global_epoch_time

        # acc and loss
        metrics['client_train_loss'] = self.client_train_loss
        metrics['client_train_acc'] = self.client_train_acc
        metrics['client_test_acc'] = self.client_test_acc
        metrics['client_test_loss'] = self.client_test_loss
        metrics['local_avg_train_acc'] = self.local_avg_train_acc
        metrics['local_avg_train_loss'] = self.local_avg_train_loss
        metrics['local_avg_test_acc'] = self.local_avg_test_acc
        metrics['local_avg_test_loss'] = self.local_avg_test_loss
        metrics['final_avg_train_acc'] = self.final_avg_train_acc
        metrics['final_avg_train_loss'] = self.final_avg_train_loss
        metrics['final_avg_test_acc'] = self.final_avg_test_acc
        metrics['final_avg_test_loss'] = self.final_avg_test_loss
        metrics['final_global_acc'] = self.final_global_acc
        metrics['final_global_loss'] = self.final_global_loss

        metrics['all_time'] = self.all_time
        

        metrics_dir = os.path.join(self.path, '{}_{}_{}_{}_{}_{}_{}_{}.json'.format(self.args.algorithm,
                                                                        self.args.model,
                                                                        self.args.ran_client_model,
                                                                        self.args.spars_name,
                                                                        self.args.spars_type,
                                                                        self.args.pack_size,
                                                                        self.args.dir_alpha, time))

        with open(metrics_dir, 'w') as out_res:
            json.dump(metrics, out_res)


if __name__ == "__main__":
    pass
