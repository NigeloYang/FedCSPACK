# -*- coding: utf-8 -*-            
# @Time : 2025/11/27
# @Author : Yang


import time
from tqdm import tqdm

from system.servers.serverbase import ServerBase
from system.clients.clientavg import clientAvg


class FedAvg(ServerBase):
    def __init__(self, args, metrics):
        super().__init__(args, metrics)
        
        self.set_slow_clients()
        self.set_clients(args, clientAvg)

        print(f"total clients: {len(self.clients)}")
        print("Finished creating server and clients. \n ")

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