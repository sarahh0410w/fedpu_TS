import numpy as np
import copy
import matplotlib.pyplot as plt
import torch

from datasets.dataSpilt import CustomImageDataset
from datasets.FMloader import DataLoader
from options import opt
from roles.client import Client
from roles.aggregator import Cloud
from datasets.dataSpilt import get_data_loaders_v0, get_default_data_transforms
from modules.fedprox import GenerateLocalEpochs
from options import args
device = args.device

class FmpuTrainer:
    def __init__(self, model_pu):
        # load data
        if 1:
            # create Clients and Aggregating Server
            local_dataloaders, local_sample_sizes, test_dataloader , indexlist, priorlist = get_data_loaders_v0()

            self.clients = [Client(_id + 1, copy.deepcopy(model_pu).to(device), local_dataloaders[_id], test_dataloader,
                                   priorlist=priorList, indexlist=indexList)
                            for _id , priorList, indexList, in zip(list(range(opt.num_clients)), priorlist, indexlist)]


        self.clientSelect_idxs = []

        self.cloud = Cloud(self.clients, model_pu, opt.num_classes, test_dataloader)
        self.communication_rounds = opt.communication_rounds
        self.current_round = 0


    def load_data(self):
        # for Fedmatch dataloader
        self.x_train, self.y_train, self.task_name = None, None, None
        self.x_valid, self.y_valid =  self.loader.get_valid()
        self.x_test, self.y_test =  self.loader.get_test()
        # self.x_test = self.loader.scale(self.x_test).transpose(0,3,1,2)
        self.x_test = self.x_test.transpose(0,3,1,2)
        self.y_test = torch.argmax(torch.from_numpy(self.y_test), -1).numpy()
        self.x_valid = self.loader.scale(self.x_valid)


    def begin_train(self):
        idc = 0.5
        acc_hist = []#store cloud acc

        for t in range (self.communication_rounds):
            self.current_round = t + 1
            self.cloud_lastmodel = self.cloud.aggregated_client_model
            self.clients_select()


            if 'SL' in opt.method:
                print("##### Full labeled setting #####")
                self.clients_train_step_SL()
            else:
                print("##### Semi-supervised setting #####")
                self.clients_train_step_SS(t)

            #if t % 3 == 0:
            self.cloud.aggregate(self.clientSelect_idxs)
            #self.cloud.aggregate_w_concat(self.clientSelect_idxs)
            self.cloud.validation(t)
            

        print('finishing...')
        print('max cloud accuracy: ', max(self.cloud.accuracy))
        print('max cloud accuracy_c: ', max(self.cloud.accuracy_c))
        #CLIENT ACCURACY
        for i in range(opt.num_clients):
            print('max accuracy of client', i, ': ', max(self.clients[i].accuracy_trace))
            
        with open(r'accuracy_server_{}_{}_dyn'.format(opt.num_clients, opt.pu_lr), 'w') as fp:
            fp.write('\n'.join(str(a) for a in self.cloud.accuracy))
        for c in self.clientSelect_idxs:

            with open(r'loss_c{}_{}_dyn'.format(c,opt.pu_lr), 'w') as fp:
                fp.write('\n'.join(str(l) for l in self.clients[c].loss_trace))
            with open(r'acc_c{}_{}_dyn'.format(c,opt.pu_lr), 'w') as fp:
                fp.write('\n'.join(str(l) for l in self.clients[c].accuracy_trace))
            with open(r'{}_client{}_max_acc_lr{}'.format(opt.dataset,c,opt.pu_lr), 'w') as fp:
                fp.write(str(max(self.clients[c].accuracy_trace)))



    def clients_select(self):
        m = max(int(opt.clientSelect_Rate * opt.num_clients), 1)
        self.clientSelect_idxs = np.random.choice(range(opt.num_clients), m, replace=False)

    def clients_train_step_SS(self,t):
        if 'FedProx' in opt.method:
            percentage = opt.percentage
            mu = opt.mu
            print(f"System heterogeneity set to {percentage}% stragglers.\n")
            print(f"Picking {len(self.clientSelect_idxs)} random clients per round.\n")
            heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=len(self.clients), max_epochs=opt.local_epochs)
            heterogenous_epoch_list = np.array(heterogenous_epoch_list)

            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                if opt.use_PULoss:
                    self.clients[idx].train_fedprox_pu(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                       globalmodel=self.cloud.aggregated_client_model)
                else:
                    self.clients[idx].train_fedprox_p(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                       globalmodel=self.cloud.aggregated_client_model)
        elif 'FedAvg' in opt.method:
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict()) #from here, used aggregated model
                if opt.use_PULoss:
                    print('use puloss')
                    self.clients[idx].train_fedavg_pu()
                    self.clients[idx].test()
                    #printing out
                elif opt.use_nnPULoss:
                    print('use_nnPULoss')
                    self.clients[idx].train_fedavg_nnpu(t)
                    self.clients[idx].test()
                
                elif opt.use_pu_teacher:
                    print('use teacher_student')
                    
                    self.clients[idx].train_fedavg_pu() #test comment
                    if t > 5:
                        self.clients[idx].train_fedavg_teacher_student(self.cloud.combined_model) #
                    
                    self.clients[idx].test()
                    

                else:
                    self.clients[idx].train_fedavg_p()


        else:
            return


    def clients_train_step_SL(self):
        if 'FedProx' in opt.method:
            percentage = opt.percentage    # 0.5  0.9
            mu = opt.mu
            print(f"System heterogeneity set to {percentage}% stragglers.\n")
            print(f"Picking {len(self.clientSelect_idxs)} random clients per round.\n")
            heterogenous_epoch_list = GenerateLocalEpochs(percentage, size=len(self.clients), max_epochs=opt.local_epochs)
            heterogenous_epoch_list = np.array(heterogenous_epoch_list)
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                self.clients[idx].train_fedprox_p(epochs=heterogenous_epoch_list[idx], mu=mu,
                                                  globalmodel=self.cloud.aggregated_client_model)
        elif 'FedAvg' in opt.method:
            for idx in self.clientSelect_idxs:
                self.clients[idx].model.load_state_dict(self.cloud_lastmodel.state_dict())
                self.clients[idx].train_fedavg_p()
        else:
            return
       
