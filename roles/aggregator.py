
"""
Cloud server
"""
import torch
import copy
import torch.nn as nn
from options import FedAVG_model_path, FedAVG_aggregated_model_path
from torch.utils.data import DataLoader
from options import args
device = args.device
class Cloud:
    def __init__(self, clients, model, numclasses, dataloader):
        self.model = model
        self._save_model()
        self.clients = clients
        self.numclasses = numclasses
        self.test_loader = dataloader
        self.participating_clients = None
        self.aggregated_client_model = model
        self.accuracy = list() #store model accuracy history
        self.accuracy_c = list() #store model accuracy history
        self.combined_model = copy.deepcopy(model)

    def aggregate_w_concat(self, clientSelect_idxs, importance=2):
        #aggragation with class importance
        #importance: a constant 

        totalsize = 0
        samplesize = 500
        num_client = 0
        for idx in clientSelect_idxs:
            totalsize += samplesize
            num_client += 1

        global_model = {}
        last_weights =[] #stores wieghts of last layers
        last_bias = []
        posi_id = [] #stores positive indices
        input_dim = None


        model_dict = self.aggregated_client_model.state_dict()
        model_dict_c = self.combined_model.state_dict()
        print('clientSelect_idxs',clientSelect_idxs)

        for k, idx in enumerate(clientSelect_idxs):
            client = self.clients[idx] #get a client
            #input_dim = client.model.conv1.in_features
            #get the positive classes at the client
            pos_class_id = (client.indexlist == 1).nonzero(as_tuple=True)[0]
            #print('client.indexlist',  pos_class_id)
            posi_id.append(pos_class_id)
            

            weight = samplesize / totalsize
            for name, param in client.model.state_dict().items():
                #print('name', name)
                if name == 'fc2.bias':
                     
                    final_layer_bias = param.data
                    last_bias.append(final_layer_bias)
                        #for i in pos_class_id:
                         #   weighted = final_layer_weights[i,:] * importance
                        #    final_layer_weights[i,:] = weighted

                if name == 'fc2.weight':
                     
                    final_layer_weights = param.data
                    last_weights.append(final_layer_weights)
                       # for i in pos_class_id:
                        #    weighted = final_layer_weights[i,:] * importance
                       #     final_layer_weights[i,:] = weighted
                
                if k == 0:
                    global_model[name] = param.data * weight
                else:
                   global_model[name] += param.data * weight

        tmp1 =  [None] * args.num_classes
        tmp2 =  [None] * args.num_classes



        for i in range(len(last_weights)):
            id = posi_id[i]
            w = last_weights[i]
            b = last_bias[i]
            for d in id:
                if tmp1[d] == None:
                    tmp1[d] =[w[d,:]]
                    tmp2[d] = [b[d]]
                else:
                    tmp1[d].append(w[d,:])
                    tmp2[d].append(b[d])
        #print(last_weights, len(last_weights))
        
        #tmp1 = [x[0] for x in tmp1]
        #print(tmp1, len(tmp1),len(tmp1[0]),tmp1[0][0].shape)
        
        with open(r'tmp2', 'w') as fp:
                fp.write('\n'.join(str(l) for l in tmp2))
     
        
        for c in range(args.num_classes):
            t1 = tmp1[c]
            t2 = tmp2[c]
            if len(t1) > 1:
                tmp1[c] = sum(t1) / len(t1)
                tmp1[c] = [tmp1[c]]
                print(tmp1[c])
                tmp2[c] = sum(t2) / len(t2)
                tmp2[c] = [tmp2[c]]
        # Concatenate weights and biases
        #print(tmp1, len(tmp1),tmp1[0].shape)
        tmp1 = [x[0].unsqueeze(0)  for x in tmp1]
        tmp2 = [x[0].unsqueeze(0) for x in tmp2]
        print(tmp2,len(tmp2))
        #print(tmp1, len(tmp1),tmp1[9],tmp1[0]) #?

        
        combined_weights = torch.cat(tmp1)  # Shape: (num_classes_model1 + num_classes_model2, input_dim)
        print('combined_weights ', combined_weights.shape )
        combined_bias = torch.cat(tmp2)    

        # Create a new model with combined output layer
        #print('input_dim:', input_dim)
        #combined_model = nn.Sequential(nn.Linear(28, combined_weights.shape[0]))  # Create a new output layer)
        #combined_model[0].weight.data = combined_weights  # Assign the custom weights
        #combined_model[0].bias.data = combined_bias 
        combined_model = global_model
        combined_model['fc2.bias'] = combined_bias
        combined_model['fc2.weight'] = combined_weights


        pretrained_dict = {k: v for k, v in global_model.items() if k in model_dict}
        pretrained_dict_c = {k: v for k, v in combined_model.items() if k in model_dict}
      
        model_dict.update(pretrained_dict)
        model_dict_c.update(pretrained_dict)
        self.aggregated_client_model.load_state_dict(pretrained_dict)
        self.combined_model.load_state_dict(pretrained_dict_c)
        return self.aggregated_client_model

    def aggregate(self, clientSelect_idxs):
        totalsize = 0
        samplesize = 500
        for idx in clientSelect_idxs:
            totalsize += samplesize

        global_model = {}
        model_dict = self.aggregated_client_model.state_dict()
        print(model_dict.keys())
        for k, idx in enumerate(clientSelect_idxs):
            client = self.clients[idx]
            #print('client.indexlist', client.indexlist, (client.indexlist == 1).nonzero(as_tuple=True)[0])
            weight = samplesize / totalsize
            for name, param in client.model.state_dict().items():
                #print('name: ',name)
                if k == 0:
                    global_model[name] = param.data * weight
                else:
                    global_model[name] += param.data * weight
                    

        pretrained_dict = {k: v for k, v in global_model.items() if k in model_dict}

        self.combined_model = ImportanceEnsembleModel(self.clients, clientSelect_idxs, importance=2)
      
        model_dict.update(pretrained_dict)
        self.aggregated_client_model.load_state_dict(pretrained_dict)
        return self.aggregated_client_model



    def validation(self, cur_rounds):
        self.aggregated_client_model.eval()
        correct = 0
        correct_c = 0
        for i, (inputs, labels) in enumerate(self.test_loader):
            # print("Test input img scale:", inputs.max(), inputs.min())
            # inputs = inputs.cuda()
            inputs = inputs.to(device)
            # labels = labels.cuda()
            labels = labels.to(device)
            outputs = self.aggregated_client_model(inputs)
            outputs_c = self.combined_model(inputs)
            # pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).cuda()
            pred = outputs.data.max(1, keepdim=True)[1].view(labels.shape[0]).to(device)
            pred_c = outputs_c.data.max(1, keepdim=True)[1].view(labels.shape[0]).to(device)
            correct += (pred == labels).sum().item()
            correct_c += (pred_c == labels).sum().item()

        print('Round:{:d}, Accuracy: {:.4f} %'.format(cur_rounds, 100 * correct / len(self.test_loader.dataset)))
        print('Round:{:d}, Accuracy_c: {:.4f} %'.format(cur_rounds, 100 * correct_c / len(self.test_loader.dataset)))
        self.accuracy.append(100 * correct / len(self.test_loader.dataset))
        self.accuracy_c.append(100 * correct_c / len(self.test_loader.dataset))
        return 100 * correct / len(self.test_loader.dataset)



    def _save_model(self):
        torch.save(self.model, FedAVG_model_path)

    def _save_params(self):
        torch.save(self.model.state_dict(), FedAVG_aggregated_model_path)



class ImportanceEnsembleModel(nn.Module):
    def __init__(self, clients, clientSelect_idxs, importance=5):
        super(ImportanceEnsembleModel, self).__init__()
        self.clients = [clients[i] for i in clientSelect_idxs]
        self.importance = importance
        self.num_clients = len(clients)

    def forward(self, test_data):
        """
        Forward pass for the ensemble model.
        
        Args:
            clientSelect_idxs (list): List of indices for the selected clients.
            test_data (torch.Tensor): The input test data.
        
        Returns:
            torch.Tensor: The softmax probabilities of the ensemble prediction.
        """
        logits = []

        for c in self.clients:
            pos_class_id = (c.indexlist == 1).nonzero(as_tuple=True)[0]

            outs = c.model(test_data)
            
            # Apply importance scaling to the positive class logits
            for p in pos_class_id:
                outs[p] = self.importance * outs[p]

            logits.append(outs)

        # Compute the weighted average of the logits
        if self.num_clients > 0:
            wavg_logits = sum(logits) / (self.num_clients + self.importance - 1)
        else:
            raise ValueError("The list of selected clients is empty.")
        
        # Apply softmax to the weighted average logits
        soft_logits = nn.functional.softmax(wavg_logits, dim=1)

        return soft_logits

