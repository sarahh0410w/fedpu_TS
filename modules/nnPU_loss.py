import torch
import torch.nn as nn
import torch.nn.functional as F
from options import opt
from options import args
from modules.focal_loss import *
from numpy import loadtxt
import numpy as np

device = args.device

class Test(nn.Module):
    def __init__(self, k, puW, posW = opt.positiveRate, gamma=1, beta=0, nnPU=True):
        super().__init__()
        self.numClass = torch.tensor(k).to(device)
        self.puW = puW
        #adapted from nnPULoss
        if not 0 < posW < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.posW = posW #positive weight
        self.gamma = gamma
        self.beta = beta
        #self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)

    def forward(self, outputs, labels, priorlist, indexlist, kp, kn):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)
        new_P_indexlist = indexlist

        torch.zeros(self.numClass).to(device)
        ones = torch.ones(self.numClass).to(device)
        uList = torch.sub(ones,new_P_indexlist)
        #print('uList: ',uList)
        #eps = 1e-6

        # P U data

        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device) #can be nan
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)

        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device)
        outputsU = torch.index_select(outputs, 0, U_mask)
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)
        #print('U_mask', U_mask)
        #print('outputs_Soft: ', outputs_Soft)


        PULoss = torch.zeros(1).to(device)

        # pu3: unlabeled not classified into positive at the client =

        pu3 = -(torch.log(1 - outputsU_Soft) * new_P_indexlist).sum() / \
                max(1, outputsU.size(0)) / len(indexlist)




       # print('outputsU_soft: ',outputsU_Soft )
        #print('indexlist: ', indexlist)
        #print('new_P_indexlist: ',new_P_indexlist)
        # tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:1')
        #new_P_indexlist:  tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], d
        PULoss += pu3
        if self.numClass > len(indexlist):
            #print("numClass: ",self.numClass)
            #positive not classifed into positive
            pu1 = (-torch.log(1 - outputsP_Soft) * new_P_indexlist).sum() * \
                 priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass-len(indexlist))
            PULoss += pu1
            print('pu1') #problem: never get because numclass == length

        label_onehot_P = torch.zeros(labelsP.size(0), self.numClass*2).to(device).scatter_(1, torch.unsqueeze(labelsP,1), 1)[:, :self.numClass]
        log_res = -torch.log(torch.exp(1 - outputsP_Soft * label_onehot_P))
        pu2 = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))
        PULoss += pu2

        #added for
        #checking condi
        #confi_thres = 0.6
        #unlabeled classfied into positive at client
        #pos_Usoft = (outputsU_Soft * new_P_indexlist).cpu().detach().numpy()

        #print('labels: ',labels)
        #print('index: ', indexlist)
        #pList_soft_sum = 0
        #uList_soft_sum = 0

        #for i in range(len(pos_Usoft)):
         #   if np.any(pos_Usoft[i] >= confi_thres):
                #pList_soft.append(outputsU_Soft[i])
          #      temp = -torch.log(outputsU_Soft[i] * new_P_indexlist).sum()
           #     pList_soft_sum += temp

            #else:
                #print(outputsU_Soft[i])
                #uList_soft.append(outputsU_Soft[i])
             #   temp = -torch.log(outputsU_Soft[i] * uList).sum()
              #  uList_soft_sum += temp

        #pu_unlabel = pList_soft_sum + 0.5 * uList_soft_sum

        # pu4: unlabeled classified into positive at the client =

        #pu4 = -(torch.log(torch.exp(outputsU_Soft)) * new_P_indexlist).sum() / \
         #     max(1, outputsU.size(0)) / len(indexlist)
        #print('pu4: ',pu4)
        #for variance
        #tmp = torch.log(torch.exp(outputsU_Soft)) * uList
        #print('tmp: ',tmp)
        #var = torch.var(tmp,dim=1,keepdim=True)

        #varSum = var.sum()
        #print('var: ', varSum)




        #add focal loss for positive classses#####
        if opt.dataset == 'chest':
            l = loadtxt('chest_weights.txt').astype(np.float32)
            l = torch.from_numpy(l).to(device)



        if opt.dataset == 'isic':
            l = loadtxt('weights.txt').astype(np.float32)
            l = torch.from_numpy(l).to(device)

        #f_loss = FocalLoss(alpha=l,gamma=5)
        #crossloss = f_loss(outputsP, labelsP)

        ###
        crossloss = 0

        if opt.dataset == 'isic' or opt.dataset == 'chest':
            if outputsP.size()[0] > 0:
                crossentropyloss = nn.CrossEntropyLoss(weight=l)
                crossloss = crossentropyloss(outputsP, labelsP)

        else:
            if outputsP.size()[0] > 0:
                crossentropyloss = nn.CrossEntropyLoss()  # cause of 'nan'
                crossloss = crossentropyloss(outputsP, labelsP)


        objective =  PULoss  + kp * (crossloss)
        #print('pu3: ', pu3)

        #if (pu3 ) < 0 and self.nnPU:
         #   objective = - kn * (pu3 )




        return objective, PULoss * self.puW, crossloss

class nnMPULoss(nn.Module):
    def __init__(self, k, puW, posW = opt.positiveRate, gamma=1, beta=0, nnPU=True):
        super().__init__()
        self.numClass = torch.tensor(k).to(device)
        self.puW = puW

    def forward(self, outputs, labels, priorlist, indexlist, kp, kn):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)  # in range [0,1]
        new_P_indexlist = indexlist


        #torch.zeros(self.numClass).to(device)
        ones = torch.ones(self.numClass).to(device)
        uList = torch.sub(ones, new_P_indexlist)
        eps = 1e-6

            # P U data

        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device)  # can be nan
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)

        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device)
        outputsU = torch.index_select(outputs, 0, U_mask)
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)

        PULoss = torch.zeros(1).to(device)
            # pu3: unlabeled not classied into pisitive at the client
        pu3 = (-torch.log(1 - outputsU_Soft + eps) * new_P_indexlist).sum() / \
        max(1, outputsU.size(0))

        #pu4: unlabeled not classied into pisitive at other client(s)
        pu4 = (-torch.log(1 - outputsU_Soft + eps) * uList).sum() / \
        max(1, outputsU.size(0))


        PULoss += pu3
        if self.numClass > len(indexlist):
                # print("numClass: ",self.numClass)
                pu1 = (-torch.log(1 - outputsP_Soft + eps) * new_P_indexlist).sum() * \
                      priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass - len(indexlist))
                PULoss += pu1
                print('pu1')  # problem: never get because numclass == length

        label_onehot_P = torch.zeros(labelsP.size(0), self.numClass * 2).to(device).scatter_(1, torch.unsqueeze(
                labelsP, 1), 1)[:, :self.numClass]
        log_res = -torch.log(1 - outputsP_Soft * label_onehot_P + eps)
        #pu2 = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))
            # print('pu2 divisor: ', max(1, outputsP.size(0)))
            # print('pu2: ', -(log_res.permute(0, 1) * priorlist).sum())
        PULoss += pu4 #version 1: p3 + p4




            # add focal loss for positive classses#####
        if opt.dataset == 'chest':
            l = loadtxt('chest_weights.txt').astype(np.float32)
            l = torch.from_numpy(l).to(device)

        if opt.dataset == 'isic':
            l = loadtxt('weights.txt').astype(np.float32)
            l = torch.from_numpy(l).to(device)

            # f_loss = FocalLoss(alpha=l,gamma=5)
            # crossloss = f_loss(outputsP, labelsP)

            ###
        crossloss = 0

        if opt.dataset == 'isic' or opt.dataset == 'chest':
            if outputsP.size()[0] > 0:
                crossentropyloss = nn.CrossEntropyLoss(weight=l)  # cause of 'nan'
                crossloss = crossentropyloss(outputsP, labelsP)

        else:
            if outputsP.size()[0] > 0:
                crossentropyloss = nn.CrossEntropyLoss()  # cause of 'nan'
                crossloss = crossentropyloss(outputsP, labelsP)

        objective = PULoss * self.puW + crossloss
            # else:
            # print('this is a full unlabeled batch')
            #   objective += float("nan")

        return objective, PULoss * self.puW, crossloss

