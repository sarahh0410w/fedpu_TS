import torch
import torch.nn as nn
import torch.nn.functional as F
from options import opt
from options import args
from modules.focal_loss import *
from numpy import loadtxt
import numpy as np

device = args.device
class MPULoss(nn.Module):
    def __init__(self, k, PiW, PkW, UiW, UkW):
        super().__init__()
        self.numClass = torch.tensor(k).to(device)
        self.piW = PiW
        self.pkW = PkW
        self.uiW = UiW
        self.ukW = UkW

    def forward(self, outputs, labels, prior):
        outputs = outputs.to(device).float()
        outputs_Soft = F.softmax(outputs, dim=1)
        # 数据划分
        P_mask = (labels < self.numClass - 1).nonzero().view(-1)
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels >= self.numClass - 1).nonzero().view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask)
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)

        # 计算目标
        ui = sum(-torch.log(1-outputsU_Soft[:, 0:self.numClass-1]+0.01)) / ((self.numClass - 1)*outputsU.size(0))
        uk = sum(-torch.log(outputsU_Soft[:, self.numClass-1]+0.01)) / outputsU.size(0)

        UnlabeledLossI = sum(ui)
        UnlabeledLossK = uk

        crossentropyloss=nn.CrossEntropyLoss()
        PositiveLossI = crossentropyloss(outputsP, labelsP)
        PositiveLossK = sum(-torch.log(1-outputsP_Soft[:, self.numClass-1]+0.01)) * prior

        objective = PositiveLossI*self.piW + PositiveLossK*self.pkW + UnlabeledLossI*self.uiW + UnlabeledLossK*self.ukW #w将三者统一到同一个数量级上
        # print("\n 1")
        # print(PositiveLossI*self.piW, PositiveLossK*self.pkW)
        # print(UnlabeledLossI*self.uiW, UnlabeledLossK*self.ukW)
        return objective, PositiveLossI*self.piW + PositiveLossK*self.pkW, UnlabeledLossI*self.uiW + UnlabeledLossK*self.ukW



class MPULoss_INDEX(nn.Module):
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).to(device)
        self.puW = puW

    def forward(self, outputs, labels, priorlist, indexlist):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)

        # 数据划分
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device)
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)
        # import pdb; pdb.set_trace()


        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device)
        outputsU = torch.index_select(outputs, 0, U_mask)               #  unlabeldata 的 ground truth. setting限制不能使用
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)     #  所有在 unlabeldata 上的预测值

        PULoss = torch.zeros(1).to(device)

        for i in range(self.numClass):
            if i in indexlist:      # calculate ui
                pu3 = sum(-torch.log(1 - outputsU_Soft[:, i] + 0.01)) / \
                      max(1, outputsU.size(0)) / len(indexlist)
                PULoss += pu3
            else:
                pu1 = sum(-torch.log(1 - outputsP_Soft[:, i] + 0.01)) * \
                      priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass-len(indexlist))
                PULoss += pu1

        pu2 = torch.zeros(1).to(device)
        for index, i in enumerate(labelsP):   # need to be optimized
            x = outputsP_Soft[index][i]
            pu2 += -torch.log(1 - x + 0.01) * priorlist[i]

        PULoss -= pu2 / max(1, outputsP.size(0))

        crossentropyloss=nn.CrossEntropyLoss()
        crossloss = crossentropyloss(outputsP, labelsP)

        # objective = PULoss * self.puW
        objective = crossloss
        # objective = PULoss * self.puW + crossloss

        return objective, PULoss * self.puW, crossloss




class PLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).to(device)

    def forward(self, outputs, labels):
        outputs = outputs.to(device).float()

        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
        labelsP = torch.index_select(labels, 0, P_mask).to(device)
        outputsP = torch.index_select(outputs, 0, P_mask).to(device)

        crossentropyloss=nn.CrossEntropyLoss().to(device)

        crossloss = crossentropyloss(outputsP, labelsP)
        return crossloss



class MPULoss_V2(nn.Module):
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).to(device)
        self.puW = puW

    def forward(self, outputs, labels, priorlist, indexlist):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1) #in range [0,1]
        #outputs_Soft = outputs
        #print('out soft:',outputs_Soft)
        new_P_indexlist = indexlist
        #print('indexl:', len(indexlist), indexlist)
        #print('length of index list: ',len(indexlist),indexlist[0]

        torch.zeros(self.numClass).to(device)
        eps = 1e-6

        # P U data

        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device) #can be nan
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)

        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).to(device)
        outputsU = torch.index_select(outputs, 0, U_mask)             
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)  
        #print( 'out u :', outputsU_Soft)  

        PULoss = torch.zeros(1).to(device)
        #pu3: unlabeled not classied into pisitive at the client
        pu3 = (-torch.log(1 - outputsU_Soft + eps) * new_P_indexlist).sum() / \
                              max(1, outputsU.size(0)) / len(indexlist)
        #print('pu3: ',pu3)
        #print('index list: ',  new_P_indexlist)

        #why this term? unlabeled data classified into positive class at the client???
        #print('outputsU_soft: ',outputsU_Soft )
        #print('indexlist: ', indexlist)
        #print('new_P_indexlist: ',new_P_indexlist)
        # tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:1')
        #new_P_indexlist:  tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], d
        #PULoss += pu3
        if self.numClass > len(indexlist):
            #print("numClass: ",self.numClass)
            pu1 = (-torch.log(1 - outputsP_Soft + eps) * new_P_indexlist).sum() * \
                 priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass-len(indexlist))
            PULoss += pu1
            print('pu1') #problem: never get because numclass == length

        label_onehot_P = torch.zeros(labelsP.size(0), self.numClass*2).to(device).scatter_(1, torch.unsqueeze(labelsP,1), 1)[:, :self.numClass]
        log_res = -torch.log(1 - outputsP_Soft * label_onehot_P + eps)
        pu2 = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))
        #print('pu2 divisor: ', max(1, outputsP.size(0)))
        #print('pu2: ', -(log_res.permute(0, 1) * priorlist).sum())
        PULoss += pu2
        #print('pu2: ', pu2)


        #add focal loss for positive classses#####
        if opt.dataset == 'chest':
            l = loadtxt('chest_weights.txt').astype(np.float32)
            l = torch.from_numpy(l).to(device)



        if opt.dataset == 'isic':
            l = loadtxt('weights.txt').astype(np.float32)
            l = torch.from_numpy(l).to(device)

        crossloss = 0

        if opt.dataset == 'isic' or opt.dataset == 'chest':
            if outputsP.size()[0] > 0:
                crossentropyloss = nn.CrossEntropyLoss(weight=l)  
                crossloss = crossentropyloss(outputsP, labelsP)

        else:
            if outputsP.size()[0] > 0:
                crossentropyloss = nn.CrossEntropyLoss()  
                crossloss = crossentropyloss(outputsP, labelsP)

        objective = PULoss * self.puW + crossloss

        return objective, PULoss * self.puW, crossloss
