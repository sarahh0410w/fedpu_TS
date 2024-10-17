
import argparse
import torch
parser = argparse.ArgumentParser()
# def args_parser():
    
# device
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
# optimizer
parser.add_argument('--pu_lr', type=float, default=0.01, help='learning rate of each client')
parser.add_argument('--adjust_lr', action='store_true', default=False,
                    help='adjust lr according to communication rounds')
parser.add_argument('--pu_batchsize', type=int, default=500, help='batchsize of dataloader')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer param')
# dataset
# parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--dataset', type=str, default='isic')
parser.add_argument('--data_root', type=str, default='./data/')
parser.add_argument('--num_classes', type=int, default=10)

# PU param
parser.add_argument('--task', type=str, default='FedPU')
parser.add_argument('--pu_weight', type=float, default=1, help='weight of puloss') #1
parser.add_argument('--local_epochs', type=int, default=20, help='epoches of each client')
parser.add_argument('--use_nnPULoss', action='store_true',
                    help='use nnPULoss')
parser.add_argument('--use_PULoss', action='store_true',
                    help='use PULoss')
parser.add_argument('--use_pu_teacher', action='store_true',
                    help='use Pu teacher student')

# pu dataloader                  
parser.add_argument('--randomIndex_num', type=int, default=2,
help='rate of positive sample')
parser.add_argument('--P_Index_accordance', action='store_true', 
                    help='the same positive class index number')
parser.add_argument('--positiveRate', type=float, default=0.2,
                    help='rate of positive sample')
parser.add_argument('--method', type=str, default='FedAvg')

# FL aggregator
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--communication_rounds', type=int, default=800)
parser.add_argument('--classes_per_client', type=int, default=2)
parser.add_argument('--clientSelect_Rate', type=float, default=0.5)
# FedProx parameters
parser.add_argument('--mu', type=float, default=0.0)
parser.add_argument('--percentage', type=float, default=0.0)
parser.add_argument('--clip', action='store_true')



#from fedclip
#parser.add_argument('--dataset', type=str, default='pacs')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--root_dir', type=str, default='../../../data/')
parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
parser.add_argument('--mode', type=str, default='FedAtImg')
parser.add_argument('--net', type=str, default='ViT-B/32',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--n_clients', type=int, default=4)
parser.add_argument('--test_envs', type=int, nargs='+', default=[])
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=0.001)
args = parser.parse_args()

    # opt, _ = parser.parse_known_args()

opt, _ = parser.parse_known_args()

FedAVG_model_path = '/home/jmw7289/fedpu/fake/mycache/model/local_model'  # '/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/linxinyang/codebase/experiment/cache/model/local_model'
FedAVG_aggregated_model_path = '/home/jmw7289/fedpu/fake/mycache/model/FedAVG_model.pth' #'/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/linxinyang/codebase/experiment/cache/model/FedAVG_model.pth'
