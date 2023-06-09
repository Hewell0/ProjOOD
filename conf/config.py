# config.py
import os.path
HOME = os.path.expanduser("~")

CIFAR10 = {
    'num_classes':10,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    # 'AvgPool_InputSize':8,
    'PEDCC_Type':'./center_pedcc/10_256.pkl',
    'feature_size':256,
    'batch_size':128,
    'name':'CIFAR10'
}

CIFAR100 = {
    'num_classes':100,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    # 'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/100_512.pkl',
    'feature_size': 512,
    'batch_size':128,
    'name':'CIFAR100'
}
SVHN = {
    'num_classes':10,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    # 'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/10_256.pkl',
    'feature_size':256,
    'batch_size':64,
    'name':'SVHN'
}