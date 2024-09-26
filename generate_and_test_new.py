import os,sys
from re import S
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from torchvision import datasets

import argparse
import time
import torchattacks

import cw_attack
from torch.autograd import Variable
from lib.cw import cw
#from lib.hsja import hsja

import torchvision
import logging
from art.attacks.evasion import ProjectedGradientDescent
#import Classifier
from art.estimators.classification import PyTorchClassifier
from load_data import load_cifar10
from art.utils import preprocess
#from art.resnet import ResNet34
import resnet
from cifar100_model import ResNet34
#import wide_resnet
from trans import Pad, Crop
from PIL import Image
logger = logging.getLogger()


parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default = 'cifar10', help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='/data/st/adversarial-robustness-toolbox-main/examples/datasets/', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet', help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
#parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CW | PGD')
parser.add_argument('--idx', type=int, default=0, help='index')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#name_to_label = {"airplane":0 , "automobile":1 ,"bird":2, "cat":3 ,"deer":4 , "dog":5, "frog":6 ,"horse":7, "ship":8 ,"truck":9}
#label_to_name = {0:"airplane" , 1:"automobile" , 2:"bird", 3:"cat" , 4:"deer" , 5:"dog", 6:"frog" , 7:"horse", 8:"ship" , 9:"truck"}

use_cuda = True
print('CUDA Available: ',torch.cuda.is_available())
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
batch_size = 128


class MyDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

 
    def __len__(self):
        return self.data.shape[0]
 
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        # return image, self.labels[idx]
        return x, y

# 输入干净样本 x 通过pgd攻击生成对抗样本 x_adv
def generate_and_return(data , label, model, attack_type, epsilon):
    print('Attack: ' + attack_type)
    if attack_type == 'cw':
        train_img, train_label = data.to(device), label.to(device)
        _, x_test_adv = cw_attack.cw(model, train_img.data.clone(), train_label.data.cpu(), 0.1, 'l2', max_iter=10, step_size=0.1, kappa=0.8, crop_frac=1.0, verbose=False)
        return x_test_adv
    elif attack_type == 'CW':
        attack = torchattacks.CW(model, c=2)
#        _, adv_data = cw(model, data.data.clone().cuda(), label.data.cuda(), 1.0, 'l2', crop_frac=1.0)
#        return adv_data
    elif attack_type   == 'DeepFool':
        #attack = torchattacks.DeepFool(model, steps=30, overshoot=0.02)
        attack = torchattacks.DeepFool(model, steps=20, overshoot=0.02)
    elif attack_type == 'MIFGSM':
        attack = torchattacks.MIFGSM(model, eps=0.1, alpha=0.01, steps=10, decay=1.0)
    elif attack_type == 'FGSM':
        attack = torchattacks.FGSM(model, eps=4/255)
    elif attack_type == 'PGD':
        attack = torchattacks.PGD(model, eps=8/255.0, alpha=2/255, steps=10, random_start=True)
#        attack = torchattacks.PGD(model, eps=0.1, alpha=1/255, steps=40, random_start=True)
    elif attack_type == 'PGDL2':
#        attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.1, steps=10, random_start=True)
        attack = torchattacks.PGDL2(model, eps=0.3, alpha=0.1, steps=10, random_start=True)
#    elif attack_type == 'CW':
#        _, adv_data = cw(model, data.data.clone(), target.data.cpu(), 1.0, 'l2', crop_frac=1.0)
    elif attack_type == 'RFGSM':
#        attack = torchattacks.RFGSM(model, eps=0.1, alpha=0.05, steps=1)
        attack = torchattacks.RFGSM(model, eps=0.2, alpha=64/255) 
    elif attack_type == 'FAB':
#        attack = torchattacks.FAB(model, norm='Linf', eps=0.1, steps=50, n_classes=args.num_classes)
        attack = torchattacks.FAB(model, norm='Linf', steps=10, eps=8/255, n_restarts=1, n_classes=10)
    elif attack_type == 'Auto':
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=0.1, n_classes=args.num_classes)
        data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            data, label = Variable(data), Variable(label)
        #data = torch.from_numpy(data)
        adv_data = attack(data, label)
        x_test_adv = adv_data
        #x_test_adv = torch.from_numpy(adv_data)
        #return adv_data
    elif attack_type == 'APGD':
        attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
#        attack = torchattacks.APGD(model, eps=0.1)
    elif attack_type == 'APGDT':
        attack = torchattacks.APGDT(model, norm='Linf', eps=0.1, steps=100, n_restarts=1, seed=0, eot_iter=1, rho=.75, n_classes=args.num_classes)
    elif attack_type == 'OnePixel':
        attack = torchattacks.OnePixel(model, pixels=1) 
#        attack = torchattacks.OnePixel(model, pixels=3, steps=75, popsize=400, inf_batch=args.batch_size)
    elif attack_type == 'Pixel':
        attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=100, max_iterations=50)
    elif attack_type == 'Square':
        attack = torchattacks.Square(model, norm='Linf', n_queries=5000, n_restarts=1, eps=0.1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
    elif attack_type == 'SparseFool':
        attack = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
    elif attack_type == 'HSJA':
        pass
    elif attack_type == 'TIFGSM':
        attack = torchattacks.TIFGSM(model, eps=16/255, alpha=1/255, steps=40, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=True)
    elif attack_type == 'DIFGSM':
        attack = torchattacks.DIFGSM(model, eps=16/255, alpha=1/255, steps=40, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=True)
    elif attack_type == 'Jitter':
        attack = torchattacks.Jitter(model, eps=0.3)
    elif attack_type == 'BIM':
        attack = torchattacks.BIM(model, eps=4/255, alpha=2/255, steps=10)
#        attack = torchattacks.BIM(model, eps=0.2, steps=7) 
#    else:
#        raise AssertionError('Attack {} is not supported'.format(attack_type))
    
        data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            data, label = Variable(data), Variable(label)
        data = torch.from_numpy(data)
        adv_data = attack(data, label)
        x_test_adv = adv_data
        #x_test_adv = torch.from_numpy(adv_data)
        #return adv_data

    elif attack_type == 'pgd':
        min_pixel_value, max_pixel_value = 0, 1
        # Step 2a: Define the loss function and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Step 3: Create the ART classifier

        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10
        )
        attack = ProjectedGradientDescent(classifier, targeted=False, eps=epsilon/255.0, max_iter=20, batch_size=64)
        x_test_adv = attack.generate(x=data.numpy())  
        x_test_adv = torch.from_numpy(x_test_adv)
    #print("x adv:",type(x_test_adv))
    return x_test_adv


# 分别输出干净样本的分类精度 & 对干净样本pgd攻击后生成对抗样本的分类精度
def generate(model, attack_type):
    model.eval()
    #print('Attack: ' + attack_type)
    if attack_type   == 'DeepFool':
        #attack = torchattacks.DeepFool(model, steps=30, overshoot=0.02)
        attack = torchattacks.DeepFool(model, steps=20, overshoot=0.02)
    elif attack_type == 'MIFGSM':
        attack = torchattacks.MIFGSM(model, eps=0.1, alpha=0.01, steps=10, decay=1.0)
    elif attack_type == 'FGSM':
        attack = torchattacks.FGSM(model, eps=4/255)
    elif attack_type == 'PGD':
        attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    elif attack_type == 'PGDL2':
        attack = torchattacks.PGDL2(model, eps=1, alpha=0.1, steps=10, random_start=True)
    elif attack_type == 'CW':
        pass
    elif attack_type == 'cw':
        pass
    elif attack_type == 'RFGSM':
        attack = torchattacks.RFGSM(model, eps=0.1, alpha=0.05, steps=1)
    elif attack_type == 'FAB':
#        attack = torchattacks.FAB(model, norm='Linf', eps=0.1, steps=50, n_classes=args.num_classes)
        attack = torchattacks.FAB(model, norm='Linf', steps=10, eps=8/255, n_restarts=1, n_classes=10)
    elif attack_type == 'Auto':
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=0.1, n_classes=args.num_classes)
    elif attack_type == 'APGD':
#        attack = torchattacks.APGD(model, norm='Linf', eps=0.1, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        attack = torchattacks.APGD(model, eps=0.1)
    elif attack_type == 'APGDT':
        attack = torchattacks.APGDT(model, norm='Linf', eps=0.1, steps=100, n_restarts=1, seed=0, eot_iter=1, rho=.75, n_classes=args.num_classes)
    elif attack_type == 'OnePixel':
        attack = torchattacks.OnePixel(model, pixels=3, steps=75, popsize=400, inf_batch=args.batch_size)
    elif attack_type == 'Pixel':
        attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=100, max_iterations=50)
    elif attack_type == 'Square':
        attack = torchattacks.Square(model, norm='Linf', n_queries=5000, n_restarts=1, eps=0.1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
    elif attack_type == 'SparseFool':
        attack = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
    elif attack_type == 'HSJA':
        pass
    elif attack_type == 'TIFGSM':
        attack = torchattacks.TIFGSM(model, eps=16/255, alpha=1/255, steps=40, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=True)
    elif attack_type == 'DIFGSM':
        attack = torchattacks.DIFGSM(model, eps=16/255, alpha=1/255, steps=40, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=True)
    elif attack_type == 'Jitter':
        attack = torchattacks.Jitter(model, eps=0.3)
    elif attack_type == 'BIM':
        attack = torchattacks.BIM(model, eps=4/255, alpha=2/255, steps=10)
#        attack = torchattacks.BIM(model, eps=0.2) 
    else:
        pass
        #raise AssertionError('Attack {} is not supported'.format(attack_type))
    
    print("load test dataset")
    in_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), Pad(),
                                        Crop(crop_type='random', crop_frac=0.8), ])
    test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    print("Evaluate the ART classifier on benign test examples")
    # 测试集准确率
    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(model(test_img), 1)
        n_correct += torch.sum(pred_lab == test_label,0)
    
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in test set: {}%\n'.format(100 * n_correct.item()/len(test_dataset)))

    # 测试集对抗准确率
    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        #fyf_暂时修改
        adv_img = generate_and_return(test_img, test_label, model, attack_type,8/225)
        adv_img, test_label = adv_img.to(device), test_label.to(device)
        
        pred_lab = torch.argmax(model(adv_img), 1)
        n_correct += torch.sum(pred_lab == test_label,0)
    
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in adv test set: {}%\n'.format(100 * n_correct.item()/len(test_dataset)))

    model.train()


def save_cifar10():
    (x_train, y_train), (x_test, y_test),(y_train_normal, y_test_normal), min_pixel_value, max_pixel_value = load_cifar10()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        "../../cifar10", train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        "../../cifar10", train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    ###按标签分类
    mnist_img_dic = {}
    for i in range(10):
        mnist_img_dic[i] = np.ones(shape = (0,3,32,32))
    for batch_idx, (data, target) in enumerate(train_loader):
        #if(batch_idx == 100) break;
        print("handle")
        print(batch_idx)
        for i in range(len(target)):
            #mnist_img_dic[target[i].item()].append(data[i].numpy(),axis = 0)
            #print(np.expand_dims(data[i].numpy(), axis=0).shape)
            mnist_img_dic[target[i].item()] = np.vstack([mnist_img_dic[target[i].item()],np.expand_dims(data[i].numpy(), axis=0)])
    
    for i in range(10):
        np.save('./np_data_cifar10/data_'+str(i),mnist_img_dic[i])

    # np.save('np_data_cifar10/data_{}.npy'.format(filename), np.array(saved_images))

if __name__ == "__main__":
    # execute only if run as a script
    #model = resnet.ResNet34().to(device)
    #model = ResNet34().to(device)
    model = wide_resnet.Wide_ResNet(28, 10, 0.3, 10).to(device)
    model.load_state_dict(torch.load("/data/st/adversarial-robustness-toolbox-main/examples/cifar10_wide_resnet.pth"))
    #model.load_state_dict(torch.load('../CIFAR10_256.pth'))
    # model.load_state_dict(torch.load('../models/pgd_resnet_45.pth'))
    #generate(model,'cw')
    generate(model)
    # save_cifar10()
