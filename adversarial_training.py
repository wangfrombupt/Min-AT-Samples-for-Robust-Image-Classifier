import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch 
print('# GPUs = %d' % (torch.cuda.device_count()))

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/public/st/")
import json
import numpy as np

import torchvision.datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#from art.utils import load_cifar10
from load_data import load_cifar10
#from networks import *
import resnet
import resnet_advprop
import wide_resnet
import wideresnet34
#import wideresnet
#import vgg
#import inceptionv3
#import densenet
#import shufflenetv2
import generate_and_test
import generate_and_test_new
from functools import partial

#model_name = 'wideresnet'
model_name = 'resnet'
batch_size = 64
use_cuda = True
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

# 设置模型结构
model = resnet_advprop.ResNet34().to(device)
#model = resnet.ResNet34().to(device)
#model = wide_resnet.Wide_ResNet(28, 10, 0.3, 10).to(device)
#model = wideresnet34.WideResNet().to(device)

# model = wideresnet.WideResNet().to(device)
# 设置模型路径
pretrained_target = "/home/public/st/adversarial_robustness_toolbox_main/examples/CIFAR10_256.pth"

#pretrained_target = "/data/st/adversarial-robustness-toolbox-main/examples/cifar10_wide_resnet.pth"
#pretrained_target = "/home/public/st/adversarial-robustness-toolbox-main/models_test/150000_pgd_resnet_80_mix_updated_all_64_00005.pth"
#pretrained_target = "/home/public/st/adversarial-robustness-toolbox-main/models_test/150000_pgd_resnet_95_mix_999_64_00005.pth"

#pretrained_target = "/data/st/adversarial-robustness-toolbox-main/examples/model_cifar_wrn.pt"

# 补充数据集路径
img_dir = "/home/public/st/pytorch-wgan-master/np_data_cifar10_gan_origin_No_t/updated_g_all_100000"
#img_dir = "/data/st/pytorch-wgan-master/new_generator_cifar/new_generator_all_100000"
#attack_type = 'cw'
#attack_type = 'MIFGSM'
#attack_type = 'SparseFool'
#attack_type = 'BIM'
#attack_type = 'RFGSM'
#attack_type = 'DeepFool'
#attack_type = 'APGD'
#attack_type = 'OnePixel'
#attack_type = 'Jitter'
attack_type = 'Auto'
#attack_type = 'PGD'
#attack_type = 'PGDL2'
#attack_type = 'DIFGSM'
#attack_type = 'FAB'
#attack_type = 'pgd'
#attack_type = 'CW'
#attack_type = 'deepfool'
#attack_type = 'fgsm'
#attack_type = 'FGSM'
#attack_type = 'AA'
#attack_type = 'Wasserstein'
#attack_type = 'GeoDA'
#attack_type = 'Square'
#attack_type = 'SquareAttack'
#attack_type = 'TIFGSM'

# filenames是训练数据文件名称列表，labels是标签列表
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

def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


def train_target_model(target_model, epochs, train_dataloader, test_dataloader, dataset_size, model_name):
    train_num = 50000
    #print('train_num is ',train_num)
    
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0005)

    #generate_and_test.generate(target_model, attack_type)
    generate_and_test_new.generate(target_model, attack_type)
    #torch.save(target_model.state_dict(), "../models/{}_{}_0_conf.pth".format(attack_type, model_name))
    model.apply(to_mix_status)
    for epoch in range(epochs):
        loss_epoch = 0

        for i, data in enumerate(train_dataloader, 0):

            train_imgs, train_labels = data
            #print("ori images:", train_imgs.shape)
            #adv_imgs = generate_and_test_new.generate_and_return(train_imgs, train_labels, target_model, attack_type)
            adv_imgs = generate_and_test.generate_and_return(train_imgs, train_labels, target_model, attack_type)
            # train_imgs = torch.from_numpy(train_imgs)
            #print("ori batch:",train_imgs.shape)
            train_imgs = torch.cat([train_imgs, adv_imgs], dim=0)
            #print("after mix:", train_imgs.shape)
            train_labels = torch.cat([train_labels, train_labels], dim=0)
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
           #print(train_labels)

            logits_model = target_model(train_imgs)
            criterion = F.cross_entropy(logits_model, train_labels)
            loss_epoch += criterion
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

        print('Loss in epoch {}: {}'.format(epoch, loss_epoch.item()))

        # 每隔5个epoch，使用测试集检测成功率
        if epoch % 5 == 0 or epoch < 5:
            generate_and_test.generate(target_model, attack_type)
            #generate_and_test_new.generate(target_model, attack_type)
            #torch.save(target_model.state_dict(), "../models_test/{}_{}_{}_{}_mix_update_all_64_00005.pth".format(train_num, attack_type, model_name, epoch))
            torch.save(target_model.state_dict(), "../models/{}_{}_{}_{}_pgd20_8255_00005.pth".format(train_num, attack_type, model_name, epoch))
        
    # save model
    targeted_model_file_name = '../models_test/{}_{}_{}_pgd20_8255_00005.pth'.format(train_num, attack_type, model_name)
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()
     
    #target_model.load_state_dict(torch.load("/data/st/adversarial-robustness-toolbox-main/models/10000_pgd_resnet_95.pth", map_location={'cuda:1':'cuda:2'}))
    # 测试集准确率
    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        
        pred_lab = torch.argmax(target_model(test_img), 1)
        print(pred_lab)
        print(test_label)
        n_correct += torch.sum(pred_lab == test_label,0)
    
    print(dataset_size)
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in test set: {}%\n'.format(100 * n_correct.item()/10000))

    # 训练集准确率
    n_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        train_img, train_label = data
        train_img, train_label = train_img.to(device), train_label.to(device)
        
        pred_lab = torch.argmax(target_model(train_img), 1)
        n_correct += torch.sum(pred_lab == train_label,0)
    
    print(train_num)
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in train set: {}%\n'.format(100 * n_correct.item()/50000))

#print("Load resnet model")
#model.load_state_dict(torch.load(pretrained_target, map_location='cpu'))
pre_weights = torch.load(pretrained_target, map_location={'cuda:1':'cuda:2'})
bn_weights = {}
keys = []
for key, w in pre_weights.items():  # 遍历预训练权重的有序字典
    keys.append(key)
    if 'bn1' in key:
        print("ori key:",key)
        key = key.replace('bn1','bn1.aux_bn')
        print("after replace:",key)
        bn_weights[key]=w
    if 'bn2' in key:
        print("ori key:",key)
        key = key.replace('bn2','bn2.aux_bn')
        print("after replace:",key)
        bn_weights[key]=w
        

missing_keys, unexpected_keys = model.load_state_dict(pre_weights,strict = False)
print("missing_keys",missing_keys)
print("unexpected_keys",unexpected_keys)

res = []
bn_missing_keys, bn_unexpected_keys = model.load_state_dict(bn_weights,strict = False)
for i in bn_missing_keys:
    if i not in keys:
        res.append(i)
print("bn missing_keys",res)
print("bn unexpected_keys",bn_unexpected_keys)

# bn1.aux_bn.weight bn1.weight
# layer4.2.bn2.aux_bn.running_var layer4.2.bn2.running_var


#model.load_state_dict(torch.load("/data/st/adversarial-robustness-toolbox-main/models_test/150000_PGD_resnet_5_mix_gc_all_64_PGD100_8_255_0001.pth", map_location={'cuda:1':'cuda:2'}))

#==========

#print("load train dataset")
#
#x_train = []
#y_train = []
#name_to_label = {"airplane":0 , "automobile":1 ,"bird":2, "cat":3 ,"deer":4 , "dog":5, "frog":6 ,"horse":7, "ship":8 ,"truck":9}
#label_to_name = {0:"airplane" , 1:"automobile" , 2:"bird", 3:"cat" , 4:"deer" , 5:"dog", 6:"frog" , 7:"horse", 8:"ship" , 9:"truck"}
#
#files = os.listdir(img_dir)
#for file in files:
#    print(file)
#    name = file.split('_')[0]
#    data = np.load(img_dir + '/' + file).astype(np.float32)
#    label = [name_to_label[name]] * len(data)
#    x_train.extend(data)
#    y_train.extend(label)
#
##print(y_train)
#
#x_train = np.array(x_train)
##y_train = np.array(y_train)
#
##gan = 1000
##
##x0 = x_train[0:gan]
##x1 = x_train[10000:10000+gan]
##x2 = x_train[20000:20000+gan]
##x3 = x_train[30000:30000+gan]
##x4 = x_train[40000:40000+gan]
##x5 = x_train[50000:50000+gan]
##x6 = x_train[60000:60000+gan]
##x7 = x_train[70000:70000+gan]
##x8 = x_train[80000:80000+gan]
##x9 = x_train[90000:90000+gan]
##
##y0 = y_train[0:gan]
###print(y0)
##y1 = y_train[10000:10000+gan]
##y2 = y_train[20000:20000+gan]
##y3 = y_train[30000:30000+gan]
##y4 = y_train[40000:40000+gan]
##y5 = y_train[50000:50000+gan]
###print(y5)
##y6 = y_train[60000:60000+gan]
##y7 = y_train[70000:70000+gan]
##y8 = y_train[80000:80000+gan]
##y9 = y_train[90000:90000+gan]
##
##x_train_gan = np.vstack((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9))
##y_train_gan = np.hstack((y0,y1,y2,y3,y4,y5,y6,y7,y8,y9))
##
##print(y_train_gan.shape)
##print('gan')
##print(x_train_gan.shape)
##print(x_train[0])
#
##
#(x_train_origin, y_train_origin), (x_test, y_test), (y_train_normal, y_test_normal), min_pixel_value, max_pixel_value = load_cifar10()
#
#x_train_origin = np.transpose(x_train_origin, (0, 3, 1, 2)).astype(np.float32)
##x_train_ori = x_train_origin[0:40000]
##y_train_ori = np.array(y_train_normal)
##y_train_ori = y_train_ori[0:40000]
##print(y_train_ori.shape)
###print('origin')
##print(x_train_ori.shape)
##print(x_train_origin[0])
#x_train = np.vstack((x_train_origin, x_train))
#y_train = np.array(y_train_normal + y_train)
##x_train_ori = x_train_origin
##y_train = np.array(y_train_normal)
#
#
#
##print('origin')
##print(x_train_origin.shape)
##print(x_train_origin[0])
##x_train = np.vstack((x_train_gan, x_train_ori))
##y_train = np.hstack((y_train_gan, y_train_ori))
## x_train = x_train_origin
## y_train = np.array(y_train_normal)
#
#print(x_train.shape)
#print(y_train.shape)
#
#
#x_train_new = x_train
#y_train_new = y_train
#
#print(x_train_new.shape)
#print(y_train_new.shape)

#===============

trans = transforms.Compose([
    #transforms.Resize(224),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,)),
])
#train_dataset = MyDataset(data=x_train_new, labels=y_train_new, transform=trans)
#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=trans, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
print("load test dataset")
test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
#print("start to train pgd attack model")
print("start to train attack model")
train_target_model(model, 90 , train_dataloader, test_dataloader, len(test_dataset), model_name)
