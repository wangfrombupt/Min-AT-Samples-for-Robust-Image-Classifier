import os
import resnet
import numpy as np
import time
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
print('# GPUs = %d' % (torch.cuda.device_count()))
import torchvision

import sys
sys.path.append("/home/public/st/ParC-Net")
sys.path.append("/home/public/st/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import numpy as np

import time

import torchvision.datasets
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import resnet

import generate_and_test
import generate_and_test_new
from trans import Pad, Crop
model_name = 'resnet34'

from common import load_state_dict,Opencv2PIL,TorchMeanStdNormalize

from ParC_ConvNets.ParC_convnext import parc_convnext_xt

model = parc_convnext_xt()

use_cuda = True
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

#model_name = 'resnet'
batch_size = 64
#pretrained_target = "/home/public/st/ParC-Net/pretrained_models/classification/checkpoint_ema_avg.pt"
#pretrained_target = "/home/public/st/adversarial-robustness-toolbox-main/models_test/150000_pgd_resnet_95_mix_999_64_00005.pth"
pretrained_target = "/home/public/st/adversarial-robustness-toolbox-main/models_test/150000_pgd_resnet_3_mix_updated_all_64_00005.pth"
#ained_target = "/home/public/st/adversarial_robustness_toolbox_main/examples/CIFAR10_256.pth"
model = resnet.ResNet34().to(device)

attack_type = 'pgd'

#img_dir = "/data/st/pytorch-wgan-master/np_data_cifar10_gan_origin_No_t/50_60_1000"

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

if __name__ == '__main__':
    print('\nCHECKING FOR CUDA...')
    use_cuda = True
    print('CUDA Available: ',torch.cuda.is_available())
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    name_to_label = {"airplane":0 , "automobile":1 ,"bird":2, "cat":3 ,"deer":4 , "dog":5, "frog":6 ,"horse":7, "ship":8 ,"truck":9}
    label_to_name = {0:"airplane" , 1:"automobile" , 2:"bird", 3:"cat" , 4:"deer" , 5:"dog", 6:"frog" , 7:"horse", 8:"ship" , 9:"truck"}
              
    model.load_state_dict(torch.load(pretrained_target, map_location={'cuda:1':'cuda:2'}))

    print("load generate test dataset")
    
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,)),
    ])

    test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=trans, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    

    model.eval()
    
    
#    print('original model direct test:')
#    # 测试集准确率
#    n_correct = 0
#    for i, data in enumerate(test_dataloader, 0):
#        test_img, test_label = data
#        print(test_img.shape)
#        n_correct_cur = 0
#        test_img, test_label = test_img.to(device), test_label.to(device)
#        model = model.to(device)
#        pred_lab = torch.argmax(model(test_img), 1)
#        n_correct += torch.sum(pred_lab == test_label,0)
#        n_correct_cur = torch.sum(pred_lab == test_label,0)
#    
#        print('Correctly Classified: ', n_correct_cur.item())
#        print('Accuracy in set {}: {}%\n'.format(i, 100 * n_correct_cur.item()/1000))
#    
#    n_correct = 0
#    for i, data in enumerate(test_dataloader 0):
#        test_img, test_label = data
#        test_img, test_label = test_img.to(device), test_label.to(device)
#        model = model.to(device)
#        pred_lab = torch.argmax(model(test_img), 1)
#        n_correct += torch.sum(pred_lab == test_label,0)
#    
#    print('Correctly Classified: ', n_correct.item())
#    print('Accuracy in test ori set: {}%\n'.format(100 * n_correct.item()/10000))

    # 测试集对抗准确率
    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        adv_img = generate_and_test.generate_and_return(test_img, test_label, model, attack_type)
        adv_img, test_label = adv_img.to(device), test_label.to(device)
        n_correct_cur = 0
        pred_lab = torch.argmax(model(adv_img), 1)
        print(torch.sum(pred_lab == test_label,0))
        if torch.sum(pred_lab == test_label,0)!=torch.tensor(1):
            print(test_label)
            print(type(adv_img))
            save_image(adv_img, '/home/public/st/adversarial_robustness_toolbox_main/examples/adv_images/{}_TO_{}.png'.format(int(test_label),int(pred_lab)))
            
        n_correct_cur = torch.sum(pred_lab == test_label,0)
        n_correct += torch.sum(pred_lab == test_label,0)
    
        print('Correctly Classified: ', n_correct_cur.item())
        print('Accuracy in adv test set {}: {}%\n'.format(i, 100 * n_correct_cur.item()/1000))
    
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in adv test set: {}%\n'.format(100 * n_correct.item()/10000))

    
  
        
    
        
