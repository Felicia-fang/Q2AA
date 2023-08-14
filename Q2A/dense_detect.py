import os
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


num_classes = 80  # 分类数量
batch_size = 2
num_epochs = 10  # 训练轮次
lr = 0.02
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取并展示图片

file_root = "/home/Wufang/FYP/Q2A/encoder/data/assistq/train"
classes = ['aircon_utr3b','airfryer_gye82','airfryer_pe2j7','airfryer_w9rzm','bicycle_g8h94','blender_d4og8','blender_tg2xq','blender_zuw28','camera_4paj0','camera_9awdp','camera_a409h','coffeemachine_d2stw','dehydrator_jvzgp','diffuser_lxcd4','dryer_am5jp','dryer_d4uqs','dryer_vc1kl','inductioncooker_bjye3','inductioncooker_v0jzx','inductioncooker_veifx','kiettle_rc3pf','kitchenscale_025qs','kitchenscale_7shbw','kitchenscale_jsgih','kitchenscale_pqejy','kitchenscale_wk150','kitchentimer_fr3ld','kitchentimer_nr0vk','lightstand_ro8fj','microwave_clhpa','microwave_etrc9','microwave_gz61t','microwave_h43qm','microwave_kflra','microwave_ljbak','microwave_m0fgh','microwave_m5vq3','microwave_r5h7q','microwave_y3fpx','microwave_y693a','microwave_yw0gr','microwave_znl6u','mixer_muhce','oven_968hd','oven_e7fsy','oven_g6xvo','oven_lhap5','oven_un32d','oven_wa67l','oven_wn85g','printer_5jpry','printer_bf69n','rangehood_2pk8j','ricecooker_26ax0','ricecooker_5apek','ricecooker_af46c','ricecooker_tlvys','ricecooker_tomn0','ricecooker_zxuqy','toaster_ja9zg','toaster_xwc03','treadmill_npdev','treadmill_nu3ob','vacuum_1csuz','washingmachine_735oe','washingmachine_8fzkt','washingmachine_dz980','washingmachine_gxblk','washingmachine_h4n8j','washingmachine_kc4eb','washingmachine_kstcf','washingmachine_m79j0','washingmachine_tyap1','washingmachine_ujs4r','washingmachine_uomyf','washingmachine_wtbih','watch_0ku25','watch_c8zhg','watch_yw2mz','waterpurifier_b2j3o']
nums = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]  # 每种类别的个数

def read_data(path):
    file_name = os.listdir(path)  # 获取所有文件的文件名称
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    # 每个类别随机抽取20%作为测试集
    train_num = [int(num * 1 / 2) for num in nums]
    test_num = [nums[i] - train_num[i] for i in range(len(nums))]

    for idx, f_name in enumerate(file_name):  # 每个类别一个idx，即以idx作为标签
        im_dirs = path + '/' + f_name+'/images'
        im_path = os.listdir(im_dirs)  # 每个不同类别图像文件夹下所有图像的名称

        index = list(range(len(im_path)))
        random.shuffle(index)  # 打乱顺序
        im_path_ = list(np.array(im_path)[index])
        test_path = im_path_[:test_num[idx]]  # 测试数据的路径
        train_path = im_path_[test_num[idx]:]  # 训练数据的路径

        for img_name in train_path:
            # 会读到desktop.ini,要去掉
            if img_name == 'desktop.ini':
                continue
            img = Image.open(im_dirs + '/' + img_name).convert("RGB")  # img shape: (120, 85, 3) 高、宽、通道
            # 对图片进行变形
            img = img.resize((64, 64), Image.ANTIALIAS)  # 宽、高
            train_data.append(img)
            train_labels.append(idx)

        for img_name in test_path:
            # 会读到desktop.ini,要去掉
            if img_name == 'desktop.ini':
                continue
            img = Image.open(im_dirs + '/' + img_name).convert("RGB")  # img shape: (120, 85, 3) 高、宽、通道
            # 对图片进行变形
            img = img.resize((64, 64), Image.ANTIALIAS)  # 宽、高
            test_data.append(img)
            test_labels.append(idx)

    print('训练集大小：', len(train_data), ' 测试集大小：', len(test_data))

    return train_data, train_labels, test_data, test_labels

# 一次性读取全部的数据
train_data, train_labels, test_data, test_labels = read_data(file_root)
#%%
transform = transforms.Compose(
    [transforms.ToTensor(),  # 变为tensor
     # 对数据按通道进行标准化，即减去均值，再除以方差, [0-1]->[-1,1]
     transforms.Normalize(mean=[0.4686, 0.4853, 0.5193], std=[0.1720, 0.1863, 0.2175])
     ]
)


# 自定义Dataset类实现每次取出图片，将PIL转换为Tensor
class MyDataset(Dataset):
    def __init__(self, data, label, trans):
        self.len = len(data)
        self.data = data
        self.label = label
        self.trans = trans

    def __getitem__(self, index):  # 根据索引返回数据和对应的标签
        return self.trans(self.data[index]), self.label[index]

    def __len__(self):
        return self.len


# 调用自己创建的Dataset
train_dataset = MyDataset(train_data, train_labels, transform)
test_dataset = MyDataset(test_data, test_labels, transform)

# 生成data loader
train_iter = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
test_iter = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)




import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision.models import densenet121

# 设置全局参数
modellr = 1e-4
BATCH_SIZE =  2
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])

class MyDataset(Dataset):
    def __init__(self, data, label, trans):
        self.len = len(data)
        self.data = data
        self.label = label
        self.trans = trans

    def __getitem__(self, index):  # 根据索引返回数据和对应的标签
        return self.trans(self.data[index]), self.label[index]

    def __len__(self):
        return self.len


# 调用自己创建的Dataset
train_dataset = MyDataset(train_data, train_labels, transform)
test_dataset = MyDataset(test_data, test_labels, transform)

# 生成data loader
train_iter = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
test_iter = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# 导入数据
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
from torchvision.models import densenet121
from collections import OrderedDict
# 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
model_ft = densenet121(pretrained=False)
pthfile = r'/home/Wufang/FYP/Q2A/densenet121-a639ec97.pth'
# model_ft.load_state_dict(torch.load(pthfile))
state_dict =torch.load(pthfile)
new_state_dict = OrderedDict()
# 修改 key
for k, v in state_dict.items():
    if 'denseblock' in k:
        param = k.split(".")
        k = ".".join(param[:-3] + [param[-3]+param[-2]] + [param[-1]])
    new_state_dict[k] = v
    model_ft.load_state_dict(new_state_dict)

num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, 80)
model_ft.to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model_ft.parameters(), lr=modellr)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))


# 验证过程
def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))


# 训练

for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model_ft, DEVICE, train_iter, optimizer, epoch)
    val(model_ft, DEVICE, test_iter)
torch.save(model_ft, 'model1.pth')