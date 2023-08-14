import os
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ['aircon_utr3b','airfryer_gye82','airfryer_pe2j7','airfryer_w9rzm','bicycle_g8h94','blender_d4og8','blender_tg2xq','blender_zuw28','camera_4paj0','camera_9awdp','camera_a409h','coffeemachine_d2stw','dehydrator_jvzgp','diffuser_lxcd4','dryer_am5jp','dryer_d4uqs','dryer_vc1kl','inductioncooker_bjye3','inductioncooker_v0jzx','inductioncooker_veifx','kiettle_rc3pf','kitchenscale_025qs','kitchenscale_7shbw','kitchenscale_jsgih','kitchenscale_pqejy','kitchenscale_wk150','kitchentimer_fr3ld','kitchentimer_nr0vk','lightstand_ro8fj','microwave_clhpa','microwave_etrc9','microwave_gz61t','microwave_h43qm','microwave_kflra','microwave_ljbak','microwave_m0fgh','microwave_m5vq3','microwave_r5h7q','microwave_y3fpx','microwave_y693a','microwave_yw0gr','microwave_znl6u','mixer_muhce','oven_968hd','oven_e7fsy','oven_g6xvo','oven_lhap5','oven_un32d','oven_wa67l','oven_wn85g','printer_5jpry','printer_bf69n','rangehood_2pk8j','ricecooker_26ax0','ricecooker_5apek','ricecooker_af46c','ricecooker_tlvys','ricecooker_tomn0','ricecooker_zxuqy','toaster_ja9zg','toaster_xwc03','treadmill_npdev','treadmill_nu3ob','vacuum_1csuz','washingmachine_735oe','washingmachine_8fzkt','washingmachine_dz980','washingmachine_gxblk','washingmachine_h4n8j','washingmachine_kc4eb','washingmachine_kstcf','washingmachine_m79j0','washingmachine_tyap1','washingmachine_ujs4r','washingmachine_uomyf','washingmachine_wtbih','watch_0ku25','watch_c8zhg','watch_yw2mz','waterpurifier_b2j3o']

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.4686, 0.4853, 0.5193], std=[0.1720, 0.1863, 0.2175])
     ]
)

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

class ResidualBlock(torch.nn.Module):
    def __init__(self, nin, nout, size, stride=1, shortcut=True):
        super(ResidualBlock, self).__init__()

        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(nin, nout, size, stride, padding=1),
                                          torch.nn.BatchNorm2d(nout),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.Conv2d(nout, nout, size, 1, padding=1),
                                          torch.nn.BatchNorm2d(nout))
        self.shortcut = shortcut

        self.block2 = torch.nn.Sequential(torch.nn.Conv2d(nin, nout, size, stride, 1),
                                          torch.nn.BatchNorm2d(nout))
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, input):
        x = input
        out = self.block1(x)

        if self.shortcut:
            out = x + out
        else:
            out = out + self.block2(x)
        out = self.relu(out)
        return out


#定义残差结构
class resnet(torch.nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                         torch.nn.BatchNorm2d(64),
                                         torch.nn.ReLU())

        self.d1 = self.make_layer(64, 64, 3, stride=1, t=2)
        self.d2 = self.make_layer(64, 128, 3, stride=2, t=2)
        self.d3 = self.make_layer(128, 256, 3, stride=2, t=2)
        self.d4 = self.make_layer(256, 512, 3, stride=2, t=2)

        self.avgp = torch.nn.AvgPool2d(8)
        self.exit = torch.nn.Linear(512, 80)

    def make_layer(self, in1, out1, ksize, stride, t):
        layers = []
        for i in range(0, t):
            if i == 0 and in1 != out1:
                layers.append(ResidualBlock(in1, out1, ksize, stride, None))
            else:
                layers.append(ResidualBlock(out1, out1, ksize, 1, True))
        return torch.nn.Sequential(*layers)

    def forward(self, input):
        x = self.block(input)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.avgp(x)

        x = x.squeeze()
        output = self.exit(x)
        return output

model = torch.load("./model.pt")
model.eval()
img = Image.open("/home/Wufang/FYP/Q2A/encoder/data/assistq/train/airfryer_pe2j7/images/airfryer-user.png").convert("RGB")  # img shape: (120, 85, 3) 高、宽、通道
img = img.resize((64, 64), Image.ANTIALIAS)
data=[]
data.append(img)
label=[]
label.append(1)
train_dataset = MyDataset(data, label, transform)

train_iter = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)
for data, target in train_iter:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    output = output.reshape(output.shape[0],1)
    print(output)
    prediction=torch.argmax(output)
    print(prediction)
#%%
print(classes[prediction])