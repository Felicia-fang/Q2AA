from model import ModelModule
import json
import base64
import torch
import model
import os
from configs import build_config
import json
import pandas as pd
import re
from PIL import Image
from flask import Flask
from flask import request
csv_model=[]
button_number = ['0']
# print(int(button_number[0]))
Path = '/home/Wufang/FYP/Q2A/encoder/data/assistq/train/'

# csv_list = pd.read_csv('/home/Wufang/FYP/Q2A/encoder/configs/data/assistq/train/ricecooker_26ax0/buttons.csv', index_col=0)
# print(csv_list)
# print(csv_list.iloc[int(button_number[0]) - 2, :])

import csv

with open('/home/Wufang/FYP/Q2A/encoder/configs/data/assistq/train/ricecooker_26ax0/buttons.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, rows in enumerate(reader):
        if i == int(button_number[0])-1:
            row = rows

csv_model.append(row)
distname = open('/home/Wufang/FYP/Q2A/button.csv', 'w')
pd_10 = pd.DataFrame(csv_model)  # 将列表转换为DataFrame格式
pd_10.to_csv(distname)