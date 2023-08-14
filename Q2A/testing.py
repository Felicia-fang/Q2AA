import json

import torch
import model
import os
from configs import build_config
cfg = build_config()
m = model.ModelModule(cfg)
rootPath = '/home/Wufang/FYP/Q2AA/encoder/outputs/vit_b16_384_fps1/train/'
item = 'microwave_y3fpx'

def initialize():
    dic = torch.load('/home/Wufang/FYP/Q2A/outputs/q2a_gru+fps1+maskx-1_vit_b16+bert_b/lightning_logs/version_6/checkpoints/epoch=59-step=1560.ckpt',map_location=torch.device('cpu'))
    print("------------------")
    m.load_state_dict(state_dict=dict(dic['state_dict']))#加载模型参数
    m.eval()
def loadData(number_select):#加载选中的数据
    samples = []
    sample = torch.load(rootPath+item+'/qa_maskx-1.pth',map_location="cpu")
    for s in sample:
        s["video"] = os.path.join(rootPath+item+'/video.pth')
        s["script"] = os.path.join(rootPath+item+'/script.pth')
    samples.extend(sample)
    sample = samples[number_select]
    video = torch.load(sample["video"], map_location="cpu")
    script = torch.load(sample["script"], map_location="cpu")
    question = sample["question"]
    actions = sample["answers"]
    meta = {'question': sample['src_question'], 'folder': sample['folder']}
    label = None

    return video, script, question, actions, label, meta

def formatOutput(output):#加载label对应的选项
    # f  = open('/home/Wufang/FYP/Q2A/encoder/data/assistq/train.json','r')
    # qaData = json.load(f)
    # f.close()
    # itemData = qaData[item]
    # # print(itemData)
    # number_select=0
    # for selec in itemData:
    #     if selec['question']!="How to run steam option after turning on the rice cooker?":
    #         number_select+=1
    #     else:
    #         break
    ansOptions = itemData[number_select]['answers']
    score = output[0]['scores']
    print('question:' + output[0]['question'])
    counter = 0
    for i in score:
        myMaxIndex = i.index(max(i))
        print(ansOptions[counter][myMaxIndex])
        counter += 1
initialize()
f = open('/home/Wufang/FYP/Q2A/encoder/data/assistq/train.json', 'r')
qaData = json.load(f)
f.close()
itemData = qaData[item]
# print(itemData)
number_select = 0
for selec in itemData:
    if selec['question'] != "How to steam frozen foods"+"?":
        number_select += 1
    else:
        break
batch = loadData(number_select)
output = m.model([batch])#调用网络
print(output)
formatOutput(output)