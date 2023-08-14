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
import csv
from PIL import Image
from flask import Flask
from flask import request
app = Flask(__name__)

cfg = build_config()
m = model.ModelModule(cfg)
# rootPath = '/home/Wufang/FYP/Q2A/encoder/outputs/vit_b16_384_fps1/train/'
rootPath = '/home/Wufang/FYP/Q2AA/encoder/outputs/vit_b16_384_fps1/train/'
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
    video = torch.load(sample["video"],map_location="cpu")
    script = torch.load(sample["script"],map_location="cpu")
    question = sample["question"]
    actions = sample["answers"]
    meta = {'question': sample['src_question'], 'folder': sample['folder']}
    label = None
    return video, script, question, actions, label, meta

def formatOutput(output,number_select):#加载label对应的选项
    f  = open('/home/Wufang/FYP/Q2A/encoder/data/assistq/train.json','r')
    qaData = json.load(f)
    f.close()
    itemData = qaData[item]
    ansOptions = itemData[number_select]['answers']
    score = output[0]['scores']
    print('question:' + output[0]['question'])
    counter = 0
    Output=[]
    dict = {}
    csv_model=[]
    for i in score:
        myMaxIndex = i.index(max(i))
        Output.append(ansOptions[counter][myMaxIndex])
        print('0')
        print(ansOptions[counter][myMaxIndex])
        rule = r'<(.*?)>'
        button = re.findall(rule, ansOptions[counter][myMaxIndex])
        button_number=re.findall("\d+",str(button))
        # print(int(button_number[0]))
        Path='/home/Wufang/FYP/Q2A/encoder/data/assistq/train/'

        with open(Path+item+'/buttons.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            for i, rows in enumerate(reader):
                if i == int(button_number[0])-1:
                    row = rows
        csv_model.append(row)
        counter += 1

    print(csv_model)
    with open("/home/Wufang/FYP/Q2A/button.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_model)
        writer.writerows(csv_model)
    csv1 = open("/home/Wufang/FYP/Q2A/button.csv", 'r')

    dict[item]=csv1.read()
    csv1.close()
    file = open("buttons.json",'w')
    buttonDict = json.dumps(dict)
    file.write(buttonDict)
    file.close()
    # distname = open('/home/Wufang/FYP/Q2A/button.csv', 'w')
    # pd_10 = pd.DataFrame(csv_model)  # 将列表转换为DataFrame格式
    # pd_10.to_csv(distname,index=False)
    return Output

@app.route('/list')
def hello_world():  # put application's code here
    f = open('list.json')
    list = f.read()
    return list
@app.route('/images',methods=['GET'])
def getImage():  # put application's code here
    # f = open('data/images.json')
    # list = f.read()
    # imagDict = json.loads(list)
    # f.close()
    if request.method == 'GET':
        name0 = request.args.get('name','')
        global item
        item=name0
        name1=name0.split("_")[0]
        Path='/home/Wufang/FYP/Q2A/encoder/data/assistq/train/'
        image = open(Path+name0+'/images/'+name1+'-user.jpg', 'rb')
        imageData = image.read()
        encoded = base64.b64encode(imageData)
        print(name0)
        dict = {}
        dict[name0] = str(encoded, encoding='utf8')
        image.close()
        # file = open("images.json", 'w')
        # imageDict = json.dumps(dict)
        # file.write(imageDict)
        # file.close()
        print(name1)
        return dict[name0]

@app.route('/search',methods=['GET'])
def getButton():  # put application's code here
    if request.method == 'GET':
        # global item
        # item = request.args.get('name', '')
        global question
        question=request.args.get('q', '')
        print(question)
        initialize()
        f = open('/home/Wufang/FYP/Q2A/encoder/data/assistq/train.json', 'r')
        qaData = json.load(f)
        f.close()
        # itemData = qaData[item]
        global item
        item='microwave_y3fpx'
        itemData = qaData[item]
        # print(itemData)
        number_select = 0
        for selec in itemData:
            if selec['question'] != question+"?":
                number_select += 1
            else:
                break
        batch = loadData(number_select)
        output = m.model([batch])  # 调用网络
        print(output)
        out=formatOutput(output,number_select)

        f = open('/home/Wufang/FYP/Q2A/buttons.json')
        list = f.read()
        buttonDict = json.loads(list)
        print("csv")
        print(buttonDict[item])
        print("cs v")
        f.close()

        return buttonDict[item]

if __name__ == '__main__':
    # app.run()
    app.run(debug=True,host='0.0.0.0',port=8000)
