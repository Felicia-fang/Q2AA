import os
import json
import base64
baseURL = '/home/Wufang/FYP/Q2A/encoder/data/assistq/train/'
def createList():
    fileName = os.listdir(baseURL)
    list = json.dumps(fileName)
    f = open("list.json",'w')
    f.write(list)
    f.close()

createList()