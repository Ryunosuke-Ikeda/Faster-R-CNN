
import numpy as np
import pandas as pd
 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


#dataloader.pyとmodel.pyからインポート
from dataloader import dataloader
from model import model


#datasetの読み込み
#データの場所
xml_paths_train=glob("########/train/*.xml")
xml_paths_val=glob("######/annotations/*.xml")

image_dir_train="#######/train"
image_dir_val="########/img"

#出力ファイルの名前
file_name='train_ALL_bdd100k'


#データをロード
train_dataloader,val_dataloader=dataloader(xml_paths_train,xml_paths_val,image_dir_train,image_dir_val)


#modelの読み込み
model=model()






##学習

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 1

#GPUのキャッシュクリア
import torch
torch.cuda.empty_cache()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
model.cuda()##


#既存手法
#engin.pyを参考にしてる

model.train()#学習モードに移行


for epoch in range(num_epochs):
 
    #model.train()#これから学習しますよ
    
    for i, batch in enumerate(train_dataloader):
        
 
        images, targets, image_ids = batch#####　batchはそのミニバッジのimage、tagets,image_idsが入ってる
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        
        ##学習モードでは画像とターゲット（ground-truth）を入力する
        ##返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）

        

        #デバック処理
        try:
            loss_dict= model(images, targets)

        except ValueError:
            print(image_ids)
        else:
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
        
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
    
            if (i+1) % 1000== 0:
                print(f"epoch #{epoch+1} Iteration #{i+1} loss: {loss_value}") 
                #モデルのセーブ
                torch.save(model, './model/'+file_name+'.pt')
               

        
        '''
        loss_dict= model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
 
        if (i+1) % 10== 0:
          print(f"epoch #{epoch+1} Iteration #{i+1} loss: {loss_value}") 
        '''
         

#セーブ
torch.save(model, './model/'+file_name+'.pt')

