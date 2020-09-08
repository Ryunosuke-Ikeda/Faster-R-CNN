import numpy as np
import pandas as pd
 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataloader import dataloader

#datasetの読み込み
#データの場所
xml_paths_train=glob("../../dataset/bdd100K_to_VOC/train_test/*.xml")
xml_paths_val=glob("../../dataset/hokkaido_test/val_test/annotations/*.xml")

image_dir_train="D:/bdd100k/bdd100k/bdd100k/images/100k/train"
image_dir_val="../../dataset/hokkaido_test/val_test/img"


#データをロード
train_dataloader,val_dataloader=dataloader(xml_paths_train,xml_paths_val,image_dir_train,image_dir_val)

#modelのロード
model=torch.load('./model/train_ALL.pt')


def show(val_dataloader,model):
    import matplotlib.pyplot as plt
    from PIL import ImageDraw, ImageFont
    from PIL import Image
    

    #GPUのキャッシュクリア
    import torch
    torch.cuda.empty_cache()
   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    #device = torch.device('cpu')    
    model.to(device)
    model.eval()

    images, targets, image_ids = next(iter(val_dataloader))

    images = list(img.to(device) for img in images)
    
    #推論時は予測を返す
    '''
     - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    '''
    outputs = model(images)

    for i, image in enumerate(images):

        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))

        boxes = outputs[i]["boxes"].data.cpu().numpy()
        scores = outputs[i]["scores"].data.cpu().numpy()
        labels = outputs[i]["labels"].data.cpu().numpy()

        #category = {0: "background", 1: "dog", 2: "cat"}
        #category = {0: 'background',1:'car',2:'bus',3:'person',4:'bicycle',5:'motorbike',6:'train'}
        category={0: 'background',1:'person', 2:'traffic light',3: 'train',4: 'traffic sign', 5:'rider', 6:'car', 7:'bike',8: 'motor', 9:'truck', 10:'bus'}



        boxes = boxes[scores >= 0.5].astype(np.int32)###############0.5
        scores = scores[scores >= 0.5]####################0.5
        image_id = image_ids[i]

        for i, box in enumerate(boxes):
            draw = ImageDraw.Draw(image)
            label = category[labels[i]]
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

            # ラベルの表示

            from PIL import Image, ImageDraw, ImageFont 
            #fnt = ImageFont.truetype('/content/mplus-1c-black.ttf', 20)
            fnt = ImageFont.truetype("arial.ttf", 10)#40
            text_w, text_h = fnt.getsize(label)
            draw.rectangle([box[0], box[1], box[0]+text_w, box[1]+text_h], fill="red")
            draw.text((box[0], box[1]), label, font=fnt, fill='white')
            
        #画像を保存したい時用
        #image.save(f"resample_test{str(i)}.png")

        fig, ax = plt.subplots(1, 1)
        ax.imshow(np.array(image))

    plt.show()

show(val_dataloader,model)