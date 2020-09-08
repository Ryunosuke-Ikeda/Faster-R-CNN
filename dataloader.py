import numpy as np
import pandas as pd
 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


#データの場所
xml_paths_train=glob("######/*.xml")
xml_paths_val=glob("######/*.xml")

image_dir_train="######/train"
image_dir_val="######/img"


class xml2list(object):
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path):
        
        ret = []
        xml = ET.parse(xml_path).getroot()
        
        for size in xml.iter("size"):     
            width = float(size.find("width").text)
            height = float(size.find("height").text)
                
        for obj in xml.iter("object"):
            #なんかdifficultなかったから消す
            #difficult = int(obj.find("difficult").text)
            #if difficult == 1:
            #    continue          
            bndbox = [width, height]        
            name = obj.find("name").text.lower().strip() 
            bbox = obj.find("bndbox")            
            pts = ["xmin", "ymin", "xmax", "ymax"]     
            for pt in pts:        
              
                cur_pixel =  float(bbox.find(pt).text)
                ###########エラー対策 
                #cur_pixel =  int(bbox.find(pt).text)               
                bndbox.append(cur_pixel)           
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)    
            ret += [bndbox]
            
        return np.array(ret) # [width, height, xmin, ymin, xamx, ymax, label_idx]

class MyDataset(torch.utils.data.Dataset):
        
        def __init__(self, df, image_dir):
            
            super().__init__()
            
            self.image_ids = df["image_id"].unique()
            self.df = df
            self.image_dir = image_dir
            
        def __getitem__(self, index):
    
            transform = transforms.Compose([
                                            transforms.ToTensor()
            ])
    
            # 入力画像の読み込み
            image_id = self.image_ids[index]
            image = Image.open(f"{self.image_dir}/{image_id}.jpg")
            image = transform(image)
            
            # アノテーションデータの読み込み
            records = self.df[self.df["image_id"] == image_id]

            #######デバック処理（int変換）
            boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
            #boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.int32)
            
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
            
            labels = torch.tensor(records["class"].values, dtype=torch.int64)
            
            iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes
            target["labels"]= labels
            target["image_id"] = torch.tensor([index])
            target["area"] = area
            target["iscrowd"] = iscrowd
            
            return image, target, image_id
        
        def __len__(self):
            return self.image_ids.shape[0]



def dataloader (xml_paths_train,xml_paths_val,image_dir_train,image_dir_val):
    #trainのanotationの読み込み
    xml_paths=xml_paths_train

    #classes = ['car', 'bus', 'person', 'bicycle', 'motorbike', 'train']
    classes = ['person', 'traffic light', 'train', 'traffic sign', 'rider', 'car', 'bike', 'motor', 'truck', 'bus']
    transform_anno = xml2list(classes)
 
    df = pd.DataFrame(columns=["image_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
 
    for path in xml_paths:
        #image_id = path.split("/")[-1].split(".")[0]
        image_id = path.split("\\")[-1].split(".")[0]
        bboxs = transform_anno(path)
    
        for bbox in bboxs:
            tmp = pd.Series(bbox, index=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
            tmp["image_id"] = image_id
            df = df.append(tmp, ignore_index=True)

    df = df.sort_values(by="image_id", ascending=True)

    #df.to_csv('./debug/data.csv')

    

    #valのanotationの読み込み

    xml_paths=xml_paths_val

  
    #bdd100kval
    classes = ['person', 'traffic light', 'train', 'traffic sign', 'rider', 'car', 'bike', 'motor', 'truck', 'bus']


    transform_anno = xml2list(classes)
 
    df_val = pd.DataFrame(columns=["image_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
 
    for path in xml_paths:
        #image_id = path.split("/")[-1].split(".")[0]
        image_id = path.split("\\")[-1].split(".")[0]
        bboxs = transform_anno(path)
    
        for bbox in bboxs:
            tmp = pd.Series(bbox, index=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
            tmp["image_id"] = image_id
            df_val = df_val.append(tmp, ignore_index=True)

    df_val = df_val.sort_values(by="image_id", ascending=True)




    #画像の読み込み
    # 背景のクラス（0）が必要
    df["class"] = df["class"] + 1

    image_dir1=image_dir_train
    dataset = MyDataset(df, image_dir1)

    image_dir2=image_dir_val
    dataset_val = MyDataset(df_val, image_dir2)


    #データのロード
    torch.manual_seed(2020)
    
    #n_train = int(len(dataset) * 0.7)
    #n_val = len(dataset) - n_train 
    #train, val = torch.utils.data.random_split(dataset, [n_train, n_val])

    train=dataset
    val=dataset_val

    #バク確認
    #for i in train:
    #    print(i)

    def collate_fn(batch):
        return tuple(zip(*batch))
        
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate_fn)#3
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=False, collate_fn=collate_fn)#3

    return train_dataloader,val_dataloader 


t,v=dataloader(xml_paths_train,xml_paths_val,image_dir_train,image_dir_val)


#for i in t:
#    print(i)
    
