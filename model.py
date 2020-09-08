import numpy as np
import pandas as pd
 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def model ():
    #モデルの定義

    #pretrain
    '''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)####True

    #num_classes = 3 # background, dog, cat 
    #num_classes = 7
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    '''


    #個人で定義

    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.



    #ここでエラーはいてる！！！！！！！！！！！！！
    '''
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    '''
    #デフォ
    roi_pooler =torchvision.ops.MultiScaleRoIAlign(
                    featmap_names=['0','1','2','3'],
                    output_size=7,
                    sampling_ratio=2)
    
        
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=11,#2
                    rpn_anchor_generator=anchor_generator)
                    #box_roi_pool=roi_pooler)

    return model
    
#print(model())