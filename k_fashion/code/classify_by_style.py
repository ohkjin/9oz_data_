from __future__ import print_function, division

import torch
import torch.utils as utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# import train_model

# style 분류
def run():
    torch.multiprocessing.freeze_support()
   
    #저장된 모델 불러오기
    model_ft = torch.jit.load('./k-fashion/model/my9oz_model.pt')
    model_ft.eval()
      
    test_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_dataset = TestDataset(test_dir,transform=test_transform)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=64, num_workers=8)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result = []
    for fnames, data in tqdm(test_dataloader):
        data = data.to(device)
        output = model_ft(data)
        _,pred = torch.max(output,1)
        for j in range(len(fnames)):
            result.append(
                {
                    'filename':fnames[j].split(".")[0],
                    'style':pred.cpu().detach().numpy()[j]
                }
        )
    pd.DataFrame(sorted(result,key=lambda x:x['filename'])).to_csv('9oz_A17.csv',index=None)
