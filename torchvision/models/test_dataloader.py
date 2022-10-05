import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
# import xmltodict # xml파일의 내용을 딕셔너리에 저장할 수 있는 메소드들이 들어있는 모듈입니다. 
from PIL import Image
import numpy as np
from tqdm import tqdm

class My_own_dataset():
    def __init__(self, root, train=True, transform=None, target_transform=None, resize=224) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.resize_factor = resize
        # torchvision/data/sample/F0350/A(친가)/1.Family
        self.dir_path = "torchvision/data/sample/"
        self.family_ids = ["F0350", "F0351", "F0352"]
        self.AorB = ( "A(친가)", "B(외가)" ) #paternal part, maternal part 
        self.relations = ("1.Family", "2.Individuals", "3.Age")
        self.pathes = [ f"{self.dir_path}{family_ids_}/{AorB_}/{relations_}" \
            for family_ids_ in self.family_ids for AorB_ in self.AorB for relations_ in self.relations ]
        
        self.images = [] # element format: ("image.jpg", "json_file") or "image.jpg"
        self.json = []
        
        for file in self.pathes:
            inner_dir = os.listdir(f'{file}/')
            # jpg_files = [ file_name for file_name in inner_dir \
            #     if (file_name[-4:]==".jpg") ] # .jpg파일 이름만 불러오기
            # print(f"# of files: {len(jpg_files)}")
            for file_name in inner_dir:
                if ( file_name[-4:]==".jpg" ):
                    # self.data.append( f"{file}/{file_name}" )
                    self.images.append( os.path.join(file, file_name) )
                else: self.json.append( os.path.join(file, file_name) )
        
        # for fullname in self.images: print(fullname)
        
        # if self.target_transform:
        #     label = self.target_transform(label)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = ( Image.open(self.images[index]).convert('RGB') ).resize((224,224))
        print(self.images[index])
        img.show()
        img_transform = transforms.Compose( [transforms.PILToTensor(), transforms.Resize((224,224))] )
        # img = torch.devide( img_transform(img), 255 )
        
        # if self.transform is not None:
        #     img = self.transform(img)
        
        return img
        
        
    def _check_exists(self) -> bool:
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))
        
        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
           os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))
           
if __name__ == '__main__':
    instance = My_own_dataset("../data/sample/", transform=transforms.ToTensor())
    print(instance.__getitem__(0))
    
    