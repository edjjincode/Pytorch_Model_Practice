
import os.path as osp
from PIL import Image
import torch.utils.data as data

from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


def make_datapath_list(rootpath):
    

    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  
        img_path = (imgpath_template % file_id)  
        anno_path = (annopath_template % file_id)  
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  
        img_path = (imgpath_template % file_id)  
        anno_path = (annopath_template % file_id)  
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


class DataTransform():
   

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  
                RandomRotation(angle=[-10, 10]),  
                RandomMirror(),  
                Resize(input_size),  
                Normalize_Tensor(color_mean, color_std)  
            ]),
            'val': Compose([
                Resize(input_size), 
                Normalize_Tensor(color_mean, color_std) 
            ])
        }

    def __call__(self, phase, img, anno_class_img):
      
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        
        return len(self.img_list)

    def __getitem__(self, index):
        
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        
       
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   

     
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   

        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img
