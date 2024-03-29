
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch
import cv2
import numpy as np
import os.path as osp
from itertools import product as product
from math import sqrt as sqrt

import xml.etree.ElementTree as ET


from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans


from utils.match import match




def make_datapath_list(rootpath):


    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

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




class Anno_xml2list(object):
 
    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
       
    
        ret = []

        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):

            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            bndbox = []

            name = obj.find('name').text.lower().strip() 
            bbox = obj.find('bndbox') 

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
      
                cur_pixel = int(bbox.find(pt).text) - 1

        
                if pt == 'xmin' or pt == 'xmax':  
                    cur_pixel /= width
                else:  
                    cur_pixel /= height

                bndbox.append(cur_pixel)

     
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

     
            ret += [bndbox]

        return np.array(ret) 




class DataTransform():
   

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  
                ToAbsoluteCoords(), 
                PhotometricDistort(), 
                Expand(color_mean),  
                RandomSampleCrop(),  
                RandomMirror(),  
                ToPercentCoords(),  
                Resize(input_size), 
                SubtractMeans(color_mean)  
            ]),
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),  
                SubtractMeans(color_mean)  
            ])
        }

    def __call__(self, img, phase, boxes, labels):
 
        return self.data_transform[phase](img, boxes, labels)


class VOCDataset(data.Dataset):
 
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  
        self.transform = transform  
        self.transform_anno = transform_anno  

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, index):
     
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):

   
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  
        height, width, channels = img.shape  

        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])

        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


def od_collate_fn(batch):

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  
        targets.append(torch.FloatTensor(sample[1])) 

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


def make_vgg():
    layers = []
    in_channels = 3 

 
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
    
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


def make_extras():
    layers = []
    in_channels = 1024 

 
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)





def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

 
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

   
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

  
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]


    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

 
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]


    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)



class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  
        self.reset_parameters()  
        self.eps = 1e-10

    def reset_parameters(self):
  
        init.constant_(self.weight, self.scale)  

    def forward(self, x):

      
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

  
        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out



class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        self.image_size = cfg['input_size'] 

        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"])  
        self.steps = cfg['steps']  

        self.min_sizes = cfg['min_sizes']

        self.max_sizes = cfg['max_sizes']
    

        self.aspect_ratios = cfg['aspect_ratios']  

    def make_dbox_list(self):

        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  
    
                f_k = self.image_size / self.steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

         
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)

        output.clamp_(max=1, min=0)

        return output



def decode(loc, dbox_list):
    
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)

    boxes[:, :2] -= boxes[:, 2:] / 2  
    boxes[:, 2:] += boxes[:, :2]  
    return boxes



def nm_suppression(boxes, scores, overlap=0.45, top_k=200):


  
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()


    v, idx = scores.sort(0)

 
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]  

       
        keep[count] = i
        count += 1

   
        if idx.size(0) == 1:
            break

        idx = idx[:-1]

      
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

     
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

  
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

    
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

   
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)


        inter = tmp_w*tmp_h

       
        rem_areas = torch.index_select(area, 0, idx)  
        union = (rem_areas - inter) + area[i]  
        IoU = inter/union

   
        idx = idx[IoU.le(overlap)]  
    

    return keep, count




class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)  
        self.conf_thresh = conf_thresh  
        self.top_k = top_k  
        self.nms_thresh = nms_thresh 

    def forward(self, loc_data, conf_data, dbox_list):
     

        num_batch = loc_data.size(0)  
        num_dbox = loc_data.size(1)  
        num_classes = conf_data.size(2)  

      
        conf_data = self.softmax(conf_data)

        
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

      
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):

         
            decoded_boxes = decode(loc_data[i], dbox_list)

   
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):

            
                c_mask = conf_scores[cl].gt(self.conf_thresh)
         
                scores = conf_scores[cl][c_mask]

       
                if scores.nelement() == 0: 
                    continue

         
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
          
                boxes = decoded_boxes[l_mask].view(-1, 4)
          

        
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
               

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output  




class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  
        self.num_classes = cfg["num_classes"]  

     
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"])

      
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

  
        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        sources = list()  
        loc = list()  
        conf = list()  

     
        for k in range(23):
            x = self.vgg[k](x)

     
        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  
                sources.append(x)

  
        for (x, l, c) in zip(sources, self.loc, self.conf):

            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)


        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":  
      
            return self.detect(output[0], output[1], output[2])

        else:  
            return output
     


class MultiBoxLoss(nn.Module):


    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  
        self.negpos_ratio = neg_pos  
        self.device = device  

    def forward(self, predictions, targets):
       

    
        loc_data, conf_data, dbox_list = predictions

        num_batch = loc_data.size(0) 
        num_dbox = loc_data.size(1)  
        num_classes = conf_data.size(2)  

        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        for idx in range(num_batch): 

    
            truths = targets[idx][:, :-1].to(self.device) 
    
            labels = targets[idx][:, -1].to(self.device)

            dbox = dbox_list.to(self.device)


            variance = [0.1, 0.2]
  
            match(self.jaccard_thresh, truths, dbox,
                  variance, labels, loc_t, conf_t_label, idx)


        pos_mask = conf_t_label > 0  


        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

    
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

  
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')


        batch_conf = conf_data.view(-1, num_classes)

     
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')

    
        num_pos = pos_mask.long().sum(1, keepdim=True)  
        loss_c = loss_c.view(num_batch, -1)  
        loss_c[pos_mask] = 0  

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

   
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

      
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)


        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, num_classes)
 
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

    
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')


        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
