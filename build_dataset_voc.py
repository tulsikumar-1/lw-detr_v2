# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:51:20 2024

@author: Administrator
"""

# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data
import torchvision

import transforms as T
from torch.utils.data import DataLoader, DistributedSampler
import misc

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        
        
        
        # Adjust class labels from 1-5 to 0-4
        classes -= 1
        
        
        
        
        

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([640], max_size=1333),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([640]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set):
    """
    """

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    scales = [448, 512, 576, 640, 704, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    elif image_set == 'val':
        return T.Compose([
            T.SquareResize([640]),
            normalize,
        ])
    elif image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([640]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def Calculate_class_weights (dataset):
    labels = []
    for _, target in dataset:
        labels.extend(target['labels'].tolist())
    
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    
    # Create sample weights
    sample_weights = [class_weights[target['labels']].mean() for _, target in dataset]
    sample_weights = torch.tensor(sample_weights)
    
    return sample_weights

def build_dataset(image_folder, ann_file, image_set, batch_size, num_workers, square_div_64=False):
    if square_div_64:
        dataset = CocoDetection(image_folder, ann_file, transforms=make_coco_transforms_square_div_64(image_set))
    else:
        dataset = CocoDetection(image_folder, ann_file, transforms=make_coco_transforms(image_set))

    if image_set == 'train':
        drop_last = True

        # Initialize RandomSampler
        sampler = torch.utils.data.RandomSampler(dataset)
        
        # Use the sampler in a DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                 collate_fn=misc.collate_fn, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
        
    else:
        drop_last = False
        sampler_val = torch.utils.data.SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size, sampler=sampler_val, drop_last=drop_last,
                                 collate_fn=misc.collate_fn, num_workers=num_workers, pin_memory=True)

    return data_loader


      

"""
example use

img_folder='/content/traffic_monitoring/coco/train'
ann_file='/content/traffic_monitoring/coco/train/_annotations.coco.json'
image_set='val'
batch_size=8
num_workers=2
square_div_64=True
train_dataset=build_dataset(img_folder, ann_file, image_set,batch_size,num_workers,square_div_64)"""