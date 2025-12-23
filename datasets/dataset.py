import os
import os.path as osp
import sys
import random
import math
import numpy as np
import torch
import pickle
import PIL
from PIL import Image
import io

from torch.utils.data import Dataset

from .utils import convert_examples_to_features, read_examples
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
from .transforms import PIL_TRANSFORMS


# Meta Information
SUPPORTED_DATASETS = {
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
    },
    'flickr': {
        'splits': ('train', 'val', 'test')
    },
    'pathology2': {
        'splits': ('train', 'val','testA','testB')
    }
}


class VGDataset(Dataset):
    def __init__(self, data_root, split_root='data', dataset='referit', transforms=[],
                 debug=False, test=False, split='train', max_query_len=128,
                 bert_mode='bert-base-uncased', cache_images=False):
        super(VGDataset, self).__init__()

        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.test = test
        self.transforms = []

        self.getitem = self.getitem__PIL
        self.read_image = self.read_image_from_path_PIL
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))


        self.debug = debug

        self.query_len = max_query_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)

        # setting datasource
        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        elif self.dataset == 'pathology':
            self.dataset_root = osp.join(self.data_root, 'pathology')
            self.im_dir = osp.join(self.dataset_root, 'images')
        elif self.dataset == 'pathology2':
            self.dataset_root = osp.join(self.data_root, 'pathology2')
            self.im_dir = osp.join(self.dataset_root, 'images')
        else:  # refer coco etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(self.dataset_root, 'COCO2014', 'train2014')

        dataset_split_root = osp.join(self.split_root, self.dataset)
        valid_splits = SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        # read the image set info
        self.imgset_info = []
        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = 'refpath_{0}.pth'.format( split)
            imgset_path = osp.join(dataset_split_root, imgset_file)
            self.imgset_info += torch.load(imgset_path, map_location="cpu")

        # process the image set info
        if self.dataset == 'flickr':
            self.img_names, self.bboxs, self.phrases = zip(*self.imgset_info)
        else:
            self.img_names, _, self.bboxs, self.phrases, _,self.reason = zip(*self.imgset_info)

        self.cache_images = cache_images
        if cache_images:
            self.images_cached = [None] * len(self) #list()
            self.read_image_orig_func = self.read_image
            self.read_image = self.read_image_from_cache

        self.covert_bbox = []
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):  # for refcoco, etc
            # xywh to xyxy
            for bbox in self.bboxs:
                bbox = np.array(bbox, dtype=np.float32)
                bbox[2:] += bbox[:2]
                self.covert_bbox.append(bbox)
        else:
            for bbox in self.bboxs:  # for referit, flickr
                bbox = np.array(bbox, dtype=np.float32)
                self.covert_bbox.append(bbox)


    def __len__(self):
        return len(self.img_names)

    def image_path(self, idx):  # notice: db index is the actual index of data.
        return osp.join(self.im_dir, self.img_names[idx])

    def annotation_box(self, idx):
        return self.covert_bbox[idx].copy()

    def phrase(self, idx):
        return self.phrases[idx]

    def cache(self, idx):
        self.images_cached[idx] = self.read_image_orig_func(idx)

    def read_image_from_path_PIL(self, idx):
        image_path = self.image_path(idx)
        pil_image = Image.open(image_path).convert('RGB')
        return pil_image

    def read_image_from_cache(self, idx):
        image = self.images_cached[idx]
        return image

    def __getitem__(self, idx):
        return self.getitem(idx)


    def getitem__PIL(self, idx):
        # reading images
        image = self.read_image(idx)
        orig_image = image

        # read bbox annotation
        bbox = self.annotation_box(idx)
        bbox = torch.tensor(bbox)
        # read phrase
        phrase = self.phrase(idx)
        phrase = phrase.lower()
        orig_phrase = phrase
        reason = self.reason[idx]
        target = {}
        target['phrase'] = phrase
        # target['reason'] = reason
        target['bbox'] = bbox
        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()

        for transform in self.transforms:
            image, target = transform(image, target)


        # For BERT
        examples = read_examples(target['phrase'], idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        
        # For Reason
        word_id_reason = []
        word_mask_reason = []
        num_reason = 3
        if len(reason)<3:
            num_reason = len(reason)
        for i in range(0,num_reason):
            examples_reason = read_examples(reason[i],idx)
            features_reason = convert_examples_to_features(
            examples=examples_reason, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id_reason.append(features_reason[0].input_ids)
            word_mask_reason.append(features_reason[0].input_mask)
        # 如果不足3个reason，补齐
        for i in range(num_reason,3):
            word_id_reason.append(features_reason[0].input_ids)
            word_mask_reason.append(features_reason[0].input_mask)
        target['word_id_reason_0'] = torch.tensor(word_id_reason[0], dtype=torch.long)
        target['word_mask_reason_0'] = torch.tensor(word_mask_reason[0], dtype=torch.bool)
        target['word_id_reason_1'] = torch.tensor(word_id_reason[1], dtype=torch.long)
        target['word_mask_reason_1'] = torch.tensor(word_mask_reason[1], dtype=torch.bool)
        target['word_id_reason_2'] = torch.tensor(word_id_reason[2], dtype=torch.long)
        target['word_mask_reason_2'] = torch.tensor(word_mask_reason[2], dtype=torch.bool)
        if 'mask' in target:
            mask = target.pop('mask')
            return image, mask, target


        return image, target