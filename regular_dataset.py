import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import argparse
import json
import logging
import math
import os, cv2
import random
from pathlib import Path
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from templates.relation_words import relation_words
from templates.stop_words import stop_words
from packaging import version

if version.parse(version.parse(
        PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif')


def is_image_file(filename):
    # return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    return filename.endswith(IMG_EXTENSIONS)



class ReVersionandRegularDatasetv3forencoder(Dataset):
    '''return single image's name, for a image-wise noise-embeds training
    based on v2, add an additional ddim_noise into the get_item'''

    def __init__(
        self,
        data_root,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
        set="train",
        crop_p=0.0,

    ):
        self.data_root = data_root
        self.noise_data = (data_root + '_noise') if data_root[-1] != '/' else (data_root[:-1] + '_noise')
        
        
        regular_f = open(os.path.join(data_root, 'bbox.json'))
        self.bboxes = json.load(regular_f)

        # read per image templates
        local_f = open(os.path.join(data_root, 'text.json'))
        self.templates = json.load(local_f)
        print(f'self.templates={self.templates}')

        self.size = size
        self.crop_p = crop_p
        self.flip_p = flip_p

        # record image paths
        self.image_paths = []
        self.noise_paths = []
        for file_path in os.listdir(self.data_root):
            # if file_path != 'text.json':

            if is_image_file(file_path):
                self.image_paths.append(
                    os.path.join(self.data_root, file_path))
                self.noise_paths.append(os.path.join(self.noise_data, file_path.replace('png', 'pth').replace('jpg', 'pth')))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        if 'back2back' in self.data_root:
            self.black_latent = torch.load('./reversion_benchmark_v1/black.pth', weights_only=True)
        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def tranform_img(self, img):
        image = Image.fromarray(img)
        # image = image.resize((self.size, self.size),
        #                     resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1)
    def tranform_mask(self, img):
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image)#.astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, i):
        example = {}


        image_path = self.image_paths[i % self.num_images]
        image_name = image_path.split('/')[-1]

        noise_path = self.noise_paths[i % self.num_images]
        latent_emb = torch.load(noise_path, weights_only=True)
        # exemplar images
        
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image_height, image_width = img.shape[0], img.shape[1]
        example["original_sizes"] = [image_height, image_width]
        regular_bboxes = []
        bboxes = self.bboxes[image_name]

        # for back-to-back relation, add some additional augmentation in horizontal dim
        if 'back2back' in self.data_root:
            # image_height, image_width = img.shape[0], img.shape[1]
            add = 96
            w_img = np.zeros([image_height, image_width+add*2,3])
            w_img[:,add:add+image_width, :] = img.copy()
            img = w_img

            tmp_latent = torch.zeros((1,4,(self.size)//8,(self.size+add*2)//8))
            tmp_latent[:,:,:,add//8:(add+self.size)//8] = latent_emb
            tmp_latent[:,:,:,:add//8] = self.black_latent[:,:,:,:add//8]
            tmp_latent[:,:,:,-add//8:] = self.black_latent[:,:,:,-add//8:]
            latent_emb = tmp_latent

            tmp_bbox = [[new_bbox[0], new_bbox[1]+add, new_bbox[2], new_bbox[3]+add] for new_bbox in bboxes]
            bboxes = tmp_bbox

            image_height, image_width = img.shape[0], img.shape[1]

            
        '''make a regular img with radom masked-bboxes  '''
        regular_img = img.copy()
        regular_mask = np.ones([image_height, image_width])
        

        #对 bbox的原始坐标进行一些初步抖动，引入一些初级的随机性
        for bbox in bboxes:
            min_height, min_width, max_height, max_width = bbox

            # Calculate the range for randomization for both min and max values
            min_height_range = (-(max_height - min_height) * 0.10, (max_height - min_height) * 0.05)
            min_width_range = (-(max_width - min_width) * 0.10, (max_width - min_width) * 0.05)

            max_height_range = (-(max_height - min_height) * 0.05, (max_height - min_height) * 0.10)
            max_width_range = (-(max_width - min_width) * 0.05, (max_width - min_width) * 0.10)

            
            # Randomly adjust min_height and min_width within the specified ranges and ensure they stay within bounds
            new_min_height = min_height + random.uniform(min_height_range[0], min_height_range[1])
            new_min_width = min_width + random.uniform(min_width_range[0], min_width_range[1])
            new_min_height = int(max(0, min(new_min_height, image_height)))
            new_min_width = int(max(0, min(new_min_width, image_width)))
            
            # Randomly adjust max_height and max_width within the specified ranges and ensure they stay within bounds
            new_max_height = max_height + random.uniform(max_height_range[0], max_height_range[1])
            new_max_width = max_width + random.uniform(max_width_range[0], max_width_range[1])
            new_max_height = int(max(new_min_height, min(new_max_height, image_height)))  # Ensure max is always >= new min
            new_max_width = int(max(new_min_width, min(new_max_width, image_width)))  # Ensure max is always >= new min
            
            # Create a new bbox with the randomized and constrained values
            # new_bbox = (int(new_min_height), int(new_min_width), int(new_max_height), int(new_max_width))

            regular_img[new_min_height:new_max_height, new_min_width:new_max_width,:] = 0
            regular_mask[new_min_height:new_max_height, new_min_width:new_max_width] = 0
            regular_bboxes.append([new_min_height,new_min_width, new_max_height, new_max_width,])
            # regular_bboxes needed to be fix if self.center_crop==True, however we didn't fix it.

        # if crop:
        if random.random() < self.crop_p:
            min_bbox_h, min_bbox_w = 9999, 9999
            max_bbox_h, max_bbox_w = -1, -1
            for bbox in regular_bboxes:
                min_height, min_width, max_height, max_width = bbox
                min_bbox_h = min(min_bbox_h, min_height)
                min_bbox_w = min(min_bbox_w, min_width)
                max_bbox_h = max(max_bbox_h, max_height)
                max_bbox_w = max(max_bbox_w, max_width)
                
            # 随机裁剪：确定裁剪区域
            crop_min_height = random.randint(0, min_bbox_h)
            crop_min_width = random.randint(0, min_bbox_w)
            crop_max_height = random.randint(max_bbox_h, image_height)
            crop_max_width = random.randint(max_bbox_w, image_width)
        
            # 进行裁剪
            cropped_image = img[crop_min_height:crop_max_height, crop_min_width:crop_max_width,:]
            
            lat_min_height = round(self.size/image_height*crop_min_height)
            lat_max_height = round(self.size/image_height*crop_max_height)
            lat_min_width = round(self.size/image_width*crop_min_width)
            lat_max_width = round(self.size/image_width*crop_max_width)
            cropped_latent = latent_emb[:,:,lat_min_height//8:lat_max_height//8, lat_min_width//8:lat_max_width//8]

            # 更新bbox坐标
            cropped_bboxes = []
            for bbox in regular_bboxes:
                min_height, min_width, max_height, max_width = bbox
                cropped_bboxes.append([new_min_height - crop_min_height, new_min_width - crop_min_width, new_max_height - crop_min_height, new_max_width - crop_min_width])

        else:
            cropped_image = img
            cropped_latent = latent_emb
            cropped_bboxes = regular_bboxes
        
        # Resize到目标尺寸
        if cropped_image.shape[0]!= self.size or cropped_image.shape[1] != self.size or cropped_latent.shape[0]!= self.size//8 or cropped_latent.shape[1]!= self.size//8:
            
            resized_image = cv2.resize(cropped_image, (self.size, self.size))
            mode = 'bicubic'
            resize_latent = torch.nn.functional.interpolate(cropped_latent, size = [self.size//8, self.size//8], mode = mode) # shape: 1,1,64,64
                            
            
            # 更新bbox坐标为resize后的坐标
            scale_x = self.size / float(cropped_image.shape[1])
            scale_y = self.size / float(cropped_image.shape[0])
            
            resized_bbox = [[
                round(new_bbox[0] * scale_y),
                round(new_bbox[1] * scale_x),
                round(new_bbox[2] * scale_y),
                round(new_bbox[3] * scale_x)
            ] for new_bbox in cropped_bboxes]
        else:
            resized_image = cropped_image
            resize_latent = cropped_latent
            resized_bbox = cropped_bboxes

        # 水平翻转
        if random.random() < self.flip_p:
            flipped_image = cv2.flip(resized_image, 1)  # 水平翻转
            flipped_bbox = [[new_bbox[0], resized_image.shape[1] - new_bbox[3], new_bbox[2], resized_image.shape[1] - new_bbox[1]]for new_bbox in resized_bbox]
            flipped_latent = torch.flip(resize_latent, dims=[3])
        else:
            flipped_image = resized_image
            flipped_latent = resize_latent
            flipped_bbox = resized_bbox

        
        
            
        # seed = torch.random.seed()

        # torch.random.manual_seed(seed)
        # example["pixel_values"] = self.tranform_img(flipped_image)
        # torch.random.manual_seed(seed)
        # example["regular_img"] = self.tranform_img(regular_img)
        # torch.random.manual_seed(seed)
        # example["regular_mask"]= self.tranform_mask(regular_mask)

        example["bboxes"]= flipped_bbox

        example["name"] = image_name

        example["latent_emb"] = flipped_latent[0]
        

        return example



class ReVersionandRegularDatasetv3(Dataset):
    '''return single image's name, for a image-wise noise-embeds training
    based on v2, add an additional ddim_noise into the get_item'''

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
        crop_p=0.0,
        set="train",
        placeholder_token="*",
        center_crop=False,
        relation_words=None,
        num_positives=1,
        regular_thres=0.1,
    ):
        self.data_root = data_root
        self.noise_data = (data_root + '_noise') if data_root[-1] != '/' else (data_root[:-1] + '_noise')
        # define the regularization dataset's path and prompts
        
        # base_root = data_root[:-(len(cur_relation))]

        '''regular method 3: based on method 2, but apply it with a random mask module'''
        regular_f = open(os.path.join(data_root, 'bbox.json'))
        self.bboxes = json.load(regular_f)
        
        # read per image templates
        local_f = open(os.path.join(data_root, 'text.json'))
        self.templates = json.load(local_f)
        print(f'self.templates={self.templates}')
        self.crop_p = crop_p
        self.flip_p = flip_p

        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.regular_thres = regular_thres
        # for Relation-Steering
        self.relation_words = relation_words
        self.num_positives = num_positives

        # record image paths
        self.image_paths = []
        self.noise_paths = []
        for file_path in os.listdir(self.data_root):
            # if file_path != 'text.json':

            if is_image_file(file_path):
                self.image_paths.append(
                    os.path.join(self.data_root, file_path))
                self.noise_paths.append(os.path.join(self.noise_data, file_path.replace('png', 'pth').replace('jpg', 'pth')))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def tranform_img(self, img):
        # image = Image.fromarray(img)
        # image = image.resize((self.size, self.size),
        #                     resample=self.interpolation)

        # image = self.flip_transform(image)
        image = img.astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1)
    def tranform_mask(self, img):
        # image = Image.fromarray(img)
        # image = image.resize((self.size, self.size),
        #                     resample=self.interpolation)

        # image = self.flip_transform(image)
        # image = np.array(image)#.astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(img).unsqueeze(0)

    def __getitem__(self, i):
        example = {}


        image_path = self.image_paths[i % self.num_images]
        image_name = image_path.split('/')[-1]

        noise_path = self.noise_paths[i % self.num_images]
        latent_emb = torch.load(noise_path, weights_only=True)

        # determin whether the exemplar images or the regular images
        r = random.random()
        example["is_regular"] = True if r <= self.regular_thres else False

        regular_text = random.choice(
            self.templates[image_name]).format('and')

            # coarse descriptions
        text = random.choice(
            self.templates[image_name]).format(self.placeholder_token)
      
        # randomly sample positive words for L_steer
        if self.num_positives > 0:
            positive_words = random.sample(
                self.relation_words, k=self.num_positives)
            positive_words_string = " ".join(positive_words)
            example["positive_ids"] = self.tokenizer(
                positive_words_string,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["regular_ids"] = self.tokenizer(
            regular_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_text"] = text
        example["regular_text"] = regular_text

        # exemplar images
        
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image_height, image_width = img.shape[0], img.shape[1]
        example["original_sizes"] = [image_height, image_width]

        # for back-to-back relation, add some additional augmentation in horizontal dim
        # if 'back2back' in self.data_root:
        #     # image_height, image_width = img.shape[0], img.shape[1]
        #     w_img = np.zeros([image_height, image_width+200,3])
        #     w_img[:,100:100+image_width, :] = img.copy()
        #     img = w_img
        #     image_height, image_width = img.shape[0], img.shape[1]


        '''make a regular img with radom masked-bboxes  '''
        # regular_img = img.copy()
        regular_bboxes = []
        bboxes = self.bboxes[image_name]
        for bbox in bboxes:
            min_height, min_width, max_height, max_width = bbox
    
            # Calculate the range for randomization for both min and max values
            min_height_range = (-(max_height - min_height) * 0.10, (max_height - min_height) * 0.05)
            min_width_range = (-(max_width - min_width) * 0.10, (max_width - min_width) * 0.05)

            max_height_range = (-(max_height - min_height) * 0.05, (max_height - min_height) * 0.10)
            max_width_range = (-(max_width - min_width) * 0.05, (max_width - min_width) * 0.10)

            
            # Randomly adjust min_height and min_width within the specified ranges and ensure they stay within bounds
            new_min_height = min_height + random.uniform(min_height_range[0], min_height_range[1])
            new_min_width = min_width + random.uniform(min_width_range[0], min_width_range[1])
            new_min_height = int(max(0, min(new_min_height, image_height)))
            new_min_width = int(max(0, min(new_min_width, image_width)))
            
            # Randomly adjust max_height and max_width within the specified ranges and ensure they stay within bounds
            new_max_height = max_height + random.uniform(max_height_range[0], max_height_range[1])
            new_max_width = max_width + random.uniform(max_width_range[0], max_width_range[1])
            new_max_height = int(max(new_min_height, min(new_max_height, image_height)))  # Ensure max is always >= new min
            new_max_width = int(max(new_min_width, min(new_max_width, image_width)))  # Ensure max is always >= new min
            
            # Create a new bbox with the randomized and constrained values
            # new_bbox = (int(new_min_height), int(new_min_width), int(new_max_height), int(new_max_width))

            # regular_img[new_min_height:new_max_height, new_min_width:new_max_width,:] = 0
            # regular_mask[new_min_height:new_max_height, new_min_width:new_max_width] = 0
            regular_bboxes.append([new_min_height,new_min_width, new_max_height, new_max_width,])
            # regular_bboxes needed to be fix if self.center_crop==True, however we didn't fix it.
        # if crop:
        if random.random() < self.crop_p:
            min_bbox_h, min_bbox_w = 9999, 9999
            max_bbox_h, max_bbox_w = -1, -1
            for bbox in regular_bboxes:
                min_height, min_width, max_height, max_width = bbox
                min_bbox_h = min(min_bbox_h, min_height)
                min_bbox_w = min(min_bbox_w, min_width)
                max_bbox_h = max(max_bbox_h, max_height)
                max_bbox_w = max(max_bbox_w, max_width)
                
            # 随机裁剪：确定裁剪区域
            crop_min_height = random.randint(0, min_bbox_h)
            crop_min_width = random.randint(0, min_bbox_w)
            crop_max_height = random.randint(max_bbox_h, image_height)
            crop_max_width = random.randint(max_bbox_w, image_width)
        
            # 进行裁剪
            cropped_image = img[crop_min_height:crop_max_height, crop_min_width:crop_max_width,:]
            
            lat_min_height = round(self.size/image_height*crop_min_height)
            lat_max_height = round(self.size/image_height*crop_max_height)
            lat_min_width = round(self.size/image_width*crop_min_width)
            lat_max_width = round(self.size/image_width*crop_max_width)
            cropped_latent = latent_emb[:,:,lat_min_height//8:lat_max_height//8, lat_min_width//8:lat_max_width//8]

            # 更新bbox坐标
            cropped_bboxes = []
            for bbox in regular_bboxes:
                min_height, min_width, max_height, max_width = bbox
                cropped_bboxes.append([new_min_height - crop_min_height, new_min_width - crop_min_width, new_max_height - crop_min_height, new_max_width - crop_min_width])

        else:
            cropped_image = img
            cropped_latent = latent_emb
            cropped_bboxes = regular_bboxes

        # Resize到目标尺寸
        if cropped_image.shape[0]!= self.size or cropped_image.shape[1] != self.size or cropped_latent.shape[0]!= self.size//8 or cropped_latent.shape[1]!= self.size//8:
            
            resized_image = cv2.resize(cropped_image, (self.size, self.size))
            mode = 'bicubic'
            resize_latent = torch.nn.functional.interpolate(cropped_latent, size = [self.size//8, self.size//8], mode = mode) # shape: 1,1,64,64
                            
            
            # 更新bbox坐标为resize后的坐标
            scale_x = self.size / float(cropped_image.shape[1])
            scale_y = self.size / float(cropped_image.shape[0])
            
            resized_bbox = [[
                round(new_bbox[0] * scale_y),
                round(new_bbox[1] * scale_x),
                round(new_bbox[2] * scale_y),
                round(new_bbox[3] * scale_x)
            ] for new_bbox in cropped_bboxes]
        else:
            resized_image = cropped_image
            resize_latent = cropped_latent
            resized_bbox = cropped_bboxes

        # 水平翻转
        if random.random() < self.flip_p:
            flipped_image = cv2.flip(resized_image, 1)  # 水平翻转
            flipped_bbox = [[new_bbox[0], resized_image.shape[1] - new_bbox[3], new_bbox[2], resized_image.shape[1] - new_bbox[1]]for new_bbox in resized_bbox]
            flipped_latent = torch.flip(resize_latent, dims=[3])
        else:
            flipped_image = resized_image
            flipped_latent = resize_latent
            flipped_bbox = resized_bbox

        flipped_regular_img = flipped_image.copy()
        regular_mask = np.ones_like(flipped_image)
        for bbox in flipped_bbox:
            flipped_regular_img[bbox[0]:bbox[2],bbox[1]:bbox[3],:] = 0
        
            regular_mask[bbox[0]:bbox[2],bbox[1]:bbox[3],:] = 0
        # seed = torch.random.seed()

        # torch.random.manual_seed(seed)
        example["pixel_values"] = self.tranform_img(flipped_image)
        # torch.random.manual_seed(seed)
        example["regular_img"] = self.tranform_img(flipped_regular_img)
        # torch.random.manual_seed(seed)
        example["regular_mask"]= self.tranform_mask(regular_mask)
        example["bboxes"]= flipped_bbox

        example["name"] = image_name
        example["latent_emb"] = flipped_latent[0]


        return example



class ReVersionandRegularDatasetv2(Dataset):
    '''return single image's name, for a image-wise noise-embeds trainging'''
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
        set="train",
        placeholder_token="*",
        center_crop=False,
        relation_words=None,
        num_positives=1,
        regular_thres=0.1,
    ):
        self.data_root = data_root
        
        # define the regularization dataset's path and prompts
        
        # base_root = data_root[:-(len(cur_relation))]

        '''regular method 3: based on method 2, but apply it with a random mask module'''
        regular_f = open(os.path.join(data_root, 'bbox.json'))
        self.bboxes = json.load(regular_f)
        
        # read per image templates
        local_f = open(os.path.join(data_root, 'text.json'))
        self.templates = json.load(local_f)
        print(f'self.templates={self.templates}')

        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.regular_thres = regular_thres
        # for Relation-Steering
        self.relation_words = relation_words
        self.num_positives = num_positives

        # record image paths
        self.image_paths = []
        for file_path in os.listdir(self.data_root):
            # if file_path != 'text.json':

            if is_image_file(file_path):
                self.image_paths.append(
                    os.path.join(self.data_root, file_path))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def tranform_img(self, img):
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1)
    def tranform_mask(self, img):
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image)#.astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, i):
        example = {}


        image_path = self.image_paths[i % self.num_images]
        image_name = image_path.split('/')[-1]

        # determin whether the exemplar images or the regular images
        r = random.random()
        example["is_regular"] = True if r <= self.regular_thres else False

        regular_text = random.choice(
            self.templates[image_name]).format('and')

            # coarse descriptions
        text = random.choice(
            self.templates[image_name]).format(self.placeholder_token)
      
        # randomly sample positive words for L_steer
        if self.num_positives > 0:
            positive_words = random.sample(
                self.relation_words, k=self.num_positives)
            positive_words_string = " ".join(positive_words)
            example["positive_ids"] = self.tokenizer(
                positive_words_string,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["regular_ids"] = self.tokenizer(
            regular_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_text"] = text
        example["regular_text"] = regular_text

        # exemplar images
        
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image_height, image_width = img.shape[0], img.shape[1]
        '''make a regular img with radom masked-bboxes  '''
        regular_img = img.copy()
        regular_mask = np.ones([image_height, image_width])
        regular_bboxes = []
        bboxes = self.bboxes[image_name]
        for bbox in bboxes:
            min_height, min_width, max_height, max_width = bbox
    
            # Calculate the range for randomization for both min and max values
            min_height_range = (-(max_height - min_height) * 0.10, (max_height - min_height) * 0.05)
            min_width_range = (-(max_width - min_width) * 0.10, (max_width - min_width) * 0.05)

            max_height_range = (-(max_height - min_height) * 0.05, (max_height - min_height) * 0.10)
            max_width_range = (-(max_width - min_width) * 0.05, (max_width - min_width) * 0.10)

            
            # Randomly adjust min_height and min_width within the specified ranges and ensure they stay within bounds
            new_min_height = min_height + random.uniform(min_height_range[0], min_height_range[1])
            new_min_width = min_width + random.uniform(min_width_range[0], min_width_range[1])
            new_min_height = int(max(0, min(new_min_height, image_height)))
            new_min_width = int(max(0, min(new_min_width, image_width)))
            
            # Randomly adjust max_height and max_width within the specified ranges and ensure they stay within bounds
            new_max_height = max_height + random.uniform(max_height_range[0], max_height_range[1])
            new_max_width = max_width + random.uniform(max_width_range[0], max_width_range[1])
            new_max_height = int(max(new_min_height, min(new_max_height, image_height)))  # Ensure max is always >= new min
            new_max_width = int(max(new_min_width, min(new_max_width, image_width)))  # Ensure max is always >= new min
            
            # Create a new bbox with the randomized and constrained values
            # new_bbox = (int(new_min_height), int(new_min_width), int(new_max_height), int(new_max_width))

            regular_img[new_min_height:new_max_height, new_min_width:new_max_width,:] = 0
            regular_mask[new_min_height:new_max_height, new_min_width:new_max_width] = 0
            regular_bboxes.append([new_min_height,new_min_width, new_max_height, new_max_width,])
            # regular_bboxes needed to be fix if self.center_crop==True, however we didn't fix it.

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]
            regular_img = regular_img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]
            regular_mask = regular_mask[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

        h, w = img.shape[0], img.shape[1]
        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        example["pixel_values"] = self.tranform_img(img)
        torch.random.manual_seed(seed)
        example["regular_img"] = self.tranform_img(regular_img)
        torch.random.manual_seed(seed)
        example["regular_mask"]= self.tranform_mask(regular_mask)

        example["name"] = image_name
        example["original_sizes"] = [h, w]
        # resized_bboxes = {}
        # for idx, bbox in enumerate(regular_bboxes):
        #     new_min_height, new_min_width,new_max_height,new_max_width = bbox
        #     new_min_height = round(self.size/h*new_min_height)
        #     new_max_height = round(self.size/h*new_max_height)
        #     new_min_width = round(self.size/w*new_min_width)
        #     new_max_width = round(self.size/w*new_max_width)
        #     resized_bboxes[idx] = [new_min_height, new_min_width,new_max_height,new_max_width]
        # example["regular_bboxes"]= resized_bboxes
        return example



class ReVersionandRegularDataset(Dataset):

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
        set="train",
        placeholder_token="*",
        center_crop=False,
        relation_words=None,
        num_positives=1,
    ):
        self.data_root = data_root
        
        # define the regularization dataset's path and prompts
        
        # base_root = data_root[:-(len(cur_relation))]

        '''regular method 3: based on method 2, but apply it with a random mask module'''
        regular_f = open(os.path.join(data_root, 'bbox.json'))
        self.bboxes = json.load(regular_f)
        
        # read per image templates
        local_f = open(os.path.join(data_root, 'text.json'))
        self.templates = json.load(local_f)
        print(f'self.templates={self.templates}')

        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        # for Relation-Steering
        self.relation_words = relation_words
        self.num_positives = num_positives

        # record image paths
        self.image_paths = []
        for file_path in os.listdir(self.data_root):
            # if file_path != 'text.json':

            if is_image_file(file_path):
                self.image_paths.append(
                    os.path.join(self.data_root, file_path))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def tranform_img(self, img):
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1)
    def tranform_mask(self, img):
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image)#.astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, i):
        example = {}


        image_path = self.image_paths[i % self.num_images]
        image_name = image_path.split('/')[-1]

        # determin whether the exemplar images or the regular images
        r = random.random()
        example["is_regular"] = True if r <= 0.1 else False

        regular_text = random.choice(
            self.templates[image_name]).format('and')

            # coarse descriptions
        text = random.choice(
            self.templates[image_name]).format(self.placeholder_token)
      
        # randomly sample positive words for L_steer
        if self.num_positives > 0:
            positive_words = random.sample(
                self.relation_words, k=self.num_positives)
            positive_words_string = " ".join(positive_words)
            example["positive_ids"] = self.tokenizer(
                positive_words_string,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["regular_ids"] = self.tokenizer(
            regular_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_text"] = text
        example["regular_text"] = regular_text

        # exemplar images
        
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image_height, image_width = img.shape[0], img.shape[1]
        '''make a regular img with radom masked-bboxes  '''
        regular_img = img.copy()
        regular_mask = np.ones([image_height, image_width])
        regular_bboxes = []
        bboxes = self.bboxes[image_name]
        for bbox in bboxes:
            min_height, min_width, max_height, max_width = bbox
    
            # Calculate the range for randomization for both min and max values
            min_height_range = (-(max_height - min_height) * 0.10, (max_height - min_height) * 0.05)
            min_width_range = (-(max_width - min_width) * 0.10, (max_width - min_width) * 0.05)

            max_height_range = (-(max_height - min_height) * 0.05, (max_height - min_height) * 0.10)
            max_width_range = (-(max_width - min_width) * 0.05, (max_width - min_width) * 0.10)

            
            # Randomly adjust min_height and min_width within the specified ranges and ensure they stay within bounds
            new_min_height = min_height + random.uniform(min_height_range[0], min_height_range[1])
            new_min_width = min_width + random.uniform(min_width_range[0], min_width_range[1])
            new_min_height = int(max(0, min(new_min_height, image_height)))
            new_min_width = int(max(0, min(new_min_width, image_width)))
            
            # Randomly adjust max_height and max_width within the specified ranges and ensure they stay within bounds
            new_max_height = max_height + random.uniform(max_height_range[0], max_height_range[1])
            new_max_width = max_width + random.uniform(max_width_range[0], max_width_range[1])
            new_max_height = int(max(new_min_height, min(new_max_height, image_height)))  # Ensure max is always >= new min
            new_max_width = int(max(new_min_width, min(new_max_width, image_width)))  # Ensure max is always >= new min
            
            # Create a new bbox with the randomized and constrained values
            # new_bbox = (int(new_min_height), int(new_min_width), int(new_max_height), int(new_max_width))

            regular_img[new_min_height:new_max_height, new_min_width:new_max_width,:] = 0
            regular_mask[new_min_height:new_max_height, new_min_width:new_max_width] = 0
            regular_bboxes.append([new_min_height,new_min_width, new_max_height, new_max_width,])
            # regular_bboxes needed to be fix if self.center_crop==True, however we didn't fix it.

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]
            regular_img = regular_img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]
            regular_mask = regular_mask[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

        
        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        example["pixel_values"] = self.tranform_img(img)
        torch.random.manual_seed(seed)
        example["regular_img"] = self.tranform_img(regular_img)
        torch.random.manual_seed(seed)
        example["regular_mask"]= self.tranform_mask(regular_mask)
        example["regular_bboxes"]= regular_bboxes

        return example

class ReVersionRegularDataset(Dataset):

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
        set="train",
        placeholder_token="*",
        center_crop=False,
        relation_words=None,
        num_positives=1,
    ):
        self.data_root = data_root
        
        # define the regularization dataset's path and prompts
        
        cur_relation = data_root.split('/')[-1]
        # base_root = data_root[:-(len(cur_relation))]

        '''regular method 1: utilize other relation exemplar image to regular current R'''
        # self.regular_sets = ['shake_hands','hug','back2back','eye_concat','on']#'talk_to','other'
        '''regular method 2:  for the current R, labeling per img artifically to get a masked regularization img set'''
        # self.regular_sets = [cur_relation+'_regular']

        # self.regular_image_paths = []
        # self.regular_json_texts = {}
        # for mode in self.regular_sets:
        #     if mode == cur_relation: continue
        #     regular_root = os.path.join(base_root, mode)
        #     for file_path in os.listdir(regular_root):
        #         if is_image_file(file_path):
        #             self.regular_image_paths.append(
        #             os.path.join(regular_root, file_path))
        #         regular_f = open(os.path.join(regular_root, 'text.json'))
        #         self.regular_json_texts[mode] = json.load(regular_f)
        '''regular method 3: based on method 2, but apply it with a random mask module'''
        regular_f = open(os.path.join(data_root, 'bbox.json'))
        self.bboxes = json.load(regular_f)
        
        # read per image templates
        local_f = open(os.path.join(data_root, 'text.json'))
        self.templates = json.load(local_f)
        print(f'self.templates={self.templates}')

        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        # for Relation-Steering
        self.relation_words = relation_words
        self.num_positives = num_positives

        # record image paths
        self.image_paths = []
        for file_path in os.listdir(self.data_root):
            # if file_path != 'text.json':

            if is_image_file(file_path):
                self.image_paths.append(
                    os.path.join(self.data_root, file_path))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}


        image_path = self.image_paths[i % self.num_images]
        image_name = image_path.split('/')[-1]

        # determin whether the exemplar images or the regular images
        r = random.random()
        if r <= 0.1:
            example["is_regular"] = True
            text = random.choice(
                self.templates[image_name]).format('and')
        else:
            example["is_regular"] = False
            # coarse descriptions
            text = random.choice(
                self.templates[image_name]).format(self.placeholder_token)
      
        # randomly sample positive words for L_steer
        if self.num_positives > 0:
            positive_words = random.sample(
                self.relation_words, k=self.num_positives)
            positive_words_string = " ".join(positive_words)
            example["positive_ids"] = self.tokenizer(
                positive_words_string,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # exemplar images
        
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image_height, image_width = img.shape[0], img.shape[1]
        if example["is_regular"]:
            bboxes = self.bboxes[image_name]
            for bbox in bboxes:
                min_height, min_width, max_height, max_width = bbox
        
                # Calculate the range for randomization for both min and max values
                min_height_range = (-(max_height - min_height) * 0.10, (max_height - min_height) * 0.05)
                min_width_range = (-(max_width - min_width) * 0.10, (max_width - min_width) * 0.05)

                max_height_range = (-(max_height - min_height) * 0.05, (max_height - min_height) * 0.10)
                max_width_range = (-(max_width - min_width) * 0.05, (max_width - min_width) * 0.10)

                
                # Randomly adjust min_height and min_width within the specified ranges and ensure they stay within bounds
                new_min_height = min_height + random.uniform(min_height_range[0], min_height_range[1])
                new_min_width = min_width + random.uniform(min_width_range[0], min_width_range[1])
                new_min_height = int(max(0, min(new_min_height, image_height)))
                new_min_width = int(max(0, min(new_min_width, image_width)))
                
                # Randomly adjust max_height and max_width within the specified ranges and ensure they stay within bounds
                new_max_height = max_height + random.uniform(max_height_range[0], max_height_range[1])
                new_max_width = max_width + random.uniform(max_width_range[0], max_width_range[1])
                new_max_height = int(max(new_min_height, min(new_max_height, image_height)))  # Ensure max is always >= new min
                new_max_width = int(max(new_min_width, min(new_max_width, image_width)))  # Ensure max is always >= new min
                
                # Create a new bbox with the randomized and constrained values
                # new_bbox = (int(new_min_height), int(new_min_width), int(new_max_height), int(new_max_width))

                img[new_min_height:new_max_height, new_min_width:new_max_width,:] = 0

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        return example

class ReVersionRegularDatasetnoRandom(Dataset):

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
        set="train",
        placeholder_token="*",
        center_crop=False,
        relation_words=None,
        num_positives=1,
    ):
        self.data_root = data_root
        
        # define the regularization dataset's path and prompts
        
        cur_relation = data_root.split('/')[-1]
        base_root = data_root[:-(len(cur_relation))]

        '''regular method 1: utilize other relation exemplar image to regular current R'''
        if cur_relation == 'hug':
            self.regular_sets = ['eye_contact']#'talk_to','other' 'shake_hands',
        elif cur_relation == 'eye_contact':
            self.regular_sets = ['eye-reg']
        elif cur_relation == 'byebye':
            self.regular_sets = ['eye_contact']
        elif cur_relation == 'back2back':
            self.regular_sets = ['back2back-reg']
        '''regular method 2:  for the current R, labeling per img artifically to get a masked regularization img set'''
        # self.regular_sets = [cur_relation+'_regular']

        regular_f = open(os.path.join(data_root, 'bbox.json'))
        self.bboxes = json.load(regular_f)

        self.regular_image_paths = []
        self.regular_json_texts = {}
        for mode in self.regular_sets:
            if mode == cur_relation: continue
            regular_root = os.path.join(base_root, mode)
            for file_path in os.listdir(regular_root):
                if is_image_file(file_path):
                    self.regular_image_paths.append(
                    os.path.join(regular_root, file_path))
                regular_f = open(os.path.join(regular_root, 'text.json'))
                self.regular_json_texts[mode] = json.load(regular_f)
        '''regular method 2: based on method 2, but apply it with a random mask module'''
        
        
        # read per image templates
        local_f = open(os.path.join(data_root, 'text.json'))
        self.templates = json.load(local_f)
        print(f'self.templates={self.templates}')

        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        # for Relation-Steering
        self.relation_words = relation_words
        self.num_positives = num_positives

        # record image paths
        self.image_paths = []
        for file_path in os.listdir(self.data_root):
            # if file_path != 'text.json':

            if is_image_file(file_path):
                self.image_paths.append(
                    os.path.join(self.data_root, file_path))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def tranform_img(self, img):
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).permute(2, 0, 1)
    def tranform_mask(self, img):
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                            resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image)#.astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)

        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, i):
        example = {}

        # determin whether the exemplar images or the regular images
        r = random.random()
        if r <= 0.1:
        # regular images
            example["is_regular"] = True
            image_path = random.choice(self.regular_image_paths)
            regular_mode = image_path.split('/')[-2]
            image_name = image_path.split('/')[-1]

            text = random.choice(
                self.regular_json_texts[regular_mode][image_name]).format('and')


        else:
        # exemplar images
            example["is_regular"] = False

            image_path = self.image_paths[i % self.num_images]
            image_name = image_path.split('/')[-1]
            

            placeholder_string = self.placeholder_token

            # coarse descriptions
            text = random.choice(
                self.templates[image_name]).format(placeholder_string)

            # randomly sample positive words for L_steer
            if self.num_positives > 0:
                positive_words = random.sample(
                    self.relation_words, k=self.num_positives)
                positive_words_string = " ".join(positive_words)
                example["positive_ids"] = self.tokenizer(
                    positive_words_string,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
        
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        if not example["is_regular"]:
            image_height, image_width = img.shape[0], img.shape[1]
            '''make a regular img with radom masked-bboxes  '''
            regular_img = img.copy()
            regular_mask = np.ones([image_height, image_width])
            bboxes = self.bboxes[image_name]
            for bbox in bboxes:
                min_height, min_width, max_height, max_width = bbox
        
                # Calculate the range for randomization for both min and max values
                min_height_range = (-(max_height - min_height) * 0.10, (max_height - min_height) * 0.05)
                min_width_range = (-(max_width - min_width) * 0.10, (max_width - min_width) * 0.05)

                max_height_range = (-(max_height - min_height) * 0.05, (max_height - min_height) * 0.10)
                max_width_range = (-(max_width - min_width) * 0.05, (max_width - min_width) * 0.10)

                
                # Randomly adjust min_height and min_width within the specified ranges and ensure they stay within bounds
                new_min_height = min_height + random.uniform(min_height_range[0], min_height_range[1])
                new_min_width = min_width + random.uniform(min_width_range[0], min_width_range[1])
                new_min_height = int(max(0, min(new_min_height, image_height)))
                new_min_width = int(max(0, min(new_min_width, image_width)))
                
                # Randomly adjust max_height and max_width within the specified ranges and ensure they stay within bounds
                new_max_height = max_height + random.uniform(max_height_range[0], max_height_range[1])
                new_max_width = max_width + random.uniform(max_width_range[0], max_width_range[1])
                new_max_height = int(max(new_min_height, min(new_max_height, image_height)))  # Ensure max is always >= new min
                new_max_width = int(max(new_min_width, min(new_max_width, image_width)))  # Ensure max is always >= new min
                
                # Create a new bbox with the randomized and constrained values
                # new_bbox = (int(new_min_height), int(new_min_width), int(new_max_height), int(new_max_width))

                regular_img[new_min_height:new_max_height, new_min_width:new_max_width,:] = 0
                regular_mask[new_min_height:new_max_height, new_min_width:new_max_width] = 0
                

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        example["pixel_values"] = self.tranform_img(img)
        if not example["is_regular"]:
            torch.random.manual_seed(seed)
            example["regular_img"] = self.tranform_img(regular_img)
            torch.random.manual_seed(seed)
            example["regular_mask"]= self.tranform_mask(regular_mask)

        return example
