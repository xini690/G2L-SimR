import json
import os
import random

import cv2
import math
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
import clip
from multiprocessing import Pool
import torch.nn.functional as F
from torch.utils.data import Dataset

from skimage.io import imread
from types import SimpleNamespace
from scipy.ndimage import gaussian_filter

from resnet import ResNet50

import models_vit
from utils.clip_head import CLIPProjectionHead, load_image, get_features, paste_lesion_semantic_vit, ensure_pil, \
    set_seed

data_disease = {
    'IDRiD':['Haemorrhages','Optic_Disc'],
    'MAPLES-DR':['Macula','Optic_Disc'],
    'MMAC': ['Fuchs_Spot', 'Optic_Disc']
}


def unique_mask_values_from_path(mask_path):
    if mask_path is None or (isinstance(mask_path, float) and math.isnan(mask_path)):
        return [False]
    mask = imread(mask_path)
    return np.unique(mask)


class BasicDataset(Dataset):
    def __init__(self, args, excel_path, mask_suffix: str = ''):

        self.args = args
        self.csv_path = excel_path
        self.data = pd.read_excel(self.csv_path)
        assert 0 < args.scale <= 1, 'Scale must be between 0 and 1'
        self.scale = args.scale
        self.mask_suffix = mask_suffix

        if ',' in args.disease:
            self.disease_list = args.disease.split(',')
            mask_paths = []
            for d in self.disease_list:
                paths = self.data[d].dropna().tolist()
                mask_paths.extend(paths)
        else:
            mask_paths = self.data[args.disease].tolist()

        with Pool() as p:
            unique_list = list(p.imap(unique_mask_values_from_path, mask_paths))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique_list), axis=0).tolist()))

        self.json_path=args.lesion_knowledge_path
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.lesion_knowledge_dict = json.load(f)

        if ',' in args.disease:
            valid_mask = self.data[self.disease_list].notna().any(axis=1)
        else:
            valid_mask = self.data[args.disease].notna()
        valid_image = self.data['images'].notna()

        valid_rows = valid_image & valid_mask
        before_count = len(self.data)
        self.data = self.data[valid_rows].reset_index(drop=True)
        after_count = len(self.data)

        valid_indices = []
        for idx in range(len(self.data)):
            mask_path = self.data.iloc[idx][args.disease]
            if not os.path.exists(mask_path):
                continue
            mask_img = load_image(mask_path)
            mask_arr = np.array(mask_img)
            if np.any(mask_arr != 0):
                valid_indices.append(idx)

        before_count2 = len(self.data)
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        after_count2 = len(self.data)



    def __len__(self):
        return len(self.data)

    @staticmethod
    def preprocess(args, mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            if args.classes == 1:
                mask = np.zeros((newH, newW), dtype=np.int64)
                if img.ndim == 2:
                    mask[img != 0] = 1
                else:
                    mask[(img != 0).any(-1)] = 1

                return mask
            else:
                mask = np.zeros((newH, newW), dtype=np.int64)
                for i, v in enumerate(mask_values):
                    if img.ndim == 2:
                        mask[img == v] = i
                    else:
                        mask[(img == v).all(-1)] = i

                return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        image_path = row['images']
        img = load_image(row['images'])
        mask = load_image(row[self.args.disease])

        assert img.size == mask.size, \
            f'Image and mask {image_path} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.args, self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.args, self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'index': idx,
            'image_path':image_path
        }


class AugmentedDataset_ViT(BasicDataset):

    def __init__(self, args, excel_path, lesion_size=32, device='cuda', name_vit='vit_base_patch16_224'):
        super().__init__(args, excel_path)
        self.args = args
        self.aug_prob = args.aug_prob
        self.lesion_size = lesion_size
        self.device = device

        self.disease_list = data_disease[args.dataSet]

        vit_path = args.vit_path
        self.vit_model = models_vit.__dict__['vit_large_patch16'](
            num_classes=0,
            drop_path_rate=0.1,
            global_pool=True,
        )

        cnn_path = args.cnn_path
        self.cnn_model = ResNet50(args.classes)
        self.cnn_model = self.cnn_model.to(device)
        self.clip_model, self.preprocess_clip = clip.load("ViT-B/32", device=device, jit=False, download_root=None)

        clip_head_path=args.clip_head_path
        self.clip_head=CLIPProjectionHead(embed_dim=self.clip_model.visual.output_dim, proj_dim=512).to(device)

        if os.path.exists(vit_path):
            state_dict = torch.load(vit_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.vit_model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Weight file {vit_path} does not exist.")

        if cnn_path is not None:
            checkpoint = torch.load(cnn_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

            self.cnn_model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Weight file {cnn_path} does not exist.")

        if os.path.exists(clip_head_path):
            self.clip_head.load_state_dict(torch.load(clip_head_path, map_location="cpu", weights_only=True))
        else:
            raise FileNotFoundError(f"Weight file {clip_head_path} does not exist.")

        self.vit_model = self.vit_model.to(device)
        self.vit_model.eval()


        self.json_path = args.image_knowledge_path
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.image_knowledge_dict = json.load(f)

        self.lesion_knowledge_path=args.lesion_knowledge_path
        self.donor_lesion_features =get_features(args,self.data ,self.lesion_knowledge_path,self.disease_list, self.preprocess_clip, self.clip_model,self.device)

        self.non_empty_mask_indices = []
        for disease in self.disease_list:
            idx_list = []
            for idx in range(len(self.data)):
                img = load_image(self.data.iloc[idx]['images'])
                mask = self._load_and_resize_mask(self.data.iloc[idx][disease], img.size)
                if np.max(mask) > 0:
                    idx_list.append(idx)
            self.non_empty_mask_indices.append(idx_list)

        if self.aug_prob <= 1.0:
            self.indices = list(range(len(self.data)))
            self.is_augmented = [random.random() < args.aug_prob for _ in self.indices]
        else:
            n_original = len(self.data)
            n_aug = int(n_original * (self.aug_prob - 1))
            if n_aug <= n_original:
                aug_indices = random.sample(range(n_original), n_aug)
            else:
                aug_indices = [random.randint(0, n_original - 1) for _ in range(n_aug)]
            self.indices = list(range(n_original)) + aug_indices
            self.is_augmented = [False] * n_original + [True] * n_aug


    def _load_and_resize_mask(self, val, target_size):
        if isinstance(val, str) and os.path.exists(val):
            m = load_image(val)
            if not isinstance(m, Image.Image):
                m = Image.fromarray(np.array(m))
            m = m.convert("L")

            if m.size != target_size:
                m = m.resize(target_size, Image.NEAREST)

            m = np.array(m, dtype=np.uint8)
        else:
            m = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        return m

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.data.iloc[real_idx]
        image_path = row['images']
        img = load_image(row['images'])
        mask = load_image(row[self.args.disease])

        masks_list = [self._load_and_resize_mask(row[d], img.size)
                      for d in self.disease_list]

        mask_arr = np.stack([(m > 0).astype(np.uint8) for m in masks_list], axis=-1)
        if len(self.disease_list) == 1:
            mask_arr = mask_arr[:, :, 0]
            mask_arr = Image.fromarray(mask_arr)
        image_name = row['images']
        image_name = image_name.split('/')[-1]
        lesion_knowledge = self.lesion_knowledge_dict[image_name]["concise_summary"]
        image_knowledge = self.image_knowledge_dict[image_name]["concise_summary"]
        if self.is_augmented[idx]:
            img_tensor = self.preprocess_clip(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                target_img_feat = self.clip_model.encode_image(img_tensor).squeeze(0)

            text_tokens = clip.tokenize([lesion_knowledge],truncate=True).to(self.device)
            with torch.no_grad():
                target_text_feat = self.clip_model.encode_text(text_tokens).squeeze(0)

            alpha=0.5
            fused_target = F.normalize(alpha * target_img_feat.float() + (1 - alpha) * target_text_feat.float(), dim=-1)
            target_proj = self.clip_head(fused_target.unsqueeze(0))

            donor_proj_all = []
            valid_donors = []
            for d in self.donor_lesion_features:
                if d['index'] == idx:
                    continue
                fused_donor = F.normalize(
                    alpha * d['image_feat'].to(self.device).float() +
                    (1 - alpha) * d['text_feat'].to(self.device).float(),
                    dim=-1
                )
                donor_proj = self.clip_head(fused_donor.unsqueeze(0))
                donor_proj_all.append(donor_proj)
                valid_donors.append(d)

            donor_proj_all = torch.cat(donor_proj_all, dim=0)

            sim = F.cosine_similarity(target_proj, donor_proj_all)
            donor_idx = sim.argmax().item()
            donor = valid_donors[donor_idx]

            results = paste_lesion_semantic_vit(
                self.args, self.disease_list,data_disease,
                img, mask_arr, donor['img'], donor['mask'],
                lesion_size=self.lesion_size,
                device=self.device,
                vit_model=self.vit_model,
                cnn_model=self.cnn_model
            )

            img=results[-1][0]['aug_img']
            mask=results[-1][0]['aug_mask']
            image_knowledge=f'This is an enhanced picture of an eye with disease {self.args.disease}'
        img=ensure_pil(img)
        mask=ensure_pil(mask)

        img = self.preprocess(self.args, self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.args, self.mask_values, mask, self.scale, is_mask=True)
        return {
            'image': torch.tensor(img, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'knowledge' : image_knowledge,
            'image_path':image_path
        }



def build_dataset(is_train, args):
    excel_path = os.path.join(args.csv_path, f"{is_train}.xlsx")
    if is_train == 'train':
        if args.dataSet_mode == 'base':
            dataSet = BasicDataset(args, excel_path)
            print(len(dataSet))
        elif args.dataSet_mode == 'single':
            dataSet = AugmentedDataset_ViT(args, excel_path)
            print(len(dataSet))
        else:
            raise ValueError(f"Invalid dataSet_mode: {args.dataSet_mode}")
    else:
        dataSet = BasicDataset(args, excel_path)

    return dataSet


