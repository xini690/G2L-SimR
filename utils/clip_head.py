import json
import os
import random
import sys

import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as T
import clip
import cv2
import numpy as np
import torch
from torch import nn

from PIL import Image
from os.path import splitext
class CLIPProjectionHead(nn.Module):
    def __init__(self, embed_dim=512, proj_dim=512, hidden_dim=2048, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(embed_dim, proj_dim))
        else:
            layers.append(nn.Linear(embed_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_dim, proj_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return x


def load_and_resize_mask(val, target_size):
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


def load_image(filename, is_mask=False, resize_to=(224,224)):
    resize_to = (448,448)
    ext = splitext(filename)[1].lower()
    if ext == '.npy':
        img = Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        img = Image.fromarray(torch.load(filename).numpy())
    else:
        img = Image.open(filename)
        if not is_mask:
            img = img.convert('RGB')

    if resize_to is not None:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    return img


def get_features(args,data,json_path,disease_list,preprocess_clip,clip_model,device):
    donor_features = []

    with open(json_path, "r", encoding="utf-8") as f:
        prior_knowledge_dict = json.load(f)

    for idx in range(len(data)):
        img = load_image(data.iloc[idx]['images'])
        masks_list = [load_and_resize_mask(data.iloc[idx][d], img.size)
                      for d in disease_list]
        mask_arr = np.stack([(m > 0).astype(np.uint8) for m in masks_list], axis=-1)

        img_tensor = preprocess_clip(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feat = clip_model.encode_image(img_tensor)
            image_feat = image_feat.squeeze(0).cpu()

        image_name = data.iloc[idx]['images'].split('/')[-1]
        text = prior_knowledge_dict[image_name]["concise_summary"]
        text_tokens = clip.tokenize([text],truncate=True).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens)
            text_feat = text_feat.squeeze(0).cpu()

        donor_features.append({
            'img': img,
            'mask': mask_arr,
            'image_feat': image_feat,
            'text_feat': text_feat,
            'index': idx
        })

    return donor_features


def paste_lesion_semantic_vit(args, disease_list, data_disease,target_img, target_mask, donor_img, donor_mask,
                              lesion_size=32, device='cuda', vit_model=None, cnn_model=None):
    lesion_size=args.lesion_size

    semantics=None
    if len(disease_list) == 1:
        donor_mask = np.array(donor_mask)[:, :]
        donor_mask = donor_mask[:, :, 0]
        target_img = np.array(target_img)
        target_mask = np.array(target_mask)[:, :]
    else:
        disease_keys = data_disease[args.dataSet]

        if args.semantics == 'None':
            semantics = None
        else:
            semantics_idx = disease_keys.index(args.semantics)
            semantics = np.array(donor_mask)[:, :, semantics_idx]

        disease_idx = disease_keys.index(args.disease)
        donor_mask = np.array(donor_mask)[:, :, disease_idx]

        donor_mask = donor_mask[:, :]
        target_img = np.array(target_img)
        target_mask = np.array(target_mask)[:, :, 0]

    results=[]
    aug_result={
        "aug_img": None,
        "aug_mask": None,
        "pasted_region_mask": None,
        "target_patch": None,
        "donor_patch": None,
        "paste_regions": None
    }
    num_labels, labels_im = cv2.connectedComponents(donor_mask.astype(np.uint8))

    if num_labels <= 1:
        results.append(((aug_result,False)))
        return results

    lesion_areas = []
    for label in range(1, num_labels):
        area = np.sum(labels_im == label)
        lesion_areas.append((label, area))

    for idx, (label, area) in enumerate(lesion_areas):
        lesion_areas[idx] = (label, area, idx)
    lesion_areas.sort(key=lambda x: (x[1], x[2]), reverse=True)

    lesion_labels = [label for label, area,_ in lesion_areas[:min(args.max_lesions, len(lesion_areas))]]

    aug_result['aug_img'] = target_img
    aug_result['aug_mask'] = target_mask

    for lesion_label in lesion_labels:
        lesion_mask_region = (labels_im == lesion_label).astype(np.uint8)
        use_vit = True
        ys, xs = np.where(lesion_mask_region > 0)
        if len(ys) == 0 or (ys.max() - ys.min() < lesion_size or xs.max() - xs.min() < lesion_size):
            use_vit = False

        aug_result, use_vit = extract_donor_mask(args,
            ys, xs,
            donor_img, donor_mask,
            aug_result['aug_img'], aug_result['aug_mask'],
            lesion_size, vit_model, cnn_model, semantics, device,
            use_vit
        )

        aug_img = aug_result['aug_img']
        aug_mask = aug_result['aug_mask']

        if aug_img.dtype != np.uint8:
            aug_img = (aug_img * 255).astype(np.uint8)

        if aug_mask.dtype != np.uint8:
            aug_mask = (aug_mask * 255).astype(np.uint8)

        if aug_mask.ndim == 3 and aug_mask.shape[2] == 1:
            aug_mask = aug_mask.squeeze(-1)

        aug_result['aug_img'] = aug_img
        aug_result['aug_mask'] = aug_mask

        results.append((aug_result, use_vit))

    return results

def extract_donor_mask(args,ys, xs, donor_img, donor_mask, target_img, target_mask,
                       lesion_size, vit_model, cnn_model, semantics, device,
                       use_vit=True,soft_penalty=True):
    donor_img = np.array(donor_img)
    donor_mask = np.array(donor_mask)

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    lesion_patch = donor_img[ymin:ymax + 1, xmin:xmax + 1]
    lesion_mask_patch = donor_mask[ymin:ymax + 1, xmin:xmax + 1]

    ph, pw = lesion_patch.shape[:2]

    paste_regions=None
    candidates=[]

    if use_vit:
        candidates = find_best_patch_positions_vit_batch(
            target_img, lesion_patch, vit_model, device
        )
    else:
        candidates = find_best_patch_positions_cnn_batch(
            target_img, lesion_patch, cnn_model, device
        )

    valid_candidates = []
    count=0
    for (i, j, score) in candidates:
        count+=1
        if count >10:
             break

        y_end = min(i + ph, target_img.shape[0])
        x_end = min(j + pw, target_img.shape[1])

        sim_feat = score

        mask_patch = target_mask[i:y_end, j:x_end]
        lesion_mask_crop = lesion_mask_patch[:y_end - i, :x_end - j]

        intersection = np.sum(mask_patch * (lesion_mask_crop > 0))
        union = np.sum(mask_patch + (lesion_mask_crop > 0)) - intersection
        sim_mask = intersection / (union + 1e-6)

        penalty_ratio = 0.0
        if semantics is not None:
            semantics_patch = semantics[i:y_end, j:x_end]
            penalty_ratio = np.mean(semantics_patch > 0)

        if soft_penalty:
            power = 1.5
            mul_weight = max(0.0, 1.0 - (penalty_ratio ** power))
        else:
            if penalty_ratio > 0.2:
                continue
            mul_weight = 1.0

        sim_feat_norm = (sim_feat + 1.0) / 2.0

        alpha = 0.7
        beta = 0.25
        gamma = 0.05

        combined = alpha * sim_feat_norm + beta * sim_mask - gamma * penalty_ratio

        adjusted_score = combined * mul_weight

        valid_candidates.append((i, j, adjusted_score))

    if len(valid_candidates) == 0:
        print("[Warning] No valid paste location found, skip lesion paste.")
        return {
            "aug_img": target_img,
            "aug_mask": target_mask,
            "pasted_region_mask": np.zeros_like(target_mask),
            "target_patch": np.zeros((ph, pw, 3), dtype=np.uint8),
            "donor_patch": lesion_patch,
            "paste_regions": paste_regions
        }, use_vit

    valid_candidates.sort(key=lambda x: x[2], reverse=True)
    i_target, j_target, _ = valid_candidates[0]

    alpha = (lesion_mask_patch > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = alpha / (alpha.max() + 1e-6)

    y_end = min(i_target + ph, target_img.shape[0])
    x_end = min(j_target + pw, target_img.shape[1])
    h_clip = y_end - i_target
    w_clip = x_end - j_target

    alpha_clip = alpha[:h_clip, :w_clip]
    lesion_patch_clip = lesion_patch[:h_clip, :w_clip, :]
    alpha_clip_3c = np.repeat(alpha_clip[:, :, np.newaxis], 3, axis=2)

    target_patch_clip = target_img[i_target:y_end, j_target:x_end, :].copy()

    target_img[i_target:y_end, j_target:x_end, :] = (
        target_patch_clip * (1 - alpha_clip_3c) +
        lesion_patch_clip * alpha_clip_3c
    )

    target_mask[i_target:y_end, j_target:x_end] = np.maximum(
        target_mask[i_target:y_end, j_target:x_end],
        lesion_mask_patch[:h_clip, :w_clip]
    )

    pasted_region_mask = np.zeros(target_mask.shape, dtype=np.uint8)
    pasted_region_mask[i_target:y_end, j_target:x_end] = (alpha_clip > 0).astype(np.uint8)

    paste_regions=collect_all_paste_regions(candidates, ph, pw, target_img)

    return {
        "aug_img": target_img,
        "aug_mask": target_mask,
        "pasted_region_mask": pasted_region_mask,
        "target_patch": target_patch_clip,
        "donor_patch": lesion_patch_clip,
        "paste_regions": paste_regions
    }, use_vit


def collect_all_paste_regions(candidates, ph, pw, target_img):
    paste_regions = []

    for (i, j, score) in candidates:
        y_end = min(i + ph, target_img.shape[0])
        x_end = min(j + pw, target_img.shape[1])

        region_patch = target_img[i:y_end, j:x_end, :].copy()
        paste_regions.append(region_patch)

    return paste_regions



def find_best_patch_positions_cnn_batch(target_arr, lesion_patch, cnn_model, device,
                                        coarse_stride=64, fine_stride=8, patch_size=64,
                                        top_k=8, resize_mode=cv2.INTER_LINEAR, batch_size=32):
    cnn_model.eval()

    H, W, _ = target_arr.shape
    patch_h, patch_w, _ = lesion_patch.shape

    lesion_patch_resized = cv2.resize(lesion_patch, (patch_size, patch_size), interpolation=resize_mode)
    lesion_tensor = torch.from_numpy(lesion_patch_resized.transpose(2,0,1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        lesion_feat = cnn_model.forward_features(lesion_tensor)
        lesion_feat_avg = lesion_feat.mean(dim=(2,3))

    large_patch_size = min(W, max(patch_w, patch_h) * 8)
    fine_stride = int(max(patch_w / 2, patch_h / 2) * 2)
    coarse_patches, coarse_coords = [], []
    for top in range(0, H - large_patch_size + 1, coarse_stride):
        for left in range(0, W - large_patch_size + 1, coarse_stride):
            patch = target_arr[top:top+large_patch_size, left:left+large_patch_size]
            patch_resized = cv2.resize(patch, (patch_size, patch_size), interpolation=resize_mode)
            coarse_patches.append(patch_resized)
            coarse_coords.append((left, top))

    if len(coarse_patches) == 0:
        return [(0,0,0)]

    coarse_candidates = batch_extract_similarity(coarse_patches, coarse_coords, cnn_model,
                                                 lesion_feat_avg, device, patch_size, batch_size)

    top_k = min(top_k, len(coarse_patches))
    coarse_candidates.sort(key=lambda x: x[2], reverse=True)
    coarse_top = coarse_candidates[:top_k]

    fine_patches, fine_coords = [], []
    for left0, top0, _ in coarse_top:
        right0 = left0 + large_patch_size
        bottom0 = top0 + large_patch_size
        for top in range(top0, bottom0 - patch_h + 1, fine_stride):
            for left in range(left0, right0 - patch_w + 1, fine_stride):
                patch = target_arr[top:top+patch_h, left:left+patch_w]
                patch_resized = cv2.resize(patch, (patch_size, patch_size), interpolation=resize_mode)
                fine_patches.append(patch_resized)
                fine_coords.append((left, top))

    if len(fine_patches) == 0:
        return [(0,0,0)]

    fine_candidates = batch_extract_similarity(fine_patches, fine_coords, cnn_model,
                                               lesion_feat_avg, device, patch_size, batch_size)

    fine_candidates.sort(key=lambda x: (-x[2], x[0], x[1]))
    return fine_candidates


def batch_extract_similarity(patches, coords, cnn_model, lesion_feat_avg, device, patch_size=64, batch_size=64):

    cnn_model.eval()
    feats = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_np = np.stack(patches[i:i+batch_size], axis=0)

            batch_tensor = torch.from_numpy(batch_np).permute(0,3,1,2).float().to(device)
            batch_feat = cnn_model.forward_features(batch_tensor).mean(dim=(2,3))
            feats.append(batch_feat)

    feats = torch.cat(feats, dim=0)

    lesion_rep = lesion_feat_avg.repeat(feats.size(0), 1)
    sims = F.cosine_similarity(lesion_rep, feats, dim=1).cpu().numpy()
    candidates = [(left, top, float(sim)) for (left, top), sim in zip(coords, sims)]

    return candidates

def find_best_patch_positions_vit_batch(target_img, lesion_patch, vit_model, device,
                                  coarse_stride=64, fine_stride=32,
                                  top_k=8, vit_input_size=224,batch_size=32):

    vit_model.eval()

    H, W, _ = target_img.shape
    patch_h, patch_w, _ = lesion_patch.shape

    transform = T.Compose([
        T.Resize((vit_input_size, vit_input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    lesion_tensor = transform(Image.fromarray(lesion_patch.astype(np.uint8))).unsqueeze(0).to(device)
    with torch.no_grad():
        lesion_feat_avg = vit_model.forward_features(lesion_tensor)

    large_patch_size = min(W, max(patch_w, patch_h) * 8)
    fine_stride = int(max(patch_w/2, patch_h/2) * 2)

    coarse_positions = []
    coarse_patches = []

    for top in range(0, H - large_patch_size + 1, coarse_stride):
        for left in range(0, W - large_patch_size + 1, coarse_stride):
            patch = target_img[top:top + large_patch_size, left:left + large_patch_size]
            patch = transform(Image.fromarray(patch.astype(np.uint8))).unsqueeze(0).to(device)
            coarse_positions.append((left, top))
            coarse_patches.append(patch)

    coarse_candidates = compute_similarity_batch(
        coarse_patches, coarse_positions, vit_model, lesion_feat_avg, device,batch_size
    )
    top_k=min(top_k, len(coarse_patches))
    coarse_candidates.sort(key=lambda x: x[2], reverse=True)
    top_regions = coarse_candidates[:top_k]

    fine_positions = []
    fine_patches = []

    for left0, top0, _ in top_regions:
        right0 = left0 + large_patch_size
        bottom0 = top0 + large_patch_size
        for top in range(top0, bottom0 - patch_h + 1, fine_stride):
            for left in range(left0, right0 - patch_w + 1, fine_stride):
                patch = target_img[top:top + patch_h, left:left + patch_w]
                patch = transform(Image.fromarray(patch.astype(np.uint8))).unsqueeze(0).to(device)
                fine_positions.append((left, top))
                fine_patches.append(patch)

    if len(fine_patches) == 0:
        return []

    fine_candidates = compute_similarity_batch(
        fine_patches, fine_positions, vit_model, lesion_feat_avg, device
    )

    fine_candidates.sort(key=lambda x: (-x[2], x[0], x[1]))

    return fine_candidates



def compute_similarity_batch(patches, coords, vit_model, lesion_feat_avg, device,batch_size=32):
    if len(patches) == 0:
        return torch.empty(0)

    feats = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]

            batch_tensor = torch.cat(batch_patches, dim=0).to(device)
            batch_feats = vit_model.forward_features(batch_tensor)
            feats.append(batch_feats)

    feats = torch.cat(feats, dim=0)
    lesion_rep = lesion_feat_avg.repeat(feats.size(0), 1)
    sims= F.cosine_similarity(lesion_rep, feats, dim=1).cpu().numpy()
    candidates = [(left, top, float(sim)) for (left, top), sim in zip(coords, sims)]

    return candidates


def ensure_pil(img):

    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] in [3, 4]:
            return Image.fromarray(img.astype(np.uint8))
        elif img.ndim == 2:
            return Image.fromarray(img.astype(np.uint8))
        else:
            raise ValueError(f"Unsupported numpy array shape for image: {img.shape}")
    else:
        raise TypeError(f"Unsupported type: {type(img)} (expected PIL.Image or np.ndarray)")



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setRNGSeed(seed)
    print(f'Random seed set to {seed}')
