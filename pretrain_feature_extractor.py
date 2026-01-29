import argparse
import json
import sys
import time

from torchvision.transforms.functional import to_pil_image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import os
import torchvision.transforms as T

from utils.clip_head import CLIPProjectionHead, get_features, set_seed
from utils.data_loading import BasicDataset, load_image, build_dataset
from resnet import ResNet50
import models_vit
import clip
from utils.data_loading import paste_lesion_semantic_vit
import warnings

from utils.dice_score import clip_proj_infoNCE_loss, patch_cosine_loss, contrastive_patch_loss
from utils.feature_extractor_evaluate import evaluate_cnn_vit, evaluate_clip_head

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--csv_path', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--lesion_knowledge_path', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--output_path', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--vit_path', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--cnn_path', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--dataSet', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--disease', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--Optic_Disc', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--aug_factor', type=float, default=0.5, help='Use bilinear upsampling')
    parser.add_argument('--seed', type=int, default=42, help='Number of classes')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--max_lesions', type=int, default=3, help='Number of classes')
    parser.add_argument('--augment_mode', type=str, default='random', help='Load model from a .pth file')
    parser.add_argument('--dataSet_mode', type=str, default='base', help='Load model from a .pth file')
    parser.add_argument('--config', default='cfgs/featExtraCfg.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args

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


def train_clip_head(args, device, clip_head, train_loader,val_loader,preprocess_clip, clip_model,donor_features):

    proj_optimizer = optim.Adam(clip_head.parameters(), lr=args.lr)
    best_clip_sim = 0.0
    best_loss=0.0
    for epoch in range(args.epochs):
        clip_head.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            batch_indices = batch['index'].tolist()
            text_tokens = clip.tokenize(batch['knowledge'],truncate=True).to(device)

            with torch.no_grad():
                imgs_clip_pre = torch.stack([preprocess_clip(to_pil_image(img.cpu())).to(device) for img in images])
                img_feats = F.normalize(clip_model.encode_image(imgs_clip_pre).float(), dim=-1)
                text_feats = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)

            alpha = 0.5
            fused_feats = F.normalize(alpha * img_feats + (1 - alpha) * text_feats, dim=-1)
            target_proj = clip_head(fused_feats)

            losses = []
            for i in range(target_proj.size(0)):
                current_idx = batch_indices[i]

                donor_feats_tensor = torch.stack([
                    F.normalize(
                        alpha * d['image_feat'].to(device).float() + (1 - alpha) * d['text_feat'].to(device).float(),
                        dim=-1
                    )
                    for d in donor_features if d['index'] != current_idx
                ])
                donor_proj_all = clip_head(donor_feats_tensor)

                sim = F.cosine_similarity(target_proj[i:i + 1], donor_proj_all)
                sorted_indices = torch.argsort(sim, descending=True)
                donor_proj_all_sorted = donor_proj_all[sorted_indices]

                num_regions=donor_proj_all.size(0)
                N=args.N
                num_pos = max(1, int(num_regions * N))
                pos_regions = donor_proj_all_sorted[:num_pos]
                neg_regions = donor_proj_all_sorted[-num_pos:]
                min_count=min(len(pos_regions), len(neg_regions))
                if min_count>0:
                    pos_regions = pos_regions[:min_count]
                    neg_regions = neg_regions[:min_count]
                    loss = contrastive_patch_loss(pos_regions, neg_regions)
                    losses.append(loss)

            loss = torch.stack(losses).mean()

            proj_optimizer.zero_grad()

            loss.backward()
            proj_optimizer.step()

            epoch_loss += loss.item()
            total_samples += target_proj.size(0)

        avg_loss = epoch_loss / total_samples
        print(f"[CLIP Head Train {epoch + 1}/{args.epochs}] | Avg Proj Loss: {avg_loss:.4f}")

        eval_loss = evaluate_clip_head(args,val_loader, donor_features, clip_model, clip_head, device, preprocess_clip)
        print(f"[CLIP Head Eval] | eval_loss: {eval_loss:.4f}")
        if epoch==0:
            best_loss=eval_loss

        if eval_loss <= best_loss:
            best_loss = eval_loss
            torch.save(clip_head.state_dict(), os.path.join(args.output_path, f'clip_model.pth'))
            print(f"Best CLIP Head saved with best_loss {eval_loss:.4f}")


def train_cnn_vit(args, device, clip_model, clip_head, train_loader,val_loader,preprocess_clip,donor_features,disease_list):

    cnn_model = ResNet50(num_classes=args.classes).to(device)
    vit_model = models_vit.__dict__['vit_large_patch16'](
        num_classes=0,
        drop_path_rate=0.1,
        global_pool=True
    ).to(device)

    if os.path.exists(args.vit_path):
        state_dict = torch.load(args.vit_path, map_location="cpu", weights_only=False)
        vit_model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Weight file {args.vit_path} does not exist.")


    if os.path.exists(args.cnn_path):
        checkpoint = torch.load(args.cnn_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        cnn_model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Weight file {args.cnn_path} does not exist.")

    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
    vit_optimizer = optim.Adam(vit_model.parameters(), lr=args.lr)


    clip_path = os.path.join(args.output_path, f'clip_model.pth')
    if os.path.exists(clip_path):
        state_dict = torch.load(clip_path, map_location="cpu", weights_only=True)
        clip_head.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Weight file {clip_path} does not exist.")

    best_loss = 0.0
    best_cnn_loss, best_vit_loss=0.0,0.0
    for epoch in range(args.epochs):

        cnn_loss_epoch = 0.0
        vit_loss_epoch = 0.0
        total_samples = 0
        total_vit_patches = 0
        total_cnn_patches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            batch_indices = batch['index'].tolist()
            text_tokens = clip.tokenize(batch['knowledge'],truncate=True).to(device)

            for i in range(images.size(0)):
                vit_patches_pos, vit_patches_neg = [], []
                cnn_patches_pos, cnn_patches_neg = [], []

                with torch.no_grad():
                    img_clip_pre = preprocess_clip(to_pil_image(images[i].cpu())).to(device).unsqueeze(0)
                    img_feat = F.normalize(clip_model.encode_image(img_clip_pre).float(), dim=-1)

                    text_tokens_single = clip.tokenize([batch['knowledge'][i]],truncate=True).to(device)
                    text_feat = F.normalize(clip_model.encode_text(text_tokens_single).float(), dim=-1)
                    alpha= 0.5
                    fused_feat = F.normalize(alpha * img_feat + (1-alpha) * text_feat, dim=-1)
                    target_proj = clip_head(fused_feat)

                    current_idx = batch['index'][i].item()

                    donor_feats = []
                    donor_indices = []

                    for idx, d in enumerate(donor_features):
                        if d['index'] == current_idx:
                            continue
                        alpha=0.5
                        fused_feat = alpha * d['image_feat'].to(device).float() + (1 - alpha) * d['text_feat'].to(device).float()
                        donor_feats.append(F.normalize(fused_feat, dim=-1))
                        donor_indices.append(idx)

                    donor_feats_tensor = torch.stack(donor_feats)

                    donor_proj_all = clip_head(donor_feats_tensor)
                    sim = F.cosine_similarity(target_proj, donor_proj_all)
                    sorted_indices = torch.argsort(sim, descending=True)


                donor = None
                disease_keys = data_disease[args.dataSet]
                disease_idx = disease_keys.index(args.disease)

                for idx in sorted_indices:
                    original_idx = donor_indices[idx.item()]
                    candidate_donor = donor_features[original_idx]
                    candidate_mask = candidate_donor['mask'][:, :, disease_idx]

                    if candidate_mask.max() > 0:
                        donor = candidate_donor
                        break

                if donor is not None:
                    donor_img, donor_mask = donor['img'], donor['mask']

                img = images[i].cpu().numpy().transpose(1, 2, 0)
                mask = masks[i].cpu().numpy()

                if mask.ndim == 2:
                    mask = mask[:, :, np.newaxis]

                results = paste_lesion_semantic_vit(
                    args, disease_list, data_disease,
                    target_img=img,
                    target_mask=mask,
                    donor_img=donor_img,
                    donor_mask=donor_mask,
                    lesion_size=args.lesion_size,
                    device=device,
                    vit_model=vit_model,
                    cnn_model=cnn_model
                )

                for res in results:
                    if 'paste_regions' not in res[0] or len(res[0]['paste_regions']) == 0:
                        continue
                    sorted_regions = res[0]['paste_regions']
                    use_vit= res[1]
                    num_regions = len(sorted_regions)

                    N=args.N
                    num_pos = max(1, int(num_regions * N))
                    pos_regions = sorted_regions[:num_pos]
                    neg_regions = sorted_regions[-num_pos:]

                    for region in pos_regions:
                        if region is not None and region.shape[0] > 0 and region.shape[1] > 0:
                            region_tensor = torch.from_numpy(region).permute(2, 0, 1).float()
                            if use_vit:
                                vit_patches_pos.append(region_tensor)
                            else:
                                cnn_patches_pos.append(region_tensor)

                    for region in neg_regions:
                        if region is not None and region.shape[0] > 0 and region.shape[1] > 0:
                            region_tensor = torch.from_numpy(region).permute(2, 0, 1).float()
                            if use_vit:
                                vit_patches_neg.append(region_tensor)
                            else:
                                cnn_patches_neg.append(region_tensor)

                if len(vit_patches_pos) > 0 and len(vit_patches_neg) > 0:
                    vit_model.train()
                    target_h, target_w = 224, 224
                    vit_pos_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(target_h, target_w),
                                                                mode='bilinear', align_corners=False).squeeze(0)
                                                  for p in vit_patches_pos]).to(device)
                    vit_neg_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(target_h, target_w),
                                                                mode='bilinear', align_corners=False).squeeze(0)
                                                  for p in vit_patches_neg]).to(device)

                    vit_pos_tensor = (vit_pos_tensor / 255.0 - 0.5) / 0.5
                    vit_neg_tensor = (vit_neg_tensor / 255.0 - 0.5) / 0.5


                    num_batches = min(len(vit_pos_tensor), len(vit_neg_tensor))
                    batch_size = min(8, num_batches)

                    for j in range(0, num_batches, batch_size):

                        pos_batch = vit_pos_tensor[j:j + batch_size]
                        neg_batch = vit_neg_tensor[j:j + batch_size]

                        pos_feats = vit_model.forward_features(pos_batch)
                        neg_feats = vit_model.forward_features(neg_batch)

                        if pos_batch.size(0) < 2 or neg_batch.size(0) < 2:
                            continue

                        vit_loss = contrastive_patch_loss(pos_feats, neg_feats)

                        vit_optimizer.zero_grad()
                        vit_loss.backward()
                        vit_optimizer.step()

                        vit_loss_epoch += vit_loss.item() * pos_batch.size(0)
                        total_vit_patches += pos_batch.size(0)

                if len(cnn_patches_pos) > 0 and len(cnn_patches_neg) > 0:
                    cnn_model.train()
                    target_h, target_w = 64,64
                    cnn_pos_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(target_h, target_w),
                                                                mode='bilinear', align_corners=False).squeeze(0)
                                                  for p in cnn_patches_pos]).to(device)
                    cnn_neg_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(target_h, target_w),
                                                                mode='bilinear', align_corners=False).squeeze(0)
                                                  for p in cnn_patches_neg]).to(device)

                    num_batches = min(len(cnn_pos_tensor), len(cnn_neg_tensor))
                    batch_size = min(8, num_batches)
                    for j in range(0, num_batches, batch_size):
                        pos_batch = cnn_pos_tensor[j:j + batch_size]
                        neg_batch = cnn_neg_tensor[j:j + batch_size]

                        if pos_batch.size(0) < 2 or neg_batch.size(0) < 2:
                            continue

                        pos_feats = cnn_model.forward_features(pos_batch).mean(dim=(2, 3))
                        neg_feats = cnn_model.forward_features(neg_batch).mean(dim=(2, 3))

                        cnn_loss = contrastive_patch_loss(pos_feats, neg_feats)

                        cnn_optimizer.zero_grad()
                        cnn_loss.backward()
                        cnn_optimizer.step()

                        cnn_loss_epoch += cnn_loss.item() * pos_batch.size(0)
                        total_cnn_patches += pos_batch.size(0)


            total_samples += images.size(0)

        avg_vit_loss = vit_loss_epoch / max(total_vit_patches, 1)
        avg_cnn_loss = cnn_loss_epoch / max(total_cnn_patches, 1)

        print(f"[Train {epoch + 1}/{args.epochs}] | CNN: {avg_cnn_loss:.4f} | ViT: {avg_vit_loss:.4f}")

        metrics = evaluate_cnn_vit(epoch, cnn_model, vit_model, clip_model, clip_head, preprocess_clip, val_loader, device,
                           args,
                           donor_features, disease_list, data_disease)
        print(f"[Eval] | eval_loss: {metrics['total_loss']:.4f} | CNN:{metrics['cnn_loss']:.4f} | ViT:{metrics['vit_loss']:.4f}")
        if epoch==0:
            best_cnn_loss = metrics['cnn_loss']
            best_vit_loss = metrics['vit_loss']
        if metrics['cnn_loss'] <= best_cnn_loss:
            best_cnn_loss = metrics['cnn_loss']
            torch.save(cnn_model.state_dict(), os.path.join(args.output_path, f'cnn_model.pth'))
            print(f"best_cnn_loss={best_cnn_loss:.4f}")
        if metrics['vit_loss'] <= best_vit_loss:
            best_vit_loss = metrics['vit_loss']
            torch.save(vit_model.state_dict(), os.path.join(args.output_path, f'vit_model.pth'))
            print(f"best_vit_loss={best_vit_loss:.4f}")

def train_features_extractor(args,device,data_disease):
    args.dataSet_mode = 'base'
    train_dataset = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    data = pd.read_excel(os.path.join(args.csv_path, 'train.xlsx'))
    disease_list = data_disease[args.dataSet]
    json_path = args.lesion_knowledge_path

    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device, jit=False)

    for p in clip_model.parameters():
        p.requires_grad = False

    clip_head = CLIPProjectionHead(embed_dim=clip_model.visual.output_dim, proj_dim=512).to(device)

    donor_features = get_features(args, data, json_path, disease_list, preprocess_clip, clip_model, device)

    train_clip_head(args, device, clip_head, train_loader,val_loader,preprocess_clip, clip_model,donor_features)

    train_cnn_vit(args, device, clip_model, clip_head, train_loader,val_loader,preprocess_clip,donor_features,disease_list)

    print("特征提取器预训练完成")




if __name__ == '__main__':
    data_disease = {
        'IDRiD':['Haemorrhages','Optic_Disc'],
        'MAPLES-DR':['Macula','Optic_Disc'],
        'MMAC': ['Fuchs_Spot', 'Optic_Disc'],
    }

    print(f"Starting ...")

    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    args = get_args()
    args.lesion_size=32
    args.N=0.5

    set_seed(2025)
    args.lesion_knowledge_path=f"/{args.dataSet}/lesion.json"
    args.csv_path = f"/{args.dataSet}/trainValTest/"

    args.output_path = f"/{args.dataSet}/output/{args.disease}/extractor"
    print(f"args.csv_path:{args.csv_path}")
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_features_extractor(args, device, data_disease)

    end_time = time.time()
    duration = end_time - start_time

    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    print("Training completed!")
    print(f"Ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training time: {hours}h {minutes}m {seconds}s")

