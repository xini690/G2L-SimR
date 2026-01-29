import time

import clip
import pandas as pd
import torch.multiprocessing as mp
from torchvision.transforms.functional import to_pil_image

from unet.G2L_model import G2L
from utils.clip_head import set_seed, CLIPProjectionHead
from utils.patchGAN import  FullResDiscriminatorText

mp.set_start_method("spawn", force=True)
import argparse
import ast
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import yaml
from torch import optim
from torch.utils.data import DataLoader, random_split

from evaluate import evaluateGLS

from utils.data_loading import BasicDataset, build_dataset
from utils.dice_score import dice_loss

import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def append_test_results_to_excel(args, test_results, excel_file_path):
    new_row = {
        'Model': args.model_name,
        'Disease': args.disease,
        'Seed': args.seed,
        'Dice':  round(test_results['dice'], 4),
        'IoU': round(test_results['iou'], 4),
        'P_Recall': round(test_results['pixel_recall'], 4),
        'L_Recall': round(test_results['lesion_recall'],4),
        'AvgFP': round(test_results['avg_fp_per_image'],4),
        'ASD':round(test_results['asd'],4),
        'HD95':round(test_results['hd95'],4),
        'AUPR': round(test_results['aupr'],4)
    }

    if os.path.exists(excel_file_path):
        existing_df = pd.read_excel(excel_file_path)
        new_df = pd.DataFrame([new_row])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame([new_row])

    combined_df.to_excel(excel_file_path, index=False)

def train_textFusionUNet(args,epochs,
                         train_loader,
                         val_loader,
                         test_loader,
                         unet_model,
                         unet_optimizer,
                         unet_discriminator,
                         d_optimizer,
                         grad_scaler,
                         criterion,
                         amp,
                         gradient_clipping,
                         learning_rate,
                         save_checkpoint=True,
                         lambda_adv=0.01,
                         update_every=20
                         ):


    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device, jit=False)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_head = CLIPProjectionHead(embed_dim=clip_model.visual.output_dim, proj_dim=512)
    clip_head.to(device)

    if os.path.exists(args.clip_head_path):
        clip_head.load_state_dict(torch.load(args.clip_head_path))
        clip_head = clip_head.to(device)
    clip_head_optimizer = optim.Adam(clip_head.parameters(), lr=learning_rate)

    bce_loss = nn.BCELoss()
    adv_loss = torch.tensor(0.0, device=device)
    best_score = 0.0
    for epoch in range(1, epochs + 1):
        unet_model.train()
        clip_head.train()
        if unet_discriminator is not None:
            unet_discriminator.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = batch['mask'].to(device, dtype=torch.long)
            knowledge = batch['knowledge']

            text_tokens = clip.tokenize(knowledge,truncate=True).to(device)
            imgs_clip_pre = torch.stack([preprocess_clip(to_pil_image(img.cpu())).to(device) for img in images])

            with torch.no_grad():
                text_feats = clip_model.encode_text(text_tokens)
                image_feats = clip_model.encode_image(imgs_clip_pre)

            text_feats = F.normalize(text_feats.float(), dim=-1)
            image_feats = F.normalize(image_feats.float(), dim=-1)
            alpha = 0.25
            fused_feats = F.normalize(alpha * image_feats + (1 - alpha) * text_feats, dim=-1)

            with torch.no_grad():
                clip_feats_for_unet = clip_head(fused_feats)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = unet_model(images, clip_feats_for_unet)

                if args.classes == 1:
                    true_mask = true_masks.float().unsqueeze(1)
                    seg_loss = criterion(masks_pred, true_mask) + dice_loss(
                        torch.sigmoid(masks_pred).squeeze(1), true_mask.squeeze(1))
                    pred_mask_for_D = torch.sigmoid(masks_pred)
                else:
                    true_mask_1hot = F.one_hot(true_masks, args.classes).permute(0, 3, 1, 2).float()
                    seg_loss = criterion(masks_pred, true_masks) + dice_loss(
                        F.softmax(masks_pred, dim=1), true_mask_1hot, multiclass=True)
                    pred_mask_for_D = F.softmax(masks_pred, dim=1)[:, :1, :, :]

                if unet_discriminator is not None:
                    real_in = torch.cat([images, true_masks.float().unsqueeze(1)], dim=1)
                    fake_in = torch.cat([images, pred_mask_for_D], dim=1)
                    d_real = unet_discriminator(real_in,text_feats)
                    d_fake = unet_discriminator(fake_in,text_feats)
                    adv_loss = (bce_loss(d_real, torch.ones_like(d_real)) +
                              bce_loss(d_fake, torch.zeros_like(d_fake))) / 2
                else:
                    adv_loss = torch.tensor(0.0, device=device)

                unet_loss = seg_loss + lambda_adv * adv_loss

            unet_optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(unet_loss).backward()
            grad_scaler.unscale_(unet_optimizer)
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), gradient_clipping)
            grad_scaler.step(unet_optimizer)
            grad_scaler.update()

            for p in unet_model.parameters():
                p.requires_grad = False

            clip_feats = clip_head(fused_feats)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred_clip = unet_model(images, clip_feats)

                if args.classes == 1:
                    true_mask = true_masks.float().unsqueeze(1)
                    seg_loss_clip = criterion(masks_pred_clip, true_mask) + dice_loss(
                        torch.sigmoid(masks_pred_clip).squeeze(1), true_mask.squeeze(1))
                else:
                    true_mask_1hot = F.one_hot(true_masks, args.classes).permute(0, 3, 1, 2).float()
                    seg_loss_clip = criterion(masks_pred_clip, true_masks) + dice_loss(
                        F.softmax(masks_pred_clip, dim=1), true_mask_1hot, multiclass=True)

            clip_head_optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(seg_loss_clip).backward()
            grad_scaler.unscale_(clip_head_optimizer)
            torch.nn.utils.clip_grad_norm_(clip_head.parameters(), gradient_clipping)
            grad_scaler.step(clip_head_optimizer)
            grad_scaler.update()

            for p in unet_model.parameters():
                p.requires_grad = True

            if unet_discriminator is not None:
                unet_discriminator.zero_grad()
                real_in = torch.cat([images, true_masks.float().unsqueeze(1)], dim=1)
                fake_in = torch.cat([images, pred_mask_for_D.detach()], dim=1)
                d_real = unet_discriminator(real_in,text_feats)
                d_fake = unet_discriminator(fake_in,text_feats)
                d_loss = (bce_loss(d_real, torch.ones_like(d_real)) +
                          bce_loss(d_fake, torch.zeros_like(d_fake))) / 2
                d_loss.backward()
                d_optimizer.step()

            epoch_loss += unet_loss.item()
            if (i + 1) % update_every == 0 or (i + 1) == len(train_loader):
                print(f'Epoch [{epoch}/{epochs}], Batch [{i + 1}/{len(train_loader)}], '
                      f'Loss: {unet_loss.item():.4f}, Seg: {seg_loss.item():.4f}, Adv: {adv_loss.item():.4f}')

        results = evaluateGLS(
            args, unet_model, unet_discriminator, val_loader,
            clip_head, clip_model, preprocess_clip, device, amp
        )
        val_dice=results['dice']
        val_iou = results['iou']
        print(f'Epoch [{epoch}/{epochs}] Val Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')

        if save_checkpoint and val_dice >= best_score:
            best_score = val_dice
            torch.save(unet_model.state_dict(), os.path.join(args.output_path, 'best_segModel.pth'))
            torch.save(clip_head.state_dict(), os.path.join(args.output_path, 'best_clip_head.pth'))
            if unet_discriminator is not None:
                torch.save(unet_discriminator.state_dict(), os.path.join(args.output_path, 'best_discriminator.pth'))

    unet_model.load_state_dict(torch.load(os.path.join(args.output_path, 'best_segModel.pth')))
    clip_head.load_state_dict(torch.load(os.path.join(args.output_path, 'best_clip_head.pth')))
    if unet_discriminator is not None:
        unet_discriminator.load_state_dict(torch.load(os.path.join(args.output_path, 'best_discriminator.pth')))
    unet_model.to(device)
    clip_head.to(device)
    if unet_discriminator is not None:
        unet_discriminator.to(device)
    results= evaluateGLS(args, unet_model, unet_discriminator,test_loader,clip_head, clip_model, preprocess_clip, device, amp)
    test_dice, test_iou = results['dice'], results['iou']
    args.excel_file_path = os.path.join(args.output_path, 'test_results.xlsx')
    append_test_results_to_excel(args, results, args.excel_file_path)
    print(f"Test Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")



def train_model(
        args,
        unet_model,
        device,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        gradient_clipping: float = 1.0,
        lambda_adv: float = 0.01
):

    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset_train, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(dataset_test, shuffle=False, drop_last=True, **loader_args)

    unet_optimizer = optim.RMSprop(unet_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss() if args.classes > 1 else nn.BCEWithLogitsLoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    unet_discriminator = FullResDiscriminatorText(in_channels=4).to(device)
    d_optimizer = optim.Adam(unet_discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    train_textFusionUNet(args,epochs,
                         train_loader,
                         val_loader,
                         test_loader,
                         unet_model,
                         unet_optimizer,
                         unet_discriminator,
                         d_optimizer,
                         grad_scaler,
                         criterion,
                         amp,
                         gradient_clipping,
                         learning_rate,
                         save_checkpoint=True,
                         update_every=20)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--unet_path', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--semantics', type=str, help='Load model from a .pth file')
    parser.add_argument('--csv_path', type=str, help='Load model from a .pth file')
    parser.add_argument('--output_path', type=str, help='Load model from a .pth file')
    parser.add_argument('--vit_path', type=str, default='/vit_path',help='Load model from a .pth file')
    parser.add_argument('--cnn_path', type=str,default='/cnn_path',help='Load model from a .pth file')
    parser.add_argument('--clip_head_path', type=str,default='/clip_head_path',help='Load model from a .pth file')
    parser.add_argument('--lesion_knowledge_path', type=str, help='Load model from a .pth file')
    parser.add_argument('--image_knowledge_path', type=str, help='Load model from a .pth file')
    parser.add_argument('--dataSet', type=str,  help='Load model from a .pth file')
    parser.add_argument('--aug_lesion_model',  action='store_true', help='Load model from a .pth file')
    parser.add_argument('--aug_clip_disc', action='store_true', help='Load model from a .pth file')
    parser.add_argument('--disease', type=str, default='unet', help='Load model from a .pth file')
    parser.add_argument('--textFusionUNet',  action='store_true', help='Load model from a .pth file')
    parser.add_argument('--base_path', type=str, help='Load model from a .pth file')
    parser.add_argument('--ablate', type=str, help='Load model from a .pth file')
    parser.add_argument('--model_name', type=str, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--unet_discriminator', action='store_true', help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--ablation', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--aug_factor', type=float, default=0.5, help='Use bilinear upsampling')
    parser.add_argument('--seed', type=int, default=42, help='Number of classes')
    parser.add_argument('--lesion_size', type=int, default=32, help='Number of classes')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--max_lesions', type=int, default=3, help='Number of classes')
    parser.add_argument('--augment_mode', type=str, default='random', help='Load model from a .pth file')
    parser.add_argument('--dataSet_mode', type=str, default='base', help='Load model from a .pth file')
    parser.add_argument('--config', default='../cfgs/synNetCfg.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    args_config, remaining = parser.parse_known_args()

    cfg = {}
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)

    parser.set_defaults(**cfg)

    args = parser.parse_args()

    for k, v in vars(args).items():
        if v is not None and v != cfg.get(k):
            setattr(args, k, v)

    return args

def train_target(args,unet_model,model_name,device):
    print(f"{model_name}:Training started...")
    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    train_model(
        args=args,
        unet_model=unet_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        amp=args.amp
    )

    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"\n{model_name} training completed!")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training duration: {hours}h {minutes}m {seconds}s")
    print("=========================================")

    del unet_model
    torch.cuda.empty_cache()
    print(f"Released the GPU memory occupied by {model_name}.")
    time.sleep(60)


if __name__ == '__main__':


    args = get_args()
    print(args)
    dataSet=args.dataSet
    args.model_name='TextFusionUNet'
    args.seed = 42
    args.lesion_size = 32
    args.aug_prob = 2.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    args.vit_path = f"/output/extractor/vit_model.pth"
    args.cnn_path = f"/output/extractor/cnn_model.pth"
    args.clip_head_path = f"/output/extractor/clip_model.pth"
    args.lesion_knowledge_path = f"/dataSet/{args.dataSet}/images.json"
    args.image_knowledge_path = f"/dataSet/{args.dataSet}/lesion.json"
    args.csv_path = f"/{args.dataSet}/{args.disease}/trainValTest/"

    model_name = args.model_name
    set_seed(args.seed)
    args.output_path = f"{args.output_path}/{args.disease}/"

    print(args.output_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = G2L(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            fusion_mode="attention"
        ).to(device)

    train_target(args, model, model_name, device)



