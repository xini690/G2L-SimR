import clip
import numpy as np
from torch import nn
from torchvision.transforms.functional import to_pil_image

import torch

import torch.nn.functional as F

import torchvision.transforms as T

from utils.data_loading import paste_lesion_semantic_vit
from utils.dice_score import clip_proj_infoNCE_loss, patch_cosine_loss, contrastive_patch_loss

@torch.no_grad()
def evaluate_clip_head(args,val_loader, donor_features, clip_model, clip_head, device, preprocess_clip):

    clip_head.eval()
    clip_model.eval()

    total_loss = 0.0
    total_samples = 0
    alpha = 0.5
    for batch in val_loader:
        images = batch['image'].to(device)
        batch_indices = batch['index'].tolist()
        text_tokens = clip.tokenize(batch['knowledge'],truncate=True).to(device)

        imgs_clip_pre = torch.stack([preprocess_clip(to_pil_image(img.cpu())).to(device) for img in images])
        img_feats = F.normalize(clip_model.encode_image(imgs_clip_pre).float(), dim=-1)
        text_feats = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)

        fused_feats = F.normalize(alpha * img_feats + (1 - alpha) * text_feats, dim=-1)
        target_proj = F.normalize(clip_head(fused_feats), dim=-1)

        for i, idx in enumerate(batch_indices):
            donor_feats_tensor = torch.stack([
                F.normalize(
                    alpha * d['image_feat'].to(device).float() + (1 - alpha) * d['text_feat'].to(device).float(),
                    dim=-1)
                for d in donor_features if d['index'] != idx
            ])
            donor_proj_all = F.normalize(clip_head(donor_feats_tensor), dim=-1)

            sim = F.cosine_similarity(target_proj[i:i + 1], donor_proj_all)
            sorted_indices = torch.argsort(sim, descending=True)
            donor_proj_sorted = donor_proj_all[sorted_indices]

            num_regions = donor_proj_all.size(0)
            N=args.N
            num_pos = max(1, int(num_regions * N))


            indices = torch.randperm(donor_proj_all.size(0))
            pos_indices = indices[:num_pos]
            neg_indices = indices[num_pos:]

            pos_regions = donor_proj_all[pos_indices]
            neg_regions = donor_proj_all[neg_indices]

            min_count = min(len(pos_regions), len(neg_regions))
            if min_count > 0:
                pos_regions = pos_regions[:min_count]
                neg_regions = neg_regions[:min_count]

                loss_i = contrastive_patch_loss(pos_regions, neg_regions)
                total_loss += loss_i.item()
                total_samples += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


@torch.no_grad()
def evaluate_cnn_vit(epoch, cnn_model, vit_model, clip_model, clip_head, preprocess_clip,
             val_loader, device, args, donor_features, disease_list, data_disease):
    cnn_model.eval()
    vit_model.eval()

    vit_loss_epoch = 0.0
    cnn_loss_epoch = 0.0
    total_samples = 0
    total_vit_pairs = 0
    total_cnn_pairs = 0

    for batch in val_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        batch_size = images.size(0)

        for i in range(batch_size):
            vit_patches_pos, vit_patches_neg = [], []
            cnn_patches_pos, cnn_patches_neg = [], []

            img_clip_pre = preprocess_clip(to_pil_image(images[i].cpu())).to(device).unsqueeze(0)
            img_feat = F.normalize(clip_model.encode_image(img_clip_pre).float(), dim=-1)

            text_tokens = clip.tokenize([batch['knowledge'][i]],truncate=True).to(device)
            text_feat = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)
            alpha=0.5
            fused_feat = F.normalize(alpha * img_feat + (1-alpha) * text_feat, dim=-1)
            target_proj = clip_head(fused_feat)

            current_idx = batch['index'][i].item()

            donor_feats = []
            donor_indices = []

            for idx, d in enumerate(donor_features):
                if d['index'] == current_idx:
                    continue
                alpha_val = alpha
                fused_feat = alpha_val * d['image_feat'].to(device).float() + (1 - alpha_val) * d['text_feat'].to(
                    device).float()
                donor_feats.append(F.normalize(fused_feat, dim=-1))
                donor_indices.append(idx)

            if len(donor_feats) > 0:
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
            else:
                donor = None

            if donor is None:
                continue
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
                use_vit = res[1]
                num_regions = len(sorted_regions)

                N=args.N
                num_pos = max(1, int(num_regions * N))

                indices = torch.randperm(num_regions)
                pos_indices = indices[:num_pos]
                neg_indices = indices[num_pos:]

                pos_regions = [sorted_regions[i] for i in pos_indices]
                neg_regions = [sorted_regions[i] for i in neg_indices]

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
                vit_pos_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(224, 224),
                                                            mode='bilinear', align_corners=False).squeeze(0)
                                              for p in vit_patches_pos]).to(device)
                vit_neg_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(224, 224),
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

                    pair_count = min(len(pos_batch), len(neg_batch))
                    vit_loss_epoch += contrastive_patch_loss(pos_feats, neg_feats).item() * pair_count
                    total_vit_pairs += pair_count

            if len(cnn_patches_pos) > 0 and len(cnn_patches_neg) > 0:
                image_size=64
                cnn_pos_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(image_size, image_size),
                                                            mode='bilinear', align_corners=False).squeeze(0)
                                              for p in cnn_patches_pos]).to(device)
                cnn_neg_tensor = torch.stack([F.interpolate(p.unsqueeze(0), size=(image_size, image_size),
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

                    pair_count = min(len(pos_batch), len(neg_batch))
                    cnn_loss_epoch += contrastive_patch_loss(pos_feats, neg_feats).item() * pair_count
                    total_cnn_pairs += pair_count

        total_samples += batch_size


    avg_vit_loss = vit_loss_epoch / max(1, total_vit_pairs)
    avg_cnn_loss = cnn_loss_epoch / max(1, total_cnn_pairs)

    avg_loss = (avg_vit_loss + avg_cnn_loss)/2

    return {
        "total_loss": avg_loss,
        "cnn_loss": avg_cnn_loss,
        "vit_loss": avg_vit_loss
    }
