import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

import cv2
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryRecall, BinaryAveragePrecision
from scipy.spatial.distance import cdist

def compute_hd95(gt, pred):
    gt_pts = np.argwhere(gt > 0)
    pred_pts = np.argwhere(pred > 0)
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan
    dist_matrix = cdist(gt_pts, pred_pts)
    hd1 = np.percentile(dist_matrix.min(axis=1), 95)
    hd2 = np.percentile(dist_matrix.min(axis=0), 95)
    return max(hd1, hd2)

def compute_asd(gt, pred):
    gt_pts = np.argwhere(gt > 0)
    pred_pts = np.argwhere(pred > 0)
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan
    dist_matrix = cdist(gt_pts, pred_pts)
    asd1 = dist_matrix.min(axis=1).mean()
    asd2 = dist_matrix.min(axis=0).mean()
    return (asd1 + asd2) / 2

@torch.inference_mode()
def evaluateGLS(args, unet_model, unet_discriminator, dataloader,
                      clip_head, clip_model, preprocess_clip, device, amp):

    unet_model.eval()
    clip_head.eval()
    if unet_discriminator is not None:
        unet_discriminator.eval()

    dice_metric = BinaryF1Score().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    pixel_recall_metric = BinaryRecall().to(device)

    aupr_metric = BinaryAveragePrecision().to(device)

    total_lesions = 0
    detected_lesions = 0
    total_fp = 0
    num_images = 0

    hd95_list = []
    asd_list = []

    print("Evaluating model...")

    with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:

            images, mask_trues, knowledge = batch['image'], batch['mask'], batch['knowledge']
            image_paths= batch['image_path']
            images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_trues = mask_trues.to(device, dtype=torch.long)
            if mask_trues.dim() == 4:
                mask_trues = mask_trues.squeeze(1)

            text_tokens = clip.tokenize(knowledge,truncate=True).to(device)
            imgs_clip_pre = torch.stack([preprocess_clip(to_pil_image(img.cpu())).to(device) for img in images])

            text_feats = clip_model.encode_text(text_tokens)
            images_feats = clip_model.encode_image(imgs_clip_pre)
            text_feats = F.normalize(text_feats.float(), dim=-1)
            images_feats = F.normalize(images_feats.float(), dim=-1)

            fused_feats = F.normalize(0.25 * images_feats + 0.75 * text_feats, dim=-1)
            clip_feats = clip_head(fused_feats)

            mask_pred = unet_model(images, clip_feats)

            if args.classes == 1:
                pred_prob = torch.sigmoid(mask_pred)
                mask_pred_bin = (pred_prob > 0.5).float().squeeze(1)

                dice_metric.update(mask_pred_bin, mask_trues.float())
                iou_metric.update(mask_pred_bin.unsqueeze(1), mask_trues.unsqueeze(1))
                pixel_recall_metric.update(mask_pred_bin, mask_trues.float())

                aupr_metric.update(pred_prob.detach().view(-1), mask_trues.view(-1).long())
            else:
                mask_true_1hot = F.one_hot(mask_trues, args.classes).permute(0, 3, 1, 2).float()
                mask_pred_1hot = F.one_hot(mask_pred.argmax(dim=1), args.classes).permute(0, 3, 1, 2).float()
                dice_metric.update(mask_pred_1hot[:, 1:], mask_true_1hot[:, 1:])
                iou_metric.update(mask_pred_1hot[:, 1:], mask_true_1hot[:, 1:])
                pixel_recall_metric.update(mask_pred_1hot[:, 1:], mask_true_1hot[:, 1:])

            save_pred_masks(mask_pred_bin, image_paths, args.output_path)

            mask_true_np = mask_trues.cpu().numpy()
            mask_pred_np = (
                mask_pred_bin.cpu().numpy()
                if args.classes == 1
                else mask_pred_1hot[:, 1:, :, :].sum(1).cpu().numpy()
            )

            B = mask_true_np.shape[0]
            num_images += B

            for b in range(B):
                true_mask = mask_true_np[b].astype(np.uint8)
                pred_mask = mask_pred_np[b].astype(np.uint8)

                true_mask_bin = (true_mask > 0).astype(np.uint8)
                pred_mask_bin = (pred_mask > 0).astype(np.uint8)

                hd95_list.append(compute_hd95(true_mask_bin, pred_mask_bin))
                asd_list.append(compute_asd(true_mask_bin, pred_mask_bin))

                num_labels_true, labels_true = cv2.connectedComponents(true_mask_bin)
                num_labels_pred, labels_pred = cv2.connectedComponents(pred_mask_bin)

                for label_idx in range(1, num_labels_true):
                    lesion = (labels_true == label_idx).astype(np.uint8)
                    if (lesion & pred_mask_bin).sum() > 0:
                        detected_lesions += 1
                total_lesions += max(num_labels_true - 1, 0)

                min_area = 5
                for label_idx in range(1, num_labels_pred):
                    pred_lesion = (labels_pred == label_idx).astype(np.uint8)
                    if pred_lesion.sum() < min_area:
                        continue
                    overlaps = [(pred_lesion & (labels_true == t)).sum() for t in range(1, num_labels_true)]
                    if len(overlaps) == 0 or max(overlaps) == 0:
                        total_fp += 1

    dice = float(dice_metric.compute().item())
    iou = float(iou_metric.compute().item())
    pixel_recall = float(pixel_recall_metric.compute().item())
    lesion_recall = detected_lesions / max(total_lesions, 1)
    avg_fp_per_image = total_fp / max(num_images, 1)
    HD95 = float(np.nanmean(hd95_list))
    ASD = float(np.nanmean(asd_list))

    aupr = float(aupr_metric.compute().item()) if args.classes == 1 else None

    print("\n===== Small Lesion Evaluation =====")
    print(f"Pixel-wise Dice:       {dice:.4f}")
    print(f"Pixel-wise IoU:        {iou:.4f}")
    print(f"Pixel-wise Recall:     {pixel_recall:.4f}")
    print(f"Lesion-wise Recall:    {lesion_recall:.4f}")
    print(f"Average FP per image:  {avg_fp_per_image:.2f}")
    print(f"ASD (ASSD):            {ASD:.4f}")
    print(f"HD95:                  {HD95:.4f}")
    print(f"AUPR:                  {aupr:.4f}")
    print("==================================\n")

    return {
        'dice': dice,
        'iou': iou,
        'pixel_recall': pixel_recall,
        'lesion_recall': lesion_recall,
        'avg_fp_per_image': avg_fp_per_image,
        'asd': ASD,
        'hd95': HD95,
        'aupr': aupr
    }

def save_pred_masks(masks_bin, image_paths, save_dir):
    save_dir=os.path.join(save_dir, 'pred_masks/')
    os.makedirs(save_dir, exist_ok=True)

    if masks_bin.dim() == 4:
        masks_bin = masks_bin.squeeze(1)

    masks_np = masks_bin.detach().cpu().numpy()

    for i, path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(save_dir, name + ".png")

        img = (masks_np[i] * 255).astype(np.uint8)
        cv2.imwrite(save_path, img)

