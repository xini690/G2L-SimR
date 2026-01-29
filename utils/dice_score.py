import torch
from torch import Tensor
import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def iou_coeff(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def clip_proj_infoNCE_loss(target_proj, donor_proj, temperature=0.07):

    sim_matrix = target_proj @ donor_proj.T
    sim_matrix /= temperature

    pos_idx = sim_matrix.argmax(dim=1)

    loss = F.cross_entropy(sim_matrix, pos_idx)
    return loss

def patch_cosine_loss(feat_img, feat_donor, temperature=0.07):
    feat_img = F.normalize(feat_img, dim=-1)
    feat_donor = F.normalize(feat_donor, dim=-1)

    logits = feat_img @ feat_donor.T / temperature
    labels = torch.arange(feat_img.size(0), device=feat_img.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def contrastive_patch_loss(pos_feats, neg_feats, temperature=0.1, eps=1e-8):

    if pos_feats is None or pos_feats.numel() == 0:
        return pos_feats.new_tensor(0.0)
    N_pos = pos_feats.size(0)
    if N_pos < 2:
        return pos_feats.new_tensor(0.0)

    pos_feats = F.normalize(pos_feats, dim=-1)
    neg_feats = F.normalize(neg_feats, dim=-1)


    sim_pos = pos_feats @ pos_feats.T / temperature
    sim_neg = pos_feats @ neg_feats.T / temperature

    mask = torch.eye(N_pos, device=pos_feats.device, dtype=torch.bool)
    sim_pos.masked_fill_(mask, -float('inf'))

    exp_pos = torch.exp(sim_pos)
    exp_neg = torch.exp(sim_neg)

    sum_pos = exp_pos.sum(dim=1)
    sum_neg = exp_neg.sum(dim=1)

    loss = -torch.log((sum_pos + eps) / (sum_pos + sum_neg + eps))
    return loss.mean()