import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms.functional as TF
from torchvision import models
l1_loss = nn.L1Loss()



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg.classifier = self.vgg.classifier[:-6]
       # checkpoint = torch.load('/run/media/mlcv/DATA_SSD/FaceFrontalization/VGGFace/best_checkpoint.pth',map_location='cpu')
      #  self.vgg.load_state_dict(checkpoint['model'])
        for param in self.vgg.parameters():
                param.requires_grad = False
        vgg_pretrained_features = self.vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        Xfeats = self.vgg(X)

        return out, Xfeats


class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()   

        self.device = device 
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, predictions, y):   
        
                
        self.vgg.eval()           
        x_vgg, x_feats = self.vgg(predictions)
        y_vgg, y_feats = self.vgg(y)

        lossvgg = 0
        for i in range(len(x_vgg)):
           lossvgg += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())      
        
        return lossvgg

class WeightedCrossEntropy(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index
        )

    def forward(self, logits, target):
        return self.loss(logits, target)

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        # 1. Softmax
        probs = torch.softmax(logits, dim=1)
        B, C, H, W = probs.shape

        # 2. Create the Mask (This must be done on the same device)
        # Ensure mask is a Float/Byte tensor to avoid logical errors
        mask = (target != self.ignore_index).float() 

        # 3. CRITICAL: Clamp every single value to [0, C-1]
        # Even if values are 255 or -1, they become 0 here for the One-Hot step.
        # The 'mask' will ensure these fake 0s don't affect the final loss.
        safe_target = torch.clamp(target, 0, C - 1).long()

        # 4. Generate One-Hot
        # Using scatter_ is safer than F.one_hot for CUDA stability
        target_onehot = torch.zeros(B, C, H, W, device=logits.device, dtype=logits.dtype)
        target_onehot.scatter_(1, safe_target.unsqueeze(1), 1.0)

        # 5. Apply the mask to remove the 'clamped' pixels from calculations
        mask_4d = mask.unsqueeze(1)
        probs = probs * mask_4d
        target_onehot = target_onehot * mask_4d

        # 6. Dice Math
        intersection = torch.sum(probs * target_onehot, dim=(0, 2, 3))
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(target_onehot, dim=(0, 2, 3))

        dice = (2. * intersection + self.eps) / (union + self.eps)

        return 1.0 - dice.mean()


def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_1h = F.one_hot(targets, logits.shape[1]).permute(0,3,1,2)
    inter = (probs * targets_1h).sum(dim=(0,2,3))
    union = (probs + targets_1h).sum(dim=(0,2,3))
    dice = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def dice_loss_ignore(logits, targets, ignore_index=255, smooth=1e-6):
    probs = torch.softmax(logits, dim=1)
    C = logits.shape[1]

    valid_mask = (targets != ignore_index)
    targets = targets.clone()
    targets[~valid_mask] = 0

    targets_1h = F.one_hot(targets, C).permute(0,3,1,2).float()
    valid_mask = valid_mask.unsqueeze(1)

    probs = probs * valid_mask
    targets_1h = targets_1h * valid_mask

    inter = (probs * targets_1h).sum(dim=(0,2,3))
    union = (probs + targets_1h).sum(dim=(0,2,3))

    dice = (2 * inter + smooth) / (union + smooth)

    # 🔑 ignore classes absent in GT
    mask = union > 0
    dice = dice[mask]

    return 1 - dice.mean()

def l1_loss_ignore(logits, targets, ignore_index=255):
    probs = torch.softmax(logits, dim=1)
    C = logits.shape[1]

    valid_mask = (targets != ignore_index)
    targets = targets.clone()
    targets[~valid_mask] = 0

    targets_1h = F.one_hot(targets, C).permute(0,3,1,2).float()
    valid_mask = valid_mask.unsqueeze(1)

    loss = torch.abs(probs - targets_1h)
    loss = loss * valid_mask

    return loss.sum() / valid_mask.sum().clamp_min(1)

def L1_loss(logits, targets):
    #probs = torch.softmax(logits, dim=1)
    targets_1h = F.one_hot(targets, logits.shape[1]).permute(0,3,1,2)
    loss = l1_loss(logits, targets_1h)
    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # This is your cityscapes_weights tensor
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # 1. Calculate standard CE (reduction="none" to keep pixel-wise)
        ce = F.cross_entropy(
            logits, target,
            reduction="none",
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:
            # Create a mask for valid pixels
            mask = target != self.ignore_index
            
            # Clamp targets to avoid index error at 255
            # These values at 255 will be masked out anyway
            safe_target = target.clone()
            safe_target[~mask] = 0 
            
            # Apply class weights
            alpha_t = self.alpha[safe_target]
            focal = alpha_t * focal
            
            # Mask the focal loss so ignore_index pixels are 0
            focal = focal * mask.float()

        # Return mean of non-ignored pixels only
        return focal.sum() / (target != self.ignore_index).sum().clamp(min=1)



# https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels):
    losses = []
    C = probs.size(1)
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.mean(torch.stack(losses))


class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        mask = target != self.ignore_index
        probs = probs.permute(0, 2, 3, 1)[mask]
        target = target[mask]
        return lovasz_softmax_flat(probs, target)


class CELovaszLoss(nn.Module):
    def __init__(self, ce_weight=1.0, lovasz_weight=1.0):
        super().__init__()
        self.ce = WeightedCrossEntropy()
        self.lovasz = LovaszSoftmax()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, target):

        ce_loss = self.ce(logits, target)
        lovasz_loss = self.lovasz(logits, target)
        return (
            self.ce_weight * ce_loss ,  self.lovasz_weight * lovasz_loss
        )

class CELovaszLossWeighted(nn.Module):
    def __init__(self, ce_weight=1.0, lovasz_weight=1.0, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        self.ce = WeightedCrossEntropy(weight=self.class_weights)
        self.lovasz = LovaszSoftmax()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        
    def forward(self, logits, target):

        ce_loss = self.ce(logits, target)
        lovasz_loss = self.lovasz(logits, target)
        return (
            self.ce_weight * ce_loss ,  self.lovasz_weight * lovasz_loss
        )

class CityscapesLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255
        )
        self.dice = DiceLoss()
        self.lovasz = LovaszSoftmax(ignore_index=255)
        
    def forward(self, logits, targets, use_lovasz=True):
        loss_ce = self.ce(logits, targets)
        loss_dice= 0.5 * self.dice(logits, targets)

        if use_lovasz:
            loss_lovasz= 0.4 * self.lovasz(
                torch.softmax(logits, dim=1), targets
            )
        return loss_ce, loss_dice, loss_lovasz




class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, target):
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)
        return focal_loss, dice_loss



class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_d = dice_loss(logits, targets)
        loss_l1 = L1_loss(logits, targets)
        return loss_ce, loss_d, loss_l1

class SegLossIgnore(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255):
        """
        Args:
            class_weights (Tensor, optional): A weight tensor of shape [num_classes].
            ignore_index (int): The label value that is ignored (not penalized).
        """
        super().__init__()
        
        # We initialize the CE loss. 
        # The weight will be moved to the correct GPU automatically 
        # when you call .to(DEVICE) on this SegLossIgnore object.
      #  self.ce = nn.CrossEntropyLoss(
       #     weight=class_weights, 
       #     ignore_index=ignore_index
        #)

        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )


    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_d = dice_loss_ignore(logits, targets, ignore_index=255)
       # loss_l1 = l1_loss_ignore(logits, targets, ignore_index=255)
        return loss_ce, loss_d