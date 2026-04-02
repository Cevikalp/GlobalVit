

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from myutils.git_models import Dinov2SegmentationModelLast
from myutils.data_augmentation import ADE20KSegmentation, get_mmseg_transforms_new, CityscapesSegmentation
from myutils.losses import SegLossIgnore, CELovaszLossWeighted, CityscapesLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os



print(torch.cuda.get_device_name(0))
print(torch.randn(1, device="cuda"))

# --- Configuration ---
IMG_SIZE = 560
PATCH_SIZE = 8
EMBED_DIM = 256
DEPTH = 6
NUM_HEADS = 8
BATCH_SIZE = 8
EPOCHS = 150
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 150
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 512  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

COCO_COLORS = [
    (0, 0, 0),        # 0 background (forced)

    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # blue
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0, 128, 128),    # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),   # brown
    (255, 250, 200),  # beige
    (128, 0, 0),      # maroon
    (170, 255, 195),  # mint
    (128, 128, 0),    # olive
    (255, 215, 180),  # coral
    (0, 0, 128),      # navy
    (128, 128, 128),  # gray

    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),

    (100, 149, 237),  # cornflower blue
    (255, 140, 0),    # dark orange
    (154, 205, 50),   # yellow green
    (75, 0, 130),     # indigo
    (199, 21, 133),   # medium violet red
    (244, 164, 96),   # sandy brown
    (46, 139, 87),    # sea green
    (210, 105, 30),   # chocolate
    (112, 128, 144),  # slate gray
    (0, 191, 255),    # deep sky blue
    (233, 150, 122),  # dark salmon
    (152, 251, 152),  # pale green
    (138, 43, 226),   # blue violet
    (255, 20, 147),   # deep pink
    (47, 79, 79),     # dark slate gray
    (139, 69, 19),    # saddle brown
    (0, 100, 0),      # dark green
    (72, 61, 139),    # dark slate blue
    (255, 228, 181),  # moccasin
    (176, 196, 222),  # light steel blue
    (95, 158, 160),   # cadet blue
    (255, 99, 71),    # tomato
    (60, 179, 113),   # medium sea green
]
# Function to visualize image and mask
def show_image_mask(image, mask):
    """
    image: torch tensor (3, H, W)
    mask: torch tensor (H, W)
    """
    image = image.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
    image = np.clip(image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)  # Denormalize
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(mask.numpy(), cmap='jet', interpolation='nearest')
    plt.title("Mask")
    plt.axis('off')
    
    plt.show()

def freeze_vit_encoder(model):
    """
    Freeze ViT encoder layers so only segmentation head is trained.
    """
    frozen, trainable = 0, 0

    for name, param in model.named_parameters():
        if any(k in name.lower() for k in [
            "patch_embed",
            "pos_embed",
            "blocks",
            "transformer",
            "encoder",
            "vit"
        ]):
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True
            trainable += 1

    print(f"[Freeze] Frozen params: {frozen}, Trainable params: {trainable}")

def get_ade20k_weights(dataloader, num_classes=150, ignore_index=255):
    print("Calculating class frequencies (this may take a few minutes)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = torch.zeros(num_classes).to(device)
    
    for _, masks in tqdm(dataloader):
        masks = masks.to(device)
        
        # Apply your specific label transformation logic
        # So the weights match the labels the model actually sees
        masks_calc = masks.clone()

        is_wall = (masks_calc == 0)

        # 3. Identify valid classes that need to be shifted (1 to 150)
        # We exclude 0 and we exclude the existing 255
        is_valid_class = (masks_calc > 0) & (masks_calc != 255)

        # 4. Perform the shift
        masks_calc[is_valid_class] -= 1

        # 5. Map the old 'Wall' pixels to the ignore index
        masks_calc[is_wall] = 255
       
        # After this, labels are 0-149, and original 0 is 255 (ignore)

        valid_mask = (masks_calc != ignore_index) & (masks_calc < num_classes)
        counts += torch.bincount(masks_calc[valid_mask].view(-1), minlength=num_classes)

    # Logarithmic Inverse Frequency Scaling
    probs = counts / counts.sum()
    # 1.02 is a constant to prevent weights from exploding for very rare classes
    weights = 1.0 / torch.log(1.02 + probs)
    
    # Normalize so that the mean of weights is 1.0
    weights = weights / weights.mean()
    
    return weights.cpu()

# -----------------------------
# Colorization helper
# -----------------------------
def colorize_mask(mask, num_classes, palette):
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)

    for c in range(num_classes):
        if c == 0:
            color = (0, 0, 0)  # background
        else:
            color = palette[c % len(palette)]
        color_mask[mask == c] = color

    # Ignore index
    color_mask[mask == 255] = (0, 0, 0)
    return color_mask


train_transform = get_mmseg_transforms_new(img_size=IMG_SIZE, split="train")
val_transform = get_mmseg_transforms_new(img_size=IMG_SIZE, split="val")

# Paths
# 2. Initialize Datasets with transforms
train_dataset = ADE20KSegmentation(
    root="/media/mlcv/Data/SegmentationProject/data/ADEChallengeData2016",
    split="training",
    transform=train_transform
)

val_dataset = ADE20KSegmentation(
    root="/media/mlcv/Data/SegmentationProject/data/ADEChallengeData2016",
    split="validation",
    transform=val_transform
)


# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Initialize counts
class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float)
from torch.optim.lr_scheduler import CosineAnnealingLR
# Iterate over dataset
# 1. Get Weights (Calculate once, then load from disk)
#class_weights = get_ade20k_weights(train_loader)
weight_path = "ade20k_weights.pt"
if os.path.exists(weight_path):
    print("Loading pre-calculated weights...")
    class_weights = torch.load(weight_path)
else:
    class_weights = get_ade20k_weights(train_loader)
    torch.save(class_weights, weight_path)

# --- Model, Loss, Optimizer ---

model = Dinov2SegmentationModelLast(
      num_classes =NUM_CLASSES,
      img_size=IMG_SIZE,
).to(DEVICE)


LOAD_MODEL = False
#loading trained models
if LOAD_MODEL:
    model.load_state_dict(torch.load("/media/mlcv/Data/SegmentationProject/outputs_model_ade20k_dino/model_epoch_best_full_transformer_last_sota.pth"))



#optimizer = optim.AdamW(model.parameters(), lr=LR)
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 0.5e-5},
    {
        "params": list(model.proj.parameters()) +
                  list(model.fpn.parameters()) +
                  list(model.feature_fusion.parameters()) +
                  [model.pos_embed] +
                  list(model.decoder.parameters()) +
                  list(model.up_and_merge.parameters()) +
                  list(model.post_skip_conv.parameters()) +
                  list(model.remaining_head.parameters()),
        "lr": LR,
    },
], weight_decay=0.05)
#criterion = CombinedLoss()
#criterion = SegLossIgnore(class_weights=class_weights, ignore_index=255).to(DEVICE)
criterion = CityscapesLoss(class_weights=class_weights).to(DEVICE)

# Add a scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

def logits_to_mask(logits):
    """
    Convert network output logits to a single predicted mask.
    
    Args:
        logits (torch.Tensor): Network output of shape [B, C, H, W]
                               where C = number of classes.

    Returns:
        mask (torch.Tensor): Predicted mask of shape [B, H, W] with
                             integer class IDs per pixel (0..C-1)
    """
    # Apply argmax over the class dimension
    mask = torch.argmax(logits, dim=1)  # [B, H, W]

    return mask




def visualize_debug_results(image_tensor, gt_mask, pred_mask, num_classes, palette):
    """
    Standalone debug function to visualize consistency.
    """
    # 1. Denormalize/Convert Image for display
    # Assumes image is normalized (0-1) or standard ViT normalization
    img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # 2. Colorize using your existing function
    # We pass gt_mask and pred_mask through colorize_mask
    # NOTE: I added a check for 256 inside the function below
    color_gt = colorize_mask_debug(gt_mask, num_classes, palette)
    color_pred = colorize_mask_debug(pred_mask, num_classes, palette)

    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    
    axes[1].imshow(color_gt)
    axes[1].set_title("Ground Truth (Shifted)")
    
    axes[2].imshow(color_pred)
    axes[2].set_title("Model Prediction")

    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def colorize_mask_debug(mask, num_classes, palette):
    """
    Updated version of your function to handle the 256 ignore index.
    """
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
        
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)

    for c in range(num_classes):
        # Using your palette logic
        color = palette[c % len(palette)]
        color_mask[mask == c] = color

    # Handle both potential ignore indices as BLACK
    color_mask[mask == 255] = (0, 0, 0)
    color_mask[mask == 256] = (0, 0, 0) 
    
    return color_mask

# -----------------------------
# Validation function
# -----------------------------
def validate_coco_segmentation(
    model,
    dataset,
    save_dir,
    device="cuda",
    batch_size=4,
    num_classes=NUM_CLASSES,
    max_batches=50,
    max_images=100,
):
    """
    Validate a semantic segmentation model and save visualizations.
    """

    os.makedirs(save_dir, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    total_pixels = 0
    correct_pixels = 0
    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    saved_images = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validating")):
            images = images.to(device)
            masks = masks.to(device)
            valid_classes = (masks > 0) & (masks != 255)
            is_wall = (masks == 0)
            masks[valid_classes] -= 1
            masks[is_wall] = 255

                       # Forward
            logits = model(images)              # [B, C, H, W]
            preds = logits.argmax(dim=1)        # [B, H, W]

            # -----------------------------
            # Metrics (ignore index = 255)
            # -----------------------------
            valid = masks != 255

            i=0
            gt_mask_np = masks[i].cpu().numpy()
            pred_mask_np = preds[i].cpu().numpy()
           # visualize_debug_results(images[i], gt_mask_np, pred_mask_np, num_classes, COCO_COLORS)

            correct_pixels += ((preds == masks) & valid).sum().item()
            total_pixels += valid.sum().item()

            for c in range(num_classes):
                pred_c = (preds == c) & valid
                mask_c = (masks == c)

                inter = (pred_c & mask_c).sum().item()
                u = (pred_c | mask_c).sum().item()

                intersection[c] += inter
                union[c] += u

            # -----------------------------
            # Save visualizations
            # -----------------------------
            for i in range(images.size(0)):
                if saved_images >= max_images:
                    break

                # Input image
                img_np = images[i].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)

                # Masks
                gt_mask = masks[i].cpu().numpy().astype(np.int32)
                pred_mask = preds[i].cpu().numpy().astype(np.int32)

                color_gt = colorize_mask(gt_mask, num_classes, COCO_COLORS)
                color_pred = colorize_mask(pred_mask, num_classes, COCO_COLORS)

                img_pil.save(os.path.join(save_dir, f"val_{batch_idx}_{i}_input.png"))
                Image.fromarray(color_gt).save(
                    os.path.join(save_dir, f"val_{batch_idx}_{i}_gt.png")
                )
                Image.fromarray(color_pred).save(
                    os.path.join(save_dir, f"val_{batch_idx}_{i}_pred.png")
                )

                saved_images += 1

           # if batch_idx >= max_batches:
             #   break

    # -----------------------------
    # Final metrics
    # -----------------------------
    pixel_acc = correct_pixels / (total_pixels + 1e-6)
    iou = intersection / (union + 1e-6)
    mean_iou = np.nanmean(iou)

    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU:       {mean_iou:.4f}")

    return pixel_acc, mean_iou


def train_one_epoch(epoch, train_loader, scheduler=None):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #show_image_mask(inputs[0], targets[0])
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        with torch.no_grad():
            targets = targets.clone()
            # ADE20K: 0 is Wall (often ignored), 1-150 are classes.
            # We want 1-150 -> 0-149 and 0 -> 255
            is_wall = (targets == 0)
            valid_classes = (targets > 0) & (targets != 255)
            targets[valid_classes] -= 1
            targets[is_wall] = 255
        
        optimizer.zero_grad()
        outputs = model(inputs)
        assert outputs.shape[1] == NUM_CLASSES
        assert targets.min() >= 0
        assert (targets < NUM_CLASSES).logical_or(targets == 255).all()
        loss_ce, loss_dice, loss_lova = criterion(outputs, targets)
        loss = loss_ce + loss_dice + loss_lova

       # refined_outputs = refine_module(outputs)

        
        #loss = 10*criterion(refined_outputs, targets) + perceptual_loss(refined_outputs, targets) 
        loss.backward()
        optimizer.step()

        running_loss += loss
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, CE: {loss_ce.item():.4f}, Dice: {loss_dice.item():.4f}") 

    
    if scheduler:
        scheduler.step()

    return running_loss / len(train_loader)




def main():
# --- Training Loop ---
    output_dir = Path("outputs_model_ade20k_dino/")
    output_dir.mkdir(exist_ok=True)
   
    #mask_feats, all_mask_label_inter, all_mask_label_intra = load_VGGmaskfeats(args,device)
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        #model.eval()
        
                
        train_loss = train_one_epoch(epoch, train_loader,scheduler=scheduler)
       
       # print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        val_acc, val_iou = validate_coco_segmentation(
                model=model,
                dataset=val_dataset,
                save_dir="val_outputs_cityscapes_pretrained_encoder/",
                batch_size=4,
                device='cuda'
            )
               

        # Save checkpoint
        torch.save(model.state_dict(), output_dir / f"model_epoch_best_full_transformer_last.pth")
        if val_iou > best_val_acc:
            best_val_acc = val_iou
            torch.save(model.state_dict(), output_dir / f"model_epoch_best_full_transformer_last_best.pth")
       # torch.save(refine_module.state_dict(), output_dir / f"refine_module_epoch_{epoch:02d}.pth")


if __name__ == "__main__":
    #args = parse_args()
    main()