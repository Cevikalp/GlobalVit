import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath("/media/mlcv/SSD/GlobalVit"))
from myutils.git_models import Dinov2SegmentationModelLast
from myutils.data_augmentation import ADE20KSegmentation, get_mmseg_transforms, CityscapesSegmentation, get_mmseg_transforms_new
from myutils.losses import SegLossIgnore, CELovaszLossWeighted
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- Configuration ---
IMG_SIZE = 770
PATCH_SIZE = 8
EMBED_DIM = 256
DEPTH = 6
NUM_HEADS = 8
BATCH_SIZE = 24
EPOCHS = 150
LR = 0.5*1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 19
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

save_dir = "val_outputs_cityscapes/"
output_dir = Path(save_dir)
output_dir.mkdir(exist_ok=True)

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


def save_predictions(image, mask, pred, save_dir, i):
    """
    image: torch tensor (3, H, W)
    gt_mask: torch tensor (H, W)
    pred_mask: torch tensor (H, W)
    """
     # Input image
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    # Masks
    gt_mask = mask.cpu().numpy().astype(np.int32)
    pred_mask = pred.cpu().numpy().astype(np.int32)

    color_gt = colorize_mask(gt_mask, NUM_CLASSES, COCO_COLORS)
    color_pred = colorize_mask(pred_mask, NUM_CLASSES, COCO_COLORS)

    img_pil.save(os.path.join(save_dir, f"val_{i}_input.png"))
    Image.fromarray(color_gt).save(
        os.path.join(save_dir, f"val_{i}_gt.png")
    )
    Image.fromarray(color_pred).save(
        os.path.join(save_dir, f"val_{i}_pred.png")
    )


train_transform = get_mmseg_transforms_new(img_size=IMG_SIZE, split="train")
val_transform = get_mmseg_transforms_new(img_size=IMG_SIZE, split="val")

# Paths
train_dataset = CityscapesSegmentation(
    root="/media/mlcv/Data/SegmentationProject/data/Cityscapes",
    split="train",
    transform=train_transform,
    image_size=(512, 512)
)

val_dataset = CityscapesSegmentation(
    root="/media/mlcv/Data/SegmentationProject/data/Cityscapes",
    split="val",
    transform=val_transform,
    image_size=(512, 512)
)



# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def evaluate_cityscapes(model, dataloader, device, num_classes=19):
    """
    Evaluates a semantic segmentation model on the Cityscapes validation set.

    Args:
        model: PyTorch model (nn.Module)
        dataloader: Validation DataLoader
        device: 'cuda' or 'cpu'
        num_classes: ADE20K standard is 150 (excluding background)
    """
    model.eval()
    # Initialize confusion matrix to store TP, FP, FN
    # size: (num_classes, num_classes)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64).to(device)

    i=0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device) # Shape: [B, H, W]

           
            # Forward pass
            outputs = model(images)
            
            # Handle model outputs (e.g., if model returns a dict like torchvision models)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            # Get prediction mask
            preds = torch.argmax(outputs, dim=1) # Shape: [B, H, W]
            save_predictions(images[0], targets[0], preds[0], save_dir, i)
            i += 1

            # Flatten and update confusion matrix
            # ADE20K targets are often 0-indexed (0 to 149). 
            # Ensure background/ignored pixels (often 255) are masked out.
            mask = (targets >= 0) & (targets < num_classes)
            
            # Efficiently calculate indices for the confusion matrix
            # indices = num_classes * ground_truth + predictions
            indices = num_classes * targets[mask].to(torch.int64) + preds[mask].to(torch.int64)
            
            # Count occurrences and add to matrix
            confusion_matrix += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)

    # --- Metric Calculations ---
    # Intersection = diagonal of confusion matrix
    intersection = torch.diag(confusion_matrix)
    
    # Ground Truth Sum = Union with FP (Rows)
    ground_truth_set = confusion_matrix.sum(dim=1)
    
    # Prediction Sum = Union with FN (Columns)
    predicted_set = confusion_matrix.sum(dim=0)
    
    # Union = GT + Pred - Intersection
    union = ground_truth_set + predicted_set - intersection
    
    # IoU = Intersection / Union
    # Use eps to avoid division by zero
    iou = intersection.float() / (union.float() + 1e-10)
    miou = torch.mean(iou).item()
    
    # Pixel Accuracy = Total Correct / Total Pixels
    pixel_acc = torch.diag(confusion_matrix).sum().float() / confusion_matrix.sum().float()

    print(f"\nEvaluation Results:")
    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Accuracy: {pixel_acc.item():.4f}")

    return {"mIoU": miou, "pixel_acc": pixel_acc.item(), "iou_per_class": iou.cpu().numpy()}

'''
# --- Model, Loss, Optimizer ---
model_2 = ViTFrontalizationEncoderDecoder6Layer(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    num_classes =NUM_CLASSES
).to(DEVICE)


model_3 = ViTFrontalizationEncoderDecoder(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    num_classes =NUM_CLASSES
).to(DEVICE)

# --- Model, Loss, Optimizer ---
model_1 = ViTFrontalizationEncoder(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    num_classes =NUM_CLASSES
).to(DEVICE)

model_4 = ViTSegmentationModel(
    img_size=IMG_SIZE,
    num_classes =NUM_CLASSES
).to(DEVICE)'''

model = Dinov2SegmentationModelLast(
      num_classes =NUM_CLASSES,
      img_size=IMG_SIZE,
).to(DEVICE)


LOAD_MODEL = True
#loading trained models
if LOAD_MODEL:
    model.load_state_dict(torch.load("/media/mlcv/SSD/Globalvit/segmentation_models/model_cityscapes.pth"))

#LOAD_MODEL = True
#loading trained models
#if LOAD_MODEL:
 #   model.load_state_dict(torch.load("/media/mlcv/Data/SegmentationProject/outputs_model/model_epoch_150.pth"))

def main():
# --- Training Loop ---
    
    results = evaluate_cityscapes(model, val_loader, device=DEVICE)
    print(results)

if __name__ == "__main__":
    #args = parse_args()
    main()

