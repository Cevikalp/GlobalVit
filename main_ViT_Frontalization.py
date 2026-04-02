import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from myutils.Frontalized_Dataset_KT_single_new import get_dataloader
import torch.nn.functional as F
from pathlib import Path
from myutils.losses import VGGLoss
from myutils.git_models import ViTFrontalizationEncoderDecoderLast, Discriminator # Assume you save the model in this file
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from backbones.iresnet_torch import iresnet100

# --- Configuration ---
IMG_SIZE = 256
PATCH_SIZE = 8
EMBED_DIM = 768
DEPTH = 6
NUM_HEADS = 8
BATCH_SIZE = 12
EPOCHS = 40
LR = 0.5*1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 550
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/train_images/"
VAL_MASK_DIR = "data/train_masks/"

arcface_model = iresnet100(pretrained=True)
arcface_model.to(DEVICE)
for param in arcface_model.parameters():
    param.requires_grad = False
arcface_model.eval()



def identity_loss(fake_imgs, real_imgs, arcface, device):
    arcface.eval()
    for param in arcface.parameters():
        param.requires_grad = False
    with torch.no_grad():
        real_emb = arcface(real_imgs.to(device))[0]   # [B, 512]
    fake_emb = arcface(fake_imgs.to(device))[0]       # [B, 512]
    
    # Cosine embedding loss
    id_loss = 1 - F.cosine_similarity(fake_emb, real_emb).mean()
    return id_loss


# --- Model, Loss, Optimizer ---
model = ViTFrontalizationEncoderDecoderLast(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS
).to(DEVICE)

#refine_module = DeepRefinementDecoderNew(in_channels=3,base_channels=64).to(DEVICE)
Disc = Discriminator(in_channels=3).to(DEVICE)

LOAD_MODEL = False
#loading trained models
if LOAD_MODEL:
    model.load_state_dict(torch.load("outputs13_id_loss/model_epoch_19.pth"))
    Disc.load_state_dict(torch.load("outputs13_id_loss/disc_model_epoch_19.pth"))
   # refine_module.load_state_dict(torch.load("outputs/refine_module_epoch_04.pth"))


optimizer_Disc = optim.Adam(Disc.parameters(), lr=LR, betas=(0.5, 0.999),)

#criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR)



perceptual_loss = VGGLoss(DEVICE)
bce = nn.BCEWithLogitsLoss()
# Add a scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scheduler_disc = CosineAnnealingLR(optimizer_Disc, T_max=EPOCHS, eta_min=1e-6)

# --- Training Function ---
def train_one_epoch(epoch, train_loader):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets, label_inter, label_intra) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
      #  targets = F.interpolate(targets, size=(2*IMAGE_HEIGHT, 2*IMAGE_WIDTH), mode='bilinear', align_corners=False)
     
        optimizer.zero_grad()
        outputs = model(inputs)
       # refined_outputs = refine_module(outputs)

        y_fake = outputs
        y_fake = y_fake.to(DEVICE)
        

        D_real = Disc(inputs, targets)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = Disc(inputs, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
       
        D_loss = (D_real_loss + D_fake_loss) / 2
        D_loss = D_loss.to(DEVICE)  

        # updating Disciminator
        optimizer_Disc.zero_grad()
        D_loss.backward()
        optimizer_Disc.step()

        D_fake = Disc(inputs, y_fake)
        G_loss = bce(D_fake, torch.ones_like(D_fake))

        arcface_outputs = F.interpolate(outputs, size=(112,112), mode='bilinear')
        arcface_outputs = (arcface_outputs - 0.5) / 0.5
        arcface_targets = F.interpolate(targets, size=(112,112), mode='bilinear')
        arcface_targets = (arcface_targets - 0.5) / 0.5
        arcface_inputs = F.interpolate(inputs, size=(112,112), mode='bilinear')
        arcface_inputs = (arcface_inputs - 0.5) / 0.5
        identity_loss_value = identity_loss(arcface_outputs, arcface_targets, arcface_model, DEVICE) #+ 0.1*identity_loss(arcface_outputs, arcface_inputs, arcface_model, DEVICE)


        resconstruction_loss = criterion(outputs, targets)
        perceptual_loss_value = perceptual_loss(outputs, targets)
        loss = 80*resconstruction_loss + 5*perceptual_loss_value + 0.01*G_loss + 0.75*identity_loss_value
        #loss = 10*criterion(refined_outputs, targets) + perceptual_loss(refined_outputs, targets) 
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, D_loss:{D_loss.item():.4f}, G_loss: {G_loss.item():.4f}, Identity Loss: {identity_loss_value.item():.4f}") 
    
    scheduler.step()
    scheduler_disc.step()
    return running_loss / len(train_loader)


# --- Validation ---
def validate(val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        count = 0
        for inputs, targets,  label_inter, label_intra in val_loader:
            count += 1
            if count > 5:
                break  # Limit to 5 batches for validation
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
           # targets = F.interpolate(targets, size=(2*IMAGE_HEIGHT, 2*IMAGE_WIDTH), mode='bilinear', align_corners=False)
     
            outputs = model(inputs)
           # refined_outputs = refine_module(outputs)
            loss = criterion( outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main(args):
# --- Training Loop ---
    output_dir = Path("outputs13_id_loss") 
    output_dir.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloader(args)
    #mask_feats, all_mask_label_inter, all_mask_label_intra = load_VGGmaskfeats(args,device)

   

    for epoch in range(1, EPOCHS + 1):
        model.eval()
        with torch.no_grad():
            for batch_idx, (sample_input, sample_targets, label_inter, label_intra) in enumerate(val_loader):
                if batch_idx >= 5:
                    break  # Only save one sample per epoch
                sample_input = sample_input.to(DEVICE)
                sample_output = model(sample_input)
                #sample_output = refine_module(sample_output)
                save_image(sample_input[0], output_dir / f"epoch_{epoch:02d}_input_{batch_idx:02d}.png")
                save_image(sample_output[0], output_dir / f"epoch_{epoch:02d}_output_{batch_idx:02d}.png")
        
        val_loss = validate(val_loader)
        train_loss = train_one_epoch(epoch, train_loader)
       
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

               

        # Save checkpoint
        torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch:02d}.pth")
       # torch.save(refine_module.state_dict(), output_dir / f"refine_module_epoch_{epoch:02d}.pth")
        torch.save(Disc.state_dict(), output_dir / f"disc_model_epoch_{epoch:02d}.pth")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training")
   
    parser.add_argument(
        "--folder-path",
        default="/media/mlcv/Data/FaceDatasets",
        help="additional note to output folder",
    )

    parser.add_argument(
        "--image-height",
        default=IMAGE_HEIGHT,
        help="additional note to output folder",
    )

    parser.add_argument(
        "--image-width",
        default=IMAGE_WIDTH,
        help="additional note to output folder",
    )

    parser.add_argument(
        "--train-meta-name",
        default="all_face_sets_meta_2026.npy",
        help="additional note to output folder",
    )

    parser.add_argument(
        "--test-meta-name",
        default="honda_subsets_meta_test.npy",
        help="additional note to output folder",
    )

    
    parser.add_argument(
        "--val-meta-name",
        default="honda_subsets_meta_test.npy",
        help="additional note to output folder",
    )

    parser.add_argument(
        "--mask-file",
        default="all_face_sets_mask_2026.txt",
        help="additional note to output folder",
    )

    parser.add_argument(
        "--savedimage-folder",
        default="/media/mlcv/DATA_SSD/FaceFrontalization/SavedImages",
        help="additional note to output folder",
    )
      
    parser.add_argument(
        "--distributed",
        default=False,
        help="additional note to output folder",
    )

    parser.add_argument(
        "--batch-size",
        default=BATCH_SIZE,
        help="additional note to output folder",
    )

   
    parser.add_argument(
        '--Seed', 
        default=0, type=int, metavar='N',
        help='Seed'
                        )

    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument('--device', default='cuda', help='device')
    
# distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)