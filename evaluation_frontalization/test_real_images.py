import os
from PIL import Image
import cv2
import torch
import re
import numpy as np
from torchvision.utils import save_image
import os
import sys
sys.path.append(os.path.abspath("/media/mlcv/SSD/GlobalVit"))
from myutils.git_models import ViTFrontalizationEncoderDecoderLast, ViTFrontalizationEncoderDecoderDETR, ViTFrontalizationEncoderDecoderPretrained, FaRLFrontalizationEncoderDecoder

folder_path = "/media/mlcv/SSD/GlobalVit/evaluation_frontalization/real_images"
save_folder = "/media/mlcv/SSD/GlobalVit/evaluation_frontalization/predicted_real_pretrained"
os.makedirs(save_folder, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# --- Model ---
model_type = "globalvit"  # "transface_encoder", "FaRL_encoder", "globalvit", "detr"

IMG_SIZE = 256
DEPTH = 6
NUM_HEADS = 8
BATCH_SIZE = 16
NUM_HEADS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == "transface_encoder":
    EMBED_DIM = 512
    model = ViTFrontalizationEncoderDecoderPretrained(
        img_size=IMG_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        decoder_depth=DEPTH,
    ).to(DEVICE)
    checkpoint = torch.load("/media/mlcv/SSD//GlobalVit/frontalization_models/model_pretrained_transface.pth", map_location="cuda")
    model.load_state_dict(checkpoint)

elif model_type == "FaRL_encoder":
    model = FaRLFrontalizationEncoderDecoder(
        img_size=IMG_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        decoder_depth=DEPTH,
    ).to(DEVICE)
elif model_type == "globalvit":
    EMBED_DIM = 768
    model = ViTFrontalizationEncoderDecoderLast(
        img_size=IMG_SIZE,
        patch_size=8,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    ).to(DEVICE)
    checkpoint = torch.load("/media/mlcv/SSD//GlobalVit/frontalization_models/model_epoch_last.pth", map_location="cuda")
    model.load_state_dict(checkpoint)
elif model_type == "detr":
    EMBED_DIM = 768
    model = ViTFrontalizationEncoderDecoderDETR(
        img_size=IMG_SIZE,
        patch_size=8,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    ).to(DEVICE)
    checkpoint = torch.load("/media/mlcv/SSD//GlobalVit/frontalization_models/model_detr.pth", map_location="cuda")
    model.load_state_dict(checkpoint)


#refine_module = DeepRefinementDecoderNew(in_channels=3,base_channels=64).to(DEVICE)



#loading trained models



model.eval()


def extract_number(filename):
    # im23.jpg → 23
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

image_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith(valid_ext)
]

# Integer'a göre sırala
image_files = sorted(image_files, key=extract_number)


for i, filename in enumerate(image_files):
    img_path = os.path.join(folder_path, filename)
    
    aligned_face1 = Image.open(img_path).convert("RGB")


    
    aligned_face1 = np.array(aligned_face1.resize((256,256))).astype(np.float32)
    aligned_face1 = torch.tensor(aligned_face1, dtype=torch.float32)
    aligned_face1 = aligned_face1.permute(2,0,1).unsqueeze(0)  # Add batch dimension


    img = torch.tensor(aligned_face1, dtype=torch.float).to(DEVICE)
            #img_set = img_set.view(16,3,120,96).to(DEVICE)
            #img_set = torch.cat([img_set,img_set],dim=0)

    img = img/255     
    with torch.no_grad():
        pred = model(img)


    save_image(pred, f"{save_folder}/frontalized_{i+1}.png")


print("Frontalization completed for all images in the folder.")