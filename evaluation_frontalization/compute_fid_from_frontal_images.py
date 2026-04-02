import os
import scipy.misc
import numpy as np
import torch
#import tensorflow as tf

import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from PIL import Image
import torch
from torchvision.utils import save_image
#import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance

import os
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

def compute_fid_from_folders(generated_dir, mask_dir, device="cuda", batch_size=32):
    # 1. Initialize FID metric
    # feature=2048 is the standard dimension for Inception-v3 pool3 layer
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    # 2. Define Image Transformation
    # FID expects uint8 [0, 255], but torchmetrics handles the scaling 
    # if we provide the right input. We will load as tensors.
    transform = transforms.Compose([
        transforms.Resize((299, 299)), # Inception-v3 native resolution
        transforms.ToTensor(),
    ])

    # 3. Get and match filenames
    gen_files = [f for f in os.listdir(generated_dir) if f.startswith("output_")]
    
    # Processing in batches
    for i in tqdm(range(0, len(gen_files), batch_size), desc="Calculating FID"):
        batch_files = gen_files[i : i + batch_size]
        
        gen_batch = []
        mask_batch = []
        
        for f in batch_files:
            # Extract x_y suffix (e.g., output_1_5.png -> 1_5.png)
            suffix = f.replace("output_", "")
            mask_name = f"mask_{suffix}"
            
            gen_path = os.path.join(generated_dir, f)
            mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                # Load and transform
                gen_img = transform(Image.open(gen_path).convert("RGB"))
                mask_img = transform(Image.open(mask_path).convert("RGB"))
                
                gen_batch.append(gen_img)
                mask_batch.append(mask_img)
        
        if not gen_batch:
            continue

        # Convert list to 4D Tensor (N, 3, 299, 299)
        # We multiply by 255 and convert to uint8 as required by torchmetrics FID
        gen_tensor = (torch.stack(gen_batch) * 255).to(torch.uint8).to(device)
        mask_tensor = (torch.stack(mask_batch) * 255).to(torch.uint8).to(device)

        # Update statistics
        fid.update(mask_tensor, real=True)
        fid.update(gen_tensor, real=False)

    # 4. Final Computationp
    fid_score = fid.compute()
    return fid_score.item()
def main():
    generated_dir = "predictions_faces" # Replace with your generated images directory
    mask_dir = "predictions_masks" # Replace with your ground truth masks directory
    device = "cuda" if torch.cuda.is_available() else "cpu"

    score = compute_fid_from_folders(generated_dir, mask_dir, device=device)
    print(f"Final FID Score: {score}")

if __name__ == "__main__":
    main()
# Example Usage:
# score = compute_fid_from_folders("./predictions", "./masks")
# print(f"Final FID Score: {score}")