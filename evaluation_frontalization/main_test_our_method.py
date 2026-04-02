import os
import scipy.misc
import numpy as np
import torch
#import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath("/media/mlcv/SSD/GlobalVit"))
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from PIL import Image
from myutils.git_models import ViTFrontalizationEncoderDecoderLast
import torch
from torchvision.utils import save_image

dataset_path = "FaceDatasets"
test_path = "FaceDatasets/honda_subsets_meta_test.npy"
save_folder_pred = "./predictions_ours"
os.makedirs(save_folder_pred, exist_ok=True)
save_folder_mask = "./masks"
os.makedirs(save_folder_mask, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model, Loss, Optimizer ---
IMG_SIZE = 256
PATCH_SIZE = 8
EMBED_DIM = 768
DEPTH = 6
NUM_HEADS = 8

# --- Model, Loss, Optimizer ---
model = ViTFrontalizationEncoderDecoderLast(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS
).to(DEVICE)



LOAD_MODEL = True
#loading trained models
checkpoint = torch.load("frontalization_models/model_globalvit.pth", map_location="cuda")
#checkpoint = torch.load("/media/mlcv/Data/TransformerFrontalization/outputs8/model_epoch_48.pth", map_location="cuda:0")
model.load_state_dict(checkpoint)

model.eval()

def main():
   
    # Read image
    data_test = np.load(test_path)
    data_num_each_person = 16

    iter_num=0
    mse_avg = 0
    ssim_avg = 0
    for i in range(0,data_test.shape[0],data_num_each_person):
        
        pred_name = data_test["path"][i].split('/')[-2]
        img_mask_path = data_test["mask"][i] 
        img_mask = Image.open(os.path.join(dataset_path,img_mask_path)).convert("RGB")
        img_mask_resize = img_mask.resize((256,256))
        img_mask_resize.save(os.path.join(save_folder_mask, pred_name + '.png'))

        img_set = torch.zeros(16,3,256,256)
        img_set = torch.tensor(img_set, dtype=torch.float)
        for j in range(data_num_each_person):
            img_path = data_test["path"][i + j] 
            img = Image.open(os.path.join(dataset_path,img_path)).convert("RGB")
            img_resize = np.array(img.resize((256,256))).astype(np.float32)
            img_resize = torch.tensor(img_resize, dtype=torch.float)
            img_resize = np.transpose(img_resize, (2,0,1))
            img_set[j,:,:,:] = img_resize    
        

       
        img_set = img_set/255
        img_set = torch.tensor(img_set, dtype=torch.float).to(DEVICE)
        #img_set = img_set.view(16,3,120,96).to(DEVICE)
        #img_set = torch.cat([img_set,img_set],dim=0)
        
        with torch.no_grad():
            preds = model(img_set)

        for k in range(preds.shape[0]):
            img_tensor = preds[k,:,:,:]
            input_tensor = img_set[k,:,:,:]
            save_image(img_tensor, "{}/output_{}_{}.png".format(save_folder_pred, i+1, k))
           # save_image(input_tensor, "{}/output_{}_{}.png".format(save_folder_input, i+1, k))

        pred_np = preds.detach().cpu().numpy().squeeze().transpose(0,2,3,1)
        mask_np = np.array(img_mask_resize)/255        

        mse_metric_sum, ssim_metric_sum = 0, 0
        for k in range(pred_np.shape[0]):
            pred_np_k = pred_np[k,:,:,:]
            mse_metric = mean_squared_error(mask_np, pred_np_k)
            ssim_metric = ssim(mask_np,  pred_np_k, win_size=3, data_range= pred_np_k.max() -  pred_np_k.min()) 
            print("Pred {0:2d}-{1:2d}: mse_metric: {2:5.3f} ssim_metric: {3:5.8f} ".format(iter_num,k, mse_metric, ssim_metric))
            mse_metric_sum += mse_metric
            ssim_metric_sum += ssim_metric
        mse_avg+=(mse_metric_sum / data_num_each_person)
        ssim_avg+=(ssim_metric_sum / data_num_each_person)

        # Save output
        #img_output = Image.fromarray((rotated_img * 255).astype(np.uint8))
        
        #img_output.save(os.path.join(save_folder_pred, FLAGS.output_prefix + pred_name + '_' + str(iter_num%2) + '.png'))
        iter_num += 1
    
    print("Avg MSE: {}".format(mse_avg/(iter_num)))
    print("Avg SSIM: {}".format(ssim_avg/(iter_num)))
        


if __name__ == '__main__':
   main()
