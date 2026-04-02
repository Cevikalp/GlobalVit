import os
import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
import pandas as pd
from PIL import Image
import cv2
import os
import sys
sys.path.append(os.path.abspath("/media/mlcv/SSD/GlobalVit"))
import insightface
from insightface.utils import face_align
from myutils.git_models import ViTFrontalizationEncoderDecoderLast, ViTFrontalizationEncoderDecoderDETR
from PIL import Image
import torch
from torchvision.utils import save_image
from pillow_heif import register_heif_opener


register_heif_opener()

def resize_down_6x(image):
    """
    Resizes the image by 6x while keeping the aspect ratio.
    """
    original_width, original_height = image.size
    
    # 6 kat küçültülmüş yeni boyutları hesapla
    new_width = original_width // 6
    new_height = original_height // 6
    
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

# --- Model, Loss, Optimizer ---
IMG_SIZE = 256
PATCH_SIZE = 8
EMBED_DIM = 768
DEPTH = 6
NUM_HEADS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model, Loss, Optimizer ---
model = ViTFrontalizationEncoderDecoderDETR(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS
).to(DEVICE)


REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041]    # right mouth
], dtype=np.float32)


# face detection algorithm
detector = insightface.app.FaceAnalysis()
detector.prepare(ctx_id=0)  # Use GPU if available, otherwise CPU

def detect_faces(img, detector):
 
    # Convert RGB → BGR (because OpenCV / RetinaFace expects BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    #cv2.imwrite("integer_im.jpg", img_bgr)
    faces = detector.get(img_bgr)

    for face in faces:
        print("BBox:", face.bbox)          # bounding box [x1, y1, x2, y2]
        print("Landmarks:", face.kps)      # 5 noktalı landmark
        print("Confidence:", face.det_score)

    return faces, img_bgr

def align_face_custom(img_bgr, landmarks, output_size=(112,112)):
    # Reference landmarks scaled to desired output size
    ref = REFERENCE_5PTS.copy()
    if output_size != (112,112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        ref[:, 0] *= scale_x
        ref[:, 1] *= scale_y

    # Compute similarity transform
    M, _ = cv2.estimateAffinePartial2D(landmarks, ref, method=cv2.LMEDS)

    # Warp image
    aligned = cv2.warpAffine(img_bgr, M, output_size, borderValue=0.0)
    return aligned


# Paths

img1_path = "/media/mlcv/SSD/GlobalVit/evaluation_frontalization/imnew.jpg" # Replace with your image path
img1 = Image.open(img1_path).convert("RGB")
#img1_small = resize_down_6x(img1)
img = np.array(img1)
faces1, img1 = detect_faces(img, detector)
landmarks1 = faces1[0].kps
aligned_face1 = align_face_custom(img1, landmarks1, output_size=(256,256))
cv2.imwrite("/media/mlcv/SSD/GlobalVit/evaluation_frontalization/im_aligned.png", aligned_face1) # Replace with your desired output path


img1_path = "/media/mlcv/SSD/GlobalVit/evaluation_frontalization/im_aligned.png" # Replace with your image path
aligned_face1 = Image.open(img1_path).convert("RGB")


#loading trained models
checkpoint = torch.load("/media/mlcv/SSD//GlobalVit/frontalization_models/model_detr.pth", map_location="cuda")
model.load_state_dict(checkpoint)

model.eval()

aligned_face1 = np.array(aligned_face1.resize((256,256))).astype(np.float32)
aligned_face1 = torch.tensor(aligned_face1, dtype=torch.float32)
aligned_face1 = aligned_face1.permute(2,0,1).unsqueeze(0)  # Add batch dimension


img = torch.tensor(aligned_face1, dtype=torch.float).to(DEVICE)
        #img_set = img_set.view(16,3,120,96).to(DEVICE)
        #img_set = torch.cat([img_set,img_set],dim=0)

img = img/255     
with torch.no_grad():
    pred = model(img)


save_image(pred, "/media/mlcv/SSD/GlobalVit/evaluation_frontalization/frontalized_im.png") # Replace with your desired output path
#save_image(img, "input.png")
