import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
import numpy as np
import os
from torchvision.transforms import v2
from torchvision import tv_tensors

class CityscapesSegmentation(Dataset):
    """
    Cityscapes semantic segmentation dataset
    Output:
        image: Tensor [3, H, W]
        target: LongTensor [H, W] with values in [0,18] or 255 (ignore)
    """

    def __init__(self, root, split="train", image_size=(512, 512), transform=None):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size

        self.image_dir = os.path.join(root, "leftImg8bit", split)
        self.label_dir = os.path.join(root, "gtFine", split)

        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []

        for city in sorted(os.listdir(self.image_dir)):
            img_city_dir = os.path.join(self.image_dir, city)
            lbl_city_dir = os.path.join(self.label_dir, city)

            for fname in os.listdir(img_city_dir):
                if not fname.endswith("_leftImg8bit.png"):
                    continue

                img_path = os.path.join(img_city_dir, fname)

                label_name = fname.replace(
                    "_leftImg8bit.png",
                    "_gtFine_labelTrainIds.png"
                )
                label_path = os.path.join(lbl_city_dir, label_name)

                if self.split != "test":
                    assert os.path.exists(label_path)

                samples.append((img_path, label_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.split != "test":
            target = Image.open(label_path)
        else:
            target = None



        image = tv_tensors.Image(image)
        target = tv_tensors.Mask(target)

        # 3. Apply transformations
        if self.transform is not None:
            # The v2 transform will now automatically skip ColorJitter/Normalize for 'target'
            image, target = self.transform(image, target)

        # 4. Convert to clean Tensors
        # We use np.array to strip all 'TVTensor' metadata and get raw class IDs
        target_np = np.array(target)
        target = torch.from_numpy(target_np).long()

        # If the transform returned [1, H, W], squeeze it to [H, W]
        if target.ndim == 3:
            target = target.squeeze(0)

        # Ensure image is a float tensor [3, H, W]
        if not isinstance(image, torch.Tensor):
            image = F.to_image(image)
            image = F.to_dtype(image, torch.float32, scale=True)

        
        return image, target


class ADE20KSegmentation(Dataset):
    def __init__(self, root, split="training", image_size=(512, 512), transform=None):
        assert split in ["training", "validation"]
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.image_dir = os.path.join(root, "images", split)
        self.label_dir = os.path.join(root, "annotations", split)
        self.samples = self._collect_samples()

    def _collect_samples(self):
        images = sorted(os.listdir(self.image_dir))
        return [(os.path.join(self.image_dir, f), 
                 os.path.join(self.label_dir, os.path.splitext(f)[0] + ".png")) 
                for f in images if f.lower().endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # 1. Load as PIL
        image = Image.open(img_path).convert("RGB")
        target = Image.open(label_path).convert("L")

        # 2. WRAP THEM (This is the critical step)
        # This tells the transform "this is an image" and "this is a mask"
        image = tv_tensors.Image(image)
        target = tv_tensors.Mask(target)

        # 3. Apply transformations
        if self.transform is not None:
            # The v2 transform will now automatically skip ColorJitter/Normalize for 'target'
            image, target = self.transform(image, target)

        # 4. Convert to clean Tensors
        # We use np.array to strip all 'TVTensor' metadata and get raw class IDs
        target_np = np.array(target)
        target = torch.from_numpy(target_np).long()

        # If the transform returned [1, H, W], squeeze it to [H, W]
        if target.ndim == 3:
            target = target.squeeze(0)

        # Ensure image is a float tensor [3, H, W]
        if not isinstance(image, torch.Tensor):
            image = F.to_image(image)
            image = F.to_dtype(image, torch.float32, scale=True)

        return image, target

   
def get_mmseg_transforms_new(img_size=770, split="train"):
    # DINOv2 mean/std (standard ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return v2.Compose([
            # 1. Resize short side to something larger than the crop, keeping aspect ratio
            # This ensures we don't have empty borders after the crop.
            v2.Resize(896, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            
            # 2. Random Crop to your target size
            # Since Resize kept the ratio, this crop preserves the natural geometry of objects
            v2.RandomCrop((img_size, img_size)),
            
            # 3. Horizontal Flip (50% probability)
            v2.RandomHorizontalFlip(p=0.5),
            
            # 4. PhotoMetric Distortion
            v2.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.01),
            
            # 5. Conversion and Normalization
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation Pipeline
        return v2.Compose([
            # Keep aspect ratio even in validation for better mIoU
            v2.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            v2.CenterCrop((img_size, img_size)), 
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

def get_mmseg_transforms(img_size=770, split="train"):
    """
    Standard MMSegmentation-style augmentation pipeline.
    """
    if split == "train":
        return v2.Compose([
            # 1. Scale and Crop (Randomly scales between 0.5x and 2.0x then crops)
            v2.RandomResizedCrop(
                size=(img_size, img_size), 
                scale=(0.5, 2.0), 
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            
            # 2. Horizontal Flip (50% probability)
            v2.RandomHorizontalFlip(p=0.5),
            
            # 3. PhotoMetric Distortion
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            
            # 4. Final conversion and Normalization
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), # Converts 0-255 to 0.0-1.0
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Validation Pipeline (Consistent and Deterministic)
        return v2.Compose([
            v2.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])