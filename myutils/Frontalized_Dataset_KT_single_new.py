import os
import time
import torch
import torch.utils.data as D
from torchvision import transforms as T
import numpy as np
from itertools import cycle
from math import floor, ceil
from PIL import Image
#from catalyst.data.sampler import BatchBalanceClassSampler, DistributedSamplerWrapper

from torch.utils.data import Sampler
import random
from collections import defaultdict


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes_per_batch):
        """
        Args:
            labels (List[int]): list of class labels (len = dataset size)
            batch_size (int): total number of samples in each batch
            num_classes_per_batch (int): how many different classes per batch
        """
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.class_to_indices = self._build_index()
        self.classes = list(self.class_to_indices.keys())
        self.samples_per_class = batch_size // num_classes_per_batch

    def _build_index(self):
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        return class_to_indices

    def __iter__(self):
        class_cursors = {cls: 0 for cls in self.classes}
        class_indices = {cls: random.sample(idxs, len(idxs)) for cls, idxs in self.class_to_indices.items()}

        # infinite loop, ends only when all indices are exhausted
        while True:
            selected_classes = random.sample(self.classes, self.num_classes_per_batch)
            batch = []

            for cls in selected_classes:
                cursor = class_cursors[cls]
                cls_indices = class_indices[cls]

                if cursor + self.samples_per_class > len(cls_indices):
                    cls_indices = random.sample(self.class_to_indices[cls], len(self.class_to_indices[cls]))
                    class_indices[cls] = cls_indices
                    cursor = 0

                batch.extend(cls_indices[cursor:cursor + self.samples_per_class])
                class_cursors[cls] = cursor + self.samples_per_class

            if len(batch) == self.batch_size:
                yield batch
            else:
                break

    def __len__(self):
        # conservative estimate: total number of samples divided by batch size
        return len(self.labels) // self.batch_size
    

class FrantalizedFace_Dataset(D.Dataset):
    def __init__(
        self,
        folder_path="/media/mlcv/DATA_SSD/FaceFrontalization/Datasets",
        meta_name="deneme.npy",
        transform=None,
        transform_mask=None,
        file_path=False,
        selected_class=None,
        transform_aux=None,
    ):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.data = np.load(os.path.join(folder_path, meta_name))
        self.num_classes = len(np.unique(self.data["class_inter"]))
       
        self.data_len = len(self.data)
        self.transform = transform
        self.transform_mask = transform_mask
        self.transform_aux = transform_aux
        self.folder_path = folder_path

        self.file_path = file_path

    def __getitem__(self, index):
        file_path, cls_idx_inter, cls_idx_intra, mask_path = self.data[index]
        img = Image.open(os.path.join(self.folder_path, file_path)).convert("RGB")
        label_inter = np.array(cls_idx_inter)
        label_intra = np.array(cls_idx_intra)
        mask = Image.open(os.path.join(self.folder_path, mask_path)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform_mask(mask)

        if self.transform_aux is not None:
            img = self.transform_aux(img)
            mask = self.transform_aux(mask)

        if self.file_path:
            return (img, mask, label_inter, label_intra, file_path)
        else:
            return (img, mask, label_inter, label_intra)

    def __len__(self):
        return self.data_len

   
def get_dataloader(args):
    # Data loading code
    print("Loading data")
    st = time.time()
    IMAGE_HEIGHT = args.image_height  # 1280 originally
    IMAGE_WIDTH = args.image_width  # 1918 originally
   
    transform = T.Compose(
    [
        T.Resize((IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)),
        T.RandomRotation(degrees=3),
        T.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),   # <-- random 256x256 crop
        T.ToTensor(),
    ]
)

    transform_mask = T.Compose(
        [
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
           # T.RandomRotation(degrees=10),  # no probability, always applies
           # T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
            #T.VerticalFlip(p=0.1),
            #T.Normalize(
            #   mean=[0.0, 0.0, 0.0],
            #   std=[1.0, 1.0, 1.0],
            #   max_pixel_value=255.0,
            #),           
            
        ]
    )
    
    dataset = FrantalizedFace_Dataset(
        folder_path=args.folder_path,
        meta_name=args.train_meta_name,
        transform=transform,
        transform_mask=transform_mask,
        file_path=False,
    )
    dataset_val = FrantalizedFace_Dataset(
        folder_path=args.folder_path,
        meta_name=args.val_meta_name,
        transform=transform,
        transform_mask=transform_mask,
        file_path=False,
    )
    dataset_test = FrantalizedFace_Dataset(
        folder_path=args.folder_path,
        meta_name=args.test_meta_name,
        transform=transform,
        transform_mask=transform_mask,
        file_path=False,
    )
    print("Took", time.time() - st)

    print("Creating data loaders")
    
     

    dataloader = D.DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )
    dataloader_val= D.DataLoader(
        dataset_val, shuffle=True, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )
    dataloader_test= D.DataLoader(
        dataset_test, shuffle=True, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )

    #dataloader_test = D.DataLoader(
     #   dataset_test,
      #  batch_size=args.batch_size,
      #  sampler=test_sampler,
      #  num_workers=args.workers,
      #  pin_memory=True,
      #  shuffle=False,
    #)

    return dataloader, dataloader_val, dataloader_test

def get_testdataloader(args):
    # Data loading code
    print("Loading data")
    st = time.time()
    IMAGE_HEIGHT = args.image_height  # 1280 originally
    IMAGE_WIDTH = args.image_width  # 1918 originally
   
    transform = T.Compose(
        [
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            T.ToTensor()
            #T.Rotate(limit=35, p=1.0),
            #T.HorizontalFlip(p=0.5),
            #T.VerticalFlip(p=0.1),
            #T.Normalize(
            #   mean=[0.0, 0.0, 0.0],
            #   std=[1.0, 1.0, 1.0],
            #   max_pixel_value=255.0,
            #),           
            
        ]
    )

    
    dataset_test = FrantalizedFace_Dataset(
        folder_path=args.folder_path,
        meta_name=args.test_meta_name,
        transform=transform,
        file_path=False,
    )
    print("Took", time.time() - st)

    print("Creating data loaders")
   

   
    dataloader_test= D.DataLoader(
        dataset_test, shuffle=True, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )

    #dataloader_test = D.DataLoader(
     #   dataset_test,
      #  batch_size=args.batch_size,
      #  sampler=test_sampler,
      #  num_workers=args.workers,
      #  pin_memory=True,
      #  shuffle=False,
    #)

    return dataloader_test