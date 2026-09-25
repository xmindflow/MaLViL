import os
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F
from PIL import Image
from utils.utils_skin import histogram_equalization_rgb
from .dataset_isic import PrepareISIC


class HAM10000DatasetFast(Dataset):
    def __init__(self,
                 mode,
                 data_dir=None,
                 one_hot=True,
                 image_size=224,
                 transform=None,
                 aug_transform=None,
                 img_transform=None,
                 msk_transform=None,
                 logger=None,
                 **kwargs):
        self.print=logger.info if logger else print
        
        # pre-set variables
        self.data_dir = data_dir if data_dir else "/path/to/datasets/HAM10000"

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.transform = transform
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode

        data_preparer = PrepareISIC(dataset_name="ham10000",
            data_dir=self.data_dir, image_size=self.image_size, logger=logger
        )
        data = data_preparer.get_data()
        X, Y = data["x"], data["y"]

        tr_length, vl_length = 7200, 1800  # test = 1015, total = 10015
        
        if mode == "tr":
            self.imgs = X[:tr_length]
            self.msks = Y[:tr_length]
        elif mode == "vl":
            self.imgs = X[7200:7200 + vl_length]
            self.msks = Y[7200:7200 + vl_length]
        elif mode == "te":
            self.imgs = X[7200 + vl_length :]
            self.msks = Y[7200 + vl_length :]
        else:
            raise ValueError()  

        self.imgs = np.moveaxis(np.uint8(self.imgs), 1, -1)
        self.msks = np.moveaxis(np.uint8(self.msks), 1, -1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = idx
        img = self.imgs[idx]
        msk = self.msks[idx]
        # if self.mode != "tr":
            # img = histogram_equalization_rgb(img)

        if self.aug_transform:
            augmented = self.aug_transform(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']

            # img = histogram_equalization_rgb(img)
            img = np.nan_to_num(img, nan=0)
            msk = np.nan_to_num(msk, nan=0)

        if self.transform:
            img = self.transform(img)
            msk = self.transform(msk)
            
        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform:
            msk = self.msk_transform(msk)
        
        if self.one_hot:
            msk = (msk - msk.min()) / (msk.max() - msk.min())
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)
        else:
            msk = msk.squeeze()

        msk = (msk > 0).long()  # binarize

        sample = {"image": img, "label": msk, "id": data_id}
        return sample



import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
def get_ham10000(args, logger=None, verbose=True):

    # Define augmentations
    aug_transform = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        # A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        # ToTensorV2(),
    ])
    msk_transform = T.Compose([
        T.ToTensor(),
    ])
    img_transform = T.Compose([
        T.ToTensor(),      # [0,1] float32
        T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
    ])

    # ----------------- dataset --------------------
    # preparing training dataset
    tr_dataset = HAM10000DatasetFast(
        mode="tr",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        aug_transform=aug_transform,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
        data_scale="full"
    )
    vl_dataset = HAM10000DatasetFast(
        mode="vl",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
    )
    te_dataset = HAM10000DatasetFast(
        mode="te",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
    )

    if verbose:
        print("HAM10000:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t   {len(te_dataset)}")

    return {
        "tr_dataset": tr_dataset,
        "vl_dataset": vl_dataset,
        "te_dataset": te_dataset,
    }
