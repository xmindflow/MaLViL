import os
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import albumentations as A
import glob
from torchvision import transforms as T
from albumentations import *
from utils.utils_isic import histogram_equalization_rgb



class PrepareISIC:
    def __init__(self, dataset_name, data_dir, image_size, logger=None):
        self.print = logger.info if logger else print
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.dataset_name = dataset_name

        # preparing input info.
        self.data_prefix = "ISIC_"
        self.target_postfix = "_segmentation"
        self.target_fex = "png"
        self.input_fex = "jpg"
        self.data_dir = self.data_dir
        self.npy_dir = os.path.join(self.data_dir, "np")

    def __get_data_path(self):
        x_path = f"{self.npy_dir}/X_tr_{self.image_size}x{self.image_size}.npy"
        y_path = f"{self.npy_dir}/Y_tr_{self.image_size}x{self.image_size}.npy"
        return {"x": x_path, "y": y_path}

    def __get_img_by_id(self, id):
        img_dir = os.path.join(
            self.imgs_dir, f"{self.data_prefix}{id}.{self.input_fex}"
        )
        img = read_image(img_dir, ImageReadMode.RGB)
        return img

    def __get_msk_by_id(self, id):
        msk_dir = os.path.join(
            self.msks_dir,
            f"{self.data_prefix}{id}{self.target_postfix}.{self.target_fex}",
        )
        msk = read_image(msk_dir, ImageReadMode.GRAY)
        return msk

    def __get_transforms(self):
        # transform for image
        img_transform = T.Compose([T.Resize(
            size=[self.image_size, self.image_size],
            interpolation=T.functional.InterpolationMode.BILINEAR,
        )])
        # transform for mask
        msk_transform = T.Compose([T.Resize(
            size=[self.image_size, self.image_size],
            interpolation=T.functional.InterpolationMode.BILINEAR,
        )])
        return {"img": img_transform, "msk": msk_transform}

    def is_data_existed(self):
        data_path = self.__get_data_path()
        return os.path.exists(data_path["x"]) and os.path.exists(data_path["y"])

    def prepare_data(self):
        data_path = self.__get_data_path()

        # Parameters
        self.transforms = self.__get_transforms()

        if "2017" in self.dataset_name.lower():
            self.imgs_dir = os.path.join(self.data_dir, "ISIC-2017_Training_Data")
            self.msks_dir = os.path.join(
                self.data_dir, "ISIC-2017_Training_Part1_GroundTruth"
            )
        elif "2018" in self.dataset_name.lower():
            self.imgs_dir = os.path.join(self.data_dir, "ISIC2018_Task1-2_Training_Input")
            self.msks_dir = os.path.join(
                self.data_dir, "ISIC2018_Task1_Training_GroundTruth"
            )
        elif "ham" in self.dataset_name.lower():
            self.imgs_dir = os.path.join(self.data_dir, "images")
            self.msks_dir = os.path.join(self.data_dir, "masks")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported!")

        self.img_dirs = sorted(glob.glob(f"{self.imgs_dir}/*.{self.input_fex}"))
        self.data_ids = [
            os.path.basename(d).split(f".{self.input_fex}")[0].split(self.data_prefix)[1]
            for d in self.img_dirs
        ]

        # gathering images
        imgs = []
        msks = []
        for data_id in tqdm(self.data_ids):
            img = self.__get_img_by_id(data_id)
            msk = self.__get_msk_by_id(data_id)

            img = self.transforms["img"](img)
            msk = self.transforms["msk"](msk)
            msk = (msk - msk.min()) / (msk.max() - msk.min() + 1e-8)  # Normalize to [0, 1]
            msk = np.where(msk > 0.5, 255, 0).astype(np.uint8)  # Ensure binary mask is uint8

            imgs.append(img.numpy())
            msks.append(msk)

        X = np.array(imgs)
        Y = np.array(msks)

        Path(self.npy_dir).mkdir(exist_ok=True)

        self.print(f"Saving data {self.dataset_name}...")
        np.save(data_path["x"].split(".npy")[0], X)
        np.save(data_path["y"].split(".npy")[0], Y)
        self.print(f"  Saved at:\n  X: {data_path['x']}\n  Y: {data_path['y']}")
        return

    def get_data(self):
        data_path = self.__get_data_path()

        if not self.is_data_existed():
            self.print(f"There are no pre-saved files for dataset {self.dataset_name}")
            self.print("Preparing data...")
            self.prepare_data()
        else:
            self.print(f"Found pre-saved files at {self.npy_dir}")

        X = np.load(data_path["x"])
        Y = np.load(data_path["y"])

        return {"x": X, "y": Y}


class ISICDatasetFast(Dataset):
    def __init__(
        self,
        mode,
        dataset_name,
        data_dir=None,
        one_hot=True,
        image_size=224,
        transform=None,
        aug_transform=None,
        img_transform=None,
        msk_transform=None,
        logger=None,
        **kwargs,
    ):
        self.print = logger.info if logger else print

        # pre-set variables
        self.data_dir = data_dir if data_dir else "/path/to/datasets/ISIC2017-8"

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.transform = transform
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode

        if "isic2017" in dataset_name.lower():
            tr_length, vl_length = 1250, 150 # test = 600 | total = 2000
        elif "isic2018" in dataset_name.lower():
            tr_length, vl_length = 1815, 259 # test = 520 | total = 2594
        else:
            raise ValueError(f"Dataset {dataset_name} not supported!")

        data_preparer = PrepareISIC(dataset_name=dataset_name,
            data_dir=self.data_dir, image_size=self.image_size, logger=logger
        )
        data = data_preparer.get_data()
        X, Y = data["x"], data["y"]        
        
        if mode == "tr":
            self.imgs = X[:tr_length]
            self.msks = Y[:tr_length]
        elif mode == "vl":
            self.imgs = X[tr_length : tr_length + vl_length]
            self.msks = Y[tr_length : tr_length + vl_length]
        elif mode == "te":
            self.imgs = X[tr_length + vl_length :]
            self.msks = Y[tr_length + vl_length :]
        else:
            raise ValueError(f"Unknown mode {mode}. Expected one of ['tr','vl','te'].")
        
        self.imgs = np.moveaxis(np.uint8(self.imgs), 1, -1)
        self.msks = np.moveaxis(np.uint8(self.msks), 1, -1)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = idx
        img = self.imgs[idx]
        msk = self.msks[idx]
        # if self.mode != "tr":
        #     img = histogram_equalization_rgb(img)

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
    T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
])
img_transform = T.Compose([
    T.ToTensor(),      # [0,1] float32
    T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
])

def get_isic2018(args, logger=None, verbose=True):

    # ----------------- dataset --------------------
    # preparing training dataset
    tr_dataset = ISICDatasetFast(
        mode="tr",
        dataset_name="isic2018",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        aug_transform=aug_transform,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
        data_scale="full"
    )
    vl_dataset = ISICDatasetFast(
        mode="vl",
        dataset_name="isic2018",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
    )
    te_dataset = ISICDatasetFast(
        mode="te",
        dataset_name="isic2018",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
    )

    if verbose:
        print("ISIC 2018:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t   {len(te_dataset)}")

    return {
        "tr_dataset": tr_dataset,
        "vl_dataset": vl_dataset,
        "te_dataset": te_dataset,
    }


def get_isic2017(args, logger=None, verbose=True):
    """
    Build train / val / test datasets for ISIC 2017.

    Expected folder structure (under args.data_dir_2017 or args.data_dir_isic17 or args.data_dir):
        ISIC-2017_Training_Data/
        ISIC-2017_Training_Part1_GroundTruth/
    """
    
    # ----------------- dataset --------------------
    # preparing training dataset
    tr_dataset = ISICDatasetFast(
        mode="tr",
        dataset_name="isic2017",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        aug_transform=aug_transform,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
    )
    vl_dataset = ISICDatasetFast(
        mode="vl",
        dataset_name="isic2017",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
    )
    te_dataset = ISICDatasetFast(
        mode="te",
        dataset_name="isic2017",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
    )

    if verbose:
        print("ISIC 2017:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t   {len(te_dataset)}")

    return {
        "tr_dataset": tr_dataset,
        "vl_dataset": vl_dataset,
        "te_dataset": te_dataset,
    }
