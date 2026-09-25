import cv2
from glob import glob
import os
import glob
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from .skin.dataset_isic import ISICDatasetFast


class PrepareBUSI:
    def __init__(self, data_dir, image_size, logger=None):
        self.print = logger.info if logger else print

        self.data_dir = data_dir
        self.image_size = image_size
        self.npy_dir = os.path.join(self.data_dir, "np")

    def __get_data_path(self):
        x_path = f"{self.npy_dir}/X_busi_{self.image_size}x{self.image_size}.npy"
        y_path = f"{self.npy_dir}/Y_busi_{self.image_size}x{self.image_size}.npy"
        return {"x": x_path, "y": y_path}

    def __get_transforms(self):
        img_transform = T.Resize(
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.BILINEAR
        )

        msk_transform = T.Resize(
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.BILINEAR
        )

        return {"img": img_transform, "msk": msk_transform}

    def is_data_existed(self):
        data_path = self.__get_data_path()
        return os.path.isfile(data_path["x"]) and os.path.isfile(data_path["y"])

    def prepare_data(self):
        data_path = self.__get_data_path()
        transforms = self.__get_transforms()

        imgs_dir = os.path.join(self.data_dir, "images")
        msks_dir = os.path.join(self.data_dir, "masks")

        img_paths = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))

        imgs = []
        msks = []

        for img_path in tqdm(img_paths):
            filename = os.path.basename(img_path)
            msk_path = os.path.join(msks_dir, filename)

            if not os.path.exists(msk_path):
                continue

            img = read_image(img_path, ImageReadMode.GRAY)
            msk = read_image(msk_path, ImageReadMode.GRAY)

            img = transforms["img"](img)
            msk = transforms["msk"](msk)
            msk = (msk - msk.min()) / (msk.max() - msk.min() + 1e-8)  # Normalize to [0, 1]
            msk = np.where(msk > 0.5, 255, 0).astype(np.uint8)  # Ensure binary mask is uint8

            imgs.append(img.numpy())
            msks.append(msk)

        X = np.array(imgs)
        Y = np.array(msks)

        Path(self.npy_dir).mkdir(exist_ok=True)

        self.print("Saving BUSI np arrays...")
        np.save(data_path["x"].split(".npy")[0], X)
        np.save(data_path["y"].split(".npy")[0], Y)
        self.print(f"Saved at:\n  X: {data_path['x']}\n  Y: {data_path['y']}")

    def get_data(self):
        data_path = self.__get_data_path()

        if not self.is_data_existed():
            self.print("No pre-saved BUSI files found. Preparing data...")
            self.prepare_data()

        X = np.load(data_path["x"])
        Y = np.load(data_path["y"])

        return {"x": X, "y": Y}




class FastBUSIDataset(ISICDatasetFast):
    def __init__(
        self,
        mode,                # "tr" or "te"
        data_dir,
        one_hot=False,
        image_size=224,
        transform=None,
        aug_transform=None,
        img_transform=None,
        msk_transform=None,
        logger=None,
        seed=42,
    ):
        self.print = logger.info if logger else print

        self.data_dir = data_dir
        self.one_hot = one_hot
        self.image_size = image_size
        self.transform = transform
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode

        # -------- Load full dataset first --------
        data_preparer = PrepareBUSI(
            data_dir=self.data_dir,
            image_size=self.image_size,
            logger=logger,
        )

        data = data_preparer.get_data()
        X, Y = data["x"], data["y"]  # shape: (N, 1, H, W)

        # Move channel to last dim (for albumentations compatibility)
        X = np.moveaxis(np.uint8(X), 1, -1)
        Y = np.moveaxis(np.uint8(Y), 1, -1)

        # Fixed-seed 70/10/20 split. After the shuffle, the last 20% is test
        # and the next 10% is validation. The rest is train.
        N = len(X)
        indices = list(range(N))

        random.seed(seed)
        random.shuffle(indices)

        n_test = int(0.2 * N)
        n_val = int(0.1 * N)
        test_idx = indices[-n_test:]
        rest = indices[:-n_test]
        val_idx = rest[-n_val:]
        train_idx = rest[:-n_val]

        if mode == "tr":
            selected_idx = train_idx
        elif mode == "vl":
            selected_idx = val_idx
        elif mode == "te":
            selected_idx = test_idx
        else:
            raise ValueError("mode must be 'tr', 'vl', or 'te'")

        self.imgs = X[selected_idx]
        self.msks = Y[selected_idx]



from .skin.dataset_isic import aug_transform, img_transform, msk_transform

def get_busi(args, logger=None, verbose=True):

    tr_dataset = FastBUSIDataset(
        mode="tr",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        aug_transform=aug_transform,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
        seed=args.seed,
    )

    vl_dataset = FastBUSIDataset(
        mode="vl",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
        seed=args.seed,
    )

    te_dataset = FastBUSIDataset(
        mode="te",
        data_dir=args.data_dir,
        one_hot=False,
        image_size=args.img_size,
        img_transform=img_transform,
        msk_transform=msk_transform,
        logger=logger,
        seed=args.seed,
    )

    if verbose:
        print("BUSI Dataset:")
        print(f"├──> Length of training_dataset:\t {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset:\t {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t {len(te_dataset)}")

    return {
        "tr_dataset": tr_dataset,
        "vl_dataset": vl_dataset,
        "te_dataset": te_dataset,
    }
