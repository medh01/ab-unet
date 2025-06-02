import os
import shutil
import random
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

################################################################################
# 2.1) SPLIT IMAGES INTO LABELED / UNLABELED / TEST FOLDERS
################################################################################
def labeled_unlabeled_test_split(
    base_dir: str,
    labeled_dir: str,
    unlabeled_dir: str,
    test_dir: str,
    label_split_ratio: float = 0.05,
    test_split_ratio: float = 0.3,
    shuffle: bool = True
):
    """
    Given base_dir (e.g. “ab-unet/data”), where:
        base_dir/
          Images/   ← your raw images (capital “I”)
          masks/    ← your raw masks
    Split into three subfolders:
        base_dir/Labeled_pool/{labeled_images/, labeled_masks/}
        base_dir/Unlabeled_pool/{unlabeled_images/, unlabeled_masks/}
        base_dir/Test/{test_images/, test_masks/}
    according to label_split_ratio and test_split_ratio.
    """

    # Originally we assumed “images”, but in your project the folder is “Images” with capital I:
    img_folder  = os.path.join(base_dir, "Images")   # <--- changed from "images"
    mask_folder = os.path.join(base_dir, "masks")    # your masks folder stays lowercase

    all_images = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    if shuffle:
        random.shuffle(all_images)

    N = len(all_images)
    n_test      = int(N * test_split_ratio)
    n_labeled   = int((N - n_test) * label_split_ratio)
    n_unlabeled = N - n_test - n_labeled

    # Helper to create subfolders if they don’t exist
    def make_subfolders(root, folder_names):
        for fn in folder_names:
            dir_path = os.path.join(root, fn)
            os.makedirs(dir_path, exist_ok=True)

    # Create structure:
    # base_dir/Labeled_pool/{labeled_images, labeled_masks}
    # base_dir/Unlabeled_pool/{unlabeled_images, unlabeled_masks}
    # base_dir/Test/{test_images, test_masks}
    make_subfolders(base_dir, [labeled_dir, unlabeled_dir, test_dir])
    make_subfolders(os.path.join(base_dir, labeled_dir),   ["labeled_images",   "labeled_masks"])
    make_subfolders(os.path.join(base_dir, unlabeled_dir), ["unlabeled_images", "unlabeled_masks"])
    make_subfolders(os.path.join(base_dir, test_dir),      ["test_images",      "test_masks"])

    # Distribute images and masks
    for i, im_name in enumerate(all_images):
        if i < n_labeled:
            split = labeled_dir
            sub_im  = "labeled_images"
            sub_msk = "labeled_masks"
        elif i < n_labeled + n_unlabeled:
            split = unlabeled_dir
            sub_im  = "unlabeled_images"
            sub_msk = "unlabeled_masks"
        else:
            split = test_dir
            sub_im  = "test_images"
            sub_msk = "test_masks"

        # Copy the image file
        src_im = os.path.join(img_folder, im_name)
        dst_im = os.path.join(base_dir, split, sub_im, im_name)
        shutil.copy(src_im, dst_im)

        # Copy the corresponding mask file
        # Here we assume masks are named by replacing "IMGNAME.ext" → "IMGNAME_mask.ext"
        # Adjust this logic if your naming is different.
        mask_name = im_name.replace(".jpg", "_mask.png")
        src_msk = os.path.join(mask_folder, mask_name)
        dst_msk = os.path.join(base_dir, split, sub_msk, mask_name)
        if os.path.exists(src_msk):
            shutil.copy(src_msk, dst_msk)
        else:
            raise FileNotFoundError(f"Mask {src_msk} not found for image {im_name}")


################################################################################
# 2.2) MOVE IMAGES & MASKS FROM UNLABELED → LABELED (USING RANDOM OR SCORE ORDER)
################################################################################
def move_images(
    base_dir: str,
    labeled_dir: str,
    unlabeled_dir: str,
    num_to_move: int = 10
):
    """
    Move `num_to_move` random images (and their matching masks) from
    base_dir/unlabeled_dir/ → base_dir/labeled_dir/
    """
    unlabeled_imgs = os.listdir(os.path.join(base_dir, unlabeled_dir, "unlabeled_images"))
    random.shuffle(unlabeled_imgs)
    to_move = unlabeled_imgs[:num_to_move]

    for im in to_move:
        # Move the image
        src_im = os.path.join(base_dir, unlabeled_dir, "unlabeled_images", im)
        dst_im = os.path.join(base_dir, labeled_dir,   "labeled_images",   im)
        shutil.copy(src_im, dst_im)
        os.remove(src_im)

        # Move the mask
        mask_name = im.replace(".jpg", "_mask.png")
        src_msk   = os.path.join(base_dir, unlabeled_dir, "unlabeled_masks", mask_name)
        dst_msk   = os.path.join(base_dir, labeled_dir,   "labeled_masks",   mask_name)
        if os.path.exists(src_msk):
            shutil.copy(src_msk, dst_msk)
            os.remove(src_msk)
        else:
            print(f"Warning: mask {src_msk} not found; skipping.")


def move_images_with_dict(
    base_dir: str,
    labeled_dir: str,
    unlabeled_dir: str,
    score_dict: dict,
    num_to_move: int = 10
):
    """
    Move the top‐`num_to_move` images (in descending score order) from unlabeled → labeled,
    based on the provided score_dict {filename:score}.
    """
    moved = 0
    for im, score in score_dict.items():
        if moved >= num_to_move:
            break
        src_im = os.path.join(base_dir, unlabeled_dir, "unlabeled_images", im)
        if not os.path.exists(src_im):
            continue  # maybe it was already moved
        dst_im = os.path.join(base_dir, labeled_dir,   "labeled_images",   im)
        shutil.copy(src_im, dst_im)
        os.remove(src_im)

        mask_name = im.replace(".jpg", "_mask.png")
        src_msk   = os.path.join(base_dir, unlabeled_dir, "unlabeled_masks", mask_name)
        dst_msk   = os.path.join(base_dir, labeled_dir,   "labeled_masks",   mask_name)
        if os.path.exists(src_msk):
            shutil.copy(src_msk, dst_msk)
            os.remove(src_msk)
        else:
            print(f"Warning: mask {src_msk} not found; skipping.")
        moved += 1

    print(f"Moved {moved} images from {unlabeled_dir} → {labeled_dir}.")


################################################################################
# 2.3) BUILD DATA LOADERS FOR ACTIVE LEARNING
################################################################################
class SegmentationFolder(Dataset):
    """
    A simple folder‐based segmentation dataset. Assumes:
      root/
        images/  (in your case, labeled_images or test_images)
        masks/   (labeled_masks or test_masks)
    Each image “xyz.jpg” → mask “xyz_mask.png”.
    Returns (img_tensor, mask_tensor, filename).
    """
    def __init__(self, root: str, transform=None):
        super().__init__()
        self.img_folder  = os.path.join(root, "images")
        self.mask_folder = os.path.join(root, "masks")
        self.transform   = transform
        self.images = [f for f in os.listdir(self.img_folder) if os.path.isfile(os.path.join(self.img_folder, f))]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_name  = self.images[idx]
        img_path = os.path.join(self.img_folder, im_name)
        mask_name= im_name.replace(".jpg", "_mask.png")
        msk_path = os.path.join(self.mask_folder, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(msk_path).convert("L")  # single‐channel

        img_t = TF.to_tensor(image)           # (3, H, W)
        msk_t = torch.from_numpy(np.array(mask)).long()  # (H, W) ints

        if self.transform:
            img_t, msk_t = self.transform(img_t, msk_t)

        return img_t, msk_t, im_name


def get_loaders_active(
    labeled_img_dir: str,
    labeled_mask_dir: str,
    unlabeled_img_dir: str,
    unlabeled_mask_dir: str,
    test_img_dir: str,
    test_mask_dir: str,
    batch_size: int,
    transform_labeled,
    transform_unlabeled,
    num_workers: int = 2
):
    """
    Return three DataLoaders for active learning:
      1) labeled_loader:   yields (img, mask, filename)
      2) unlabeled_loader: yields (img, filename)
      3) test_loader:      yields (img, mask, filename)
    """
    class LabeledDataset(Dataset):
        def __init__(self, img_dir, mask_dir, transform=None):
            super().__init__()
            self.img_folder  = img_dir
            self.mask_folder = mask_dir
            self.transform   = transform
            self.images      = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
            self.images.sort()

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            im_name = self.images[idx]
            img_path = os.path.join(self.img_folder, im_name)
            mask_name= im_name.replace(".jpg", "_mask.png")
            msk_path = os.path.join(self.mask_folder, mask_name)

            image = Image.open(img_path).convert("RGB")
            mask  = Image.open(msk_path).convert("L")
            img_t = TF.to_tensor(image)
            msk_t = torch.from_numpy(np.array(mask)).long()

            if self.transform:
                img_t, msk_t = self.transform(img_t, msk_t)
            return img_t, msk_t, im_name

    class UnlabeledDataset(Dataset):
        def __init__(self, img_dir, transform=None):
            super().__init__()
            self.img_folder = img_dir
            self.transform  = transform
            self.images     = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
            self.images.sort()

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            im_name = self.images[idx]
            img_path = os.path.join(self.img_folder, im_name)
            image = Image.open(img_path).convert("RGB")
            img_t  = TF.to_tensor(image)
            if self.transform:
                img_t = self.transform(img_t)
            return img_t, im_name  # note: no mask

    labeled_ds     = LabeledDataset(labeled_img_dir, labeled_mask_dir, transform_labeled)
    unlabeled_ds   = UnlabeledDataset(unlabeled_img_dir, transform_unlabeled)
    test_ds        = LabeledDataset(test_img_dir, test_mask_dir, transform_labeled)

    labeled_loader   = DataLoader(labeled_ds,   batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=1,           shuffle=False, num_workers=num_workers)
    test_loader      = DataLoader(test_ds,      batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return labeled_loader, unlabeled_loader, test_loader
