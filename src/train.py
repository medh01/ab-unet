# train.py

import os
import shutil
import random
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import your BayesianUNet (or AB_UNET) class
from bayesian_unet import BayesianUNet

# Import the SegmentationFolder and any needed utilities from data_utils.py
# (we assume data_utils.py already contains SegmentationFolder, TARGET_SIZE)
from data_utils import SegmentationFolder, TARGET_SIZE

# Import accuracy/dice helpers (you should have these in utils.py)
from utils import check_accuracy, check_accuracy_batch

# TensorBoard (optional)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/AB-UNET')

# Hyperparameters
LEARNING_RATE = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 2
NUM_EPOCHS    = 20
NUM_WORKERS   = 2

# Paths (adjust if your directory structure differs)
BASE_DATA_DIR  = "./DATA_AO_preprocessed/labeled_pool"
TRAIN_SUBDIR   = "TRAIN"
VAL_SUBDIR     = "VAL"
TRAIN_IMG_DIR  = os.path.join(BASE_DATA_DIR, TRAIN_SUBDIR, "train_images")
TRAIN_MASK_DIR = os.path.join(BASE_DATA_DIR, TRAIN_SUBDIR, "train_masks")
VAL_IMG_DIR    = os.path.join(BASE_DATA_DIR, VAL_SUBDIR,   "val_images")
VAL_MASK_DIR   = os.path.join(BASE_DATA_DIR, VAL_SUBDIR,   "val_masks")


# ─────────────────────────────────────────────────────────────────────────────
def reset_DATA(base_dir: str):
    """
    Delete any existing TRAIN/ and VAL/ subdirectories under base_dir,
    so that we can recreate them from scratch.
    """
    for sub in [TRAIN_SUBDIR, VAL_SUBDIR]:
        full_path = os.path.join(base_dir, sub)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)


def train_val_split(
    base_dir: str,
    train_subdir: str,
    val_subdir: str,
    split_ratio: float = 0.8,
    shuffle: bool = True
):
    """
    Split all .BMP images (with matching .png masks) in base_dir into:
      base_dir/train_subdir/{train_images, train_masks}
      base_dir/val_subdir/{val_images,   val_masks}

    Uses split_ratio (e.g. 0.8) to decide how many go to TRAIN vs VAL.
    Assumes:
      base_dir/
        Images/   (raw .BMP files)
        masks/    (raw .png masks, same basename as .BMP)
    """
    img_folder  = os.path.join(base_dir, "Images")
    mask_folder = os.path.join(base_dir, "masks")

    # List all .BMP files
    all_images = [
        f for f in os.listdir(img_folder)
        if os.path.isfile(os.path.join(img_folder, f)) and f.lower().endswith(".bmp")
    ]
    if shuffle:
        random.shuffle(all_images)

    N       = len(all_images)
    n_train = int(N * split_ratio)
    n_val   = N - n_train

    # Create TRAIN and VAL subfolders
    train_root = os.path.join(base_dir, train_subdir)
    val_root   = os.path.join(base_dir, val_subdir)
    os.makedirs(os.path.join(train_root, "train_images"), exist_ok=True)
    os.makedirs(os.path.join(train_root, "train_masks"),  exist_ok=True)
    os.makedirs(os.path.join(val_root,   "val_images"),   exist_ok=True)
    os.makedirs(os.path.join(val_root,   "val_masks"),    exist_ok=True)

    # Copy first n_train into TRAIN, the rest into VAL
    for i, im_name in enumerate(all_images):
        base_name = os.path.splitext(im_name)[0]    # e.g. "XYZ"
        mask_name = base_name + ".png"              # e.g. "XYZ.png"

        src_im = os.path.join(img_folder, im_name)
        src_mk = os.path.join(mask_folder, mask_name)

        if i < n_train:
            dst_im = os.path.join(train_root, "train_images", im_name)
            dst_mk = os.path.join(train_root, "train_masks",  mask_name)
        else:
            dst_im = os.path.join(val_root,   "val_images",   im_name)
            dst_mk = os.path.join(val_root,   "val_masks",    mask_name)

        shutil.copy(src_im, dst_im)
        if os.path.exists(src_mk):
            shutil.copy(src_mk, dst_mk)
        else:
            raise FileNotFoundError(f"Mask {src_mk} not found for image {im_name}")


# ─────────────────────────────────────────────────────────────────────────────
def train_fn(loader, model, optimizer, loss_fn, scheduler):
    """
    One epoch of supervised training on `loader`.
    Automatically sends inputs & targets to the same device as `model`.
    """
    loop = tqdm(loader, desc="Training", leave=False)
    acc_list, dice_list = [], []

    device = next(model.parameters()).device

    for batch_idx, (imgs, masks, _) in enumerate(loop):
        # 1) Move to device
        imgs  = imgs.float().to(device)
        masks = masks.long().to(device)

        # 2) Forward + loss
        with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss  = loss_fn(preds, masks)

        # 3) Backward + optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            # We step the scheduler on the loss if it is a ReduceLROnPlateau
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        # 4) Compute batch metrics
        batch_acc, batch_dice = check_accuracy_batch(imgs, masks, model, imgs.shape[0], device=device)
        acc_list.append(batch_acc.item())
        dice_list.append(batch_dice.item())

        # 5) Update progress bar
        loop.set_postfix({
            "loss": loss.item(),
            "acc":  np.mean(acc_list),
            "dice": np.mean(dice_list)
        })

    loop.close()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1) Re-split the labeled pool into TRAIN / VAL (80/20 by default).
    reset_DATA(BASE_DATA_DIR)
    train_val_split(BASE_DATA_DIR, TRAIN_SUBDIR, VAL_SUBDIR, split_ratio=0.8, shuffle=True)

    # 2) Define transforms
    train_transform = A.Compose([
        # (Optional) more augmentations here
        A.Equalize(mode="cv", p=1.0),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        ToTensorV2()
    ])

    # 3) Create Datasets & DataLoaders
    train_dataset = SegmentationFolder(root=os.path.join(BASE_DATA_DIR, TRAIN_SUBDIR),
                                       transform=train_transform)
    val_dataset   = SegmentationFolder(root=os.path.join(BASE_DATA_DIR, VAL_SUBDIR),
                                       transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Found {len(train_dataset)} training images, {len(val_dataset)} validation images.")
    print(f"Each image/mask has been resized to {TARGET_SIZE[0]}×{TARGET_SIZE[1]} before tensor conversion.")

    # 4) Initialize model, loss, optimizer, scheduler
    model = BayesianUNet(in_channels=3, out_channels=4, features=[64, 128, 256, 512], dropout_prob=0.1)
    model = model.to(DEVICE)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, verbose=True)

    # 5) Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        # 5a) Train for one epoch
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn, scheduler)

        # 5b) Evaluate on validation set
        model.eval()
        val_acc_total, val_dice_total, count_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs  = imgs.float().to(DEVICE)
                masks = masks.long().to(DEVICE)
                _     = model(imgs)

                batch_acc, batch_dice = check_accuracy_batch(imgs, masks, model, imgs.shape[0], device=DEVICE)
                val_acc_total  += batch_acc.item()
                val_dice_total += batch_dice.item()
                count_batches  += 1

        if count_batches > 0:
            avg_val_acc  = val_acc_total  / count_batches
            avg_val_dice = val_dice_total / count_batches
        else:
            avg_val_acc = avg_val_dice = 0.0

        print(f"Validation  →  Acc: {avg_val_acc:.4f},  Dice: {avg_val_dice:.4f}")

    # 6) Save the final model weights
    torch.save(model.state_dict(), "bayesian_unet_final.pth")
    print("\nTraining complete. Saved model as bayesian_unet_final.pth")

    writer.close()


if __name__ == "__main__":
    main()
