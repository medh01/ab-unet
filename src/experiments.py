# experiments.py

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Adjust this path if your data folder is elsewhere
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

from active_learning_ab_unet import create_score_dict
from data_utils          import (
    labeled_unlabeled_test_split,
    move_images_with_dict,
    get_loaders_active
)
from train               import train_fn, check_accuracy_batch
from bayesian_unet       import BayesianUNet

# 1) Count total number of raw .BMP images in data/Images
raw_imgs = [
    f for f in os.listdir(os.path.join(DATA_ROOT, "Images"))
    if os.path.isfile(os.path.join(DATA_ROOT, "Images", f)) and f.lower().endswith(".bmp")
]
N = len(raw_imgs)
print(f"Total raw images (N) = {N}")

# 2) Active‐learning will be run for each of these acquisition functions:
acq_list   = ["random", "entropy", "bald", "js", "kl"]
thresholds = [0.10, 0.20, 0.50, 1.00]

# Prepare a data structure to hold {threshold → {acq_fn → test_dice}}
table_data = {p: {} for p in thresholds}

def run_single_al(
        acq_type: str,
        label_split_ratio=0.05,
        test_split_ratio=0.30,
        sample_size=10,
        mc_runs=5,
        num_epochs=5,
        batch_size=4,
        lr=1e-3,
        device="cuda"
    ):
    """
    Executes one full active‐learning run for a given acquisition type.
    Returns a dict mapping each threshold (0.10, 0.20, 0.50, 1.00) to the
    test‐Dice when at least that fraction of images have been labeled.
    """

    # 1) Clear any existing Labeled_pool, Unlabeled_pool, Test folders
    for subdir in ["Labeled_pool", "Unlabeled_pool", "Test"]:
        fullpath = os.path.join(DATA_ROOT, subdir)
        if os.path.isdir(fullpath):
            import shutil
            shutil.rmtree(fullpath)

    # 2) Create a fresh three‐way split
    labeled_unlabeled_test_split(
        base_dir          = DATA_ROOT,
        labeled_dir       = "Labeled_pool",
        unlabeled_dir     = "Unlabeled_pool",
        test_dir          = "Test",
        label_split_ratio = label_split_ratio,
        test_split_ratio  = test_split_ratio,
        shuffle           = True
    )

    filled = {p: False for p in thresholds}
    results_for_acq = {}

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- AL iteration {iteration}  (acquisition = '{acq_type}') ---")

        # 3) Build DataLoaders for current pools
        LAB_IMG_DIR  = os.path.join(DATA_ROOT, "Labeled_pool",   "labeled_images")
        LAB_MSK_DIR  = os.path.join(DATA_ROOT, "Labeled_pool",   "labeled_masks")
        UNL_IMG_DIR  = os.path.join(DATA_ROOT, "Unlabeled_pool", "unlabeled_images")
        UNL_MSK_DIR  = os.path.join(DATA_ROOT, "Unlabeled_pool", "unlabeled_masks")
        TEST_IMG_DIR = os.path.join(DATA_ROOT, "Test",           "test_images")
        TEST_MSK_DIR = os.path.join(DATA_ROOT, "Test",           "test_masks")

        labeled_loader, unlabeled_loader, test_loader = get_loaders_active(
            labeled_img_dir     = LAB_IMG_DIR,
            labeled_mask_dir    = LAB_MSK_DIR,
            unlabeled_img_dir   = UNL_IMG_DIR,
            unlabeled_mask_dir  = UNL_MSK_DIR,
            test_img_dir        = TEST_IMG_DIR,
            test_mask_dir       = TEST_MSK_DIR,
            batch_size          = batch_size,
            transform_labeled   = None,
            transform_unlabeled = None,
            num_workers         = 2
        )

        # 4) Train a fresh BayesianUNet on the labeled pool
        model     = BayesianUNet(in_channels=3, out_channels=4,
                                 features=[64,128,256,512], dropout_prob=0.1
                                ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn   = nn.CrossEntropyLoss()

        print(f"  Training on {len(labeled_loader.dataset)} labeled examples ...")
        for ep in range(num_epochs):
            train_fn(labeled_loader, model, optimizer, loss_fn, None)
            print(f"   → Epoch {ep+1}/{num_epochs} done.")

        # 5) Evaluate on the test set and compute average test Dice
        model.eval()
        total_dice   = 0.0
        batch_count  = 0
        with torch.no_grad():
            for imgs_t, masks_t, _ in test_loader:
                imgs_t  = imgs_t.to(device)
                masks_t = masks_t.to(device)
                _ = model(imgs_t)
                _, batch_dice = check_accuracy_batch(
                    imgs_t, masks_t, model, imgs_t.shape[0], device
                )
                total_dice  += batch_dice.item()
                batch_count += 1

        avg_test_dice = (total_dice / batch_count) if (batch_count > 0) else 0.0
        print(f"  → Test Dice = {avg_test_dice:.4f}")

        # 6) Compute fraction labeled so far
        num_labeled   = len(os.listdir(LAB_IMG_DIR))
        frac_labeled  = num_labeled / float(N)
        print(f"  → {num_labeled}/{N} labeled  (={frac_labeled*100:.1f} %)")

        # 7) Record Dice for any new threshold passed
        for p in thresholds:
            if frac_labeled >= p and not filled[p]:
                filled[p] = True
                results_for_acq[p] = avg_test_dice
                print(f"    ** Recorded at {int(p*100)}% → Dice = {avg_test_dice:.4f} **")

        # 8) If all images are labeled, stop
        if num_labeled >= N:
            print("  → Labeled pool is 100% of N; stopping AL loop.\n")
            break

        # 9) Otherwise, sample `sample_size` from unlabeled to score & move
        all_unlabeled = [
            f for f in os.listdir(UNL_IMG_DIR)
            if os.path.isfile(os.path.join(UNL_IMG_DIR, f))
        ]
        random.shuffle(all_unlabeled)
        subset_to_score = all_unlabeled[:sample_size]

        # Build a tiny loader for that subset
        class SubsetUnlabeledLoader:
            def __init__(self, img_dir, filenames):
                self.img_dir = img_dir
                self.files   = filenames

            def __iter__(self):
                for fn in self.files:
                    full   = os.path.join(self.img_dir, fn)
                    img    = Image.open(full).convert("RGB")
                    tensor = TF.to_tensor(img).unsqueeze(0)  # (1,3,H,W)
                    yield tensor, fn

            def __len__(self):
                return len(self.files)

        subset_loader = SubsetUnlabeledLoader(UNL_IMG_DIR, subset_to_score)
        score_dict = create_score_dict(
            model            = model,
            unlabeled_loader = subset_loader,
            device           = device,
            acquisition_type = acq_type,
            mc_runs          = mc_runs,
            num_classes      = 4
        )

        move_images_with_dict(
            base_dir      = DATA_ROOT,
            labeled_dir   = "Labeled_pool",
            unlabeled_dir = "Unlabeled_pool",
            score_dict    = score_dict,
            num_to_move   = sample_size
        )

    return results_for_acq


# 10) Execute active‐learning runs for each acquisition function
for acq in acq_list:
    print(f"\n================ Running AL with acquisition = '{acq}' ================")
    results_dict = run_single_al(
        acq_type        = acq,
        label_split_ratio = 0.05,
        test_split_ratio  = 0.30,
        sample_size       = int(0.05 * N),  # e.g. 5% of N each iteration
        mc_runs           = 5,
        num_epochs        = 5,
        batch_size        = 4,
        lr                = 1e-3,
        device            = "cuda" if torch.cuda.is_available() else "cpu"
    )
    for p in thresholds:
        table_data[p][acq] = results_dict.get(p, np.nan)


# 11) Build a pandas DataFrame and print neatly
df = pd.DataFrame.from_dict(table_data, orient="index")
df = df[acq_list]  # order columns exactly as ["random","entropy","bald","js","kl"]
df.index = ["10 %", "20 %", "50 %", "100 %"]

print("\n—— Final Active‐Learning Dice Scores ———")
print(df.to_markdown(tablefmt="github", floatfmt=".3f"))
df.to_csv("al_dice_table.csv")


# 12) Plot Dice vs. % labeled for each acquisition function
plt.figure(figsize=(7,4))
x = [10, 20, 50, 100]
for acq in acq_list:
    y = [
        df.loc["10 %",  acq],
        df.loc["20 %",  acq],
        df.loc["50 %",  acq],
        df.loc["100 %", acq]
    ]
    plt.plot(x, y, marker="o", label=acq)

plt.title("Active‐Learning Test Dice vs. % of Dataset")
plt.xlabel("% of dataset labeled")
plt.ylabel("Test Dice")
plt.xticks(x)
plt.ylim(0.70, 0.82)
plt.grid(linestyle="--", alpha=0.4)
plt.legend(title="Acquisition fn.", loc="lower right")
plt.tight_layout()
plt.savefig("al_dice_plot.png", dpi=200)
plt.show()