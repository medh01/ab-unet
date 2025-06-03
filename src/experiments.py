# run_al_experiments.py

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# We assume these imports work relative to `src/`
from active_learning_ab_unet import create_score_dict, active_learning_loop  # we'll modify the loop below
from data_utils import (
    labeled_unlabeled_test_split,
    move_images_with_dict,
    get_loaders_active
)
from train import train_fn, check_accuracy_batch
from bayesian_unet import BayesianUNet

# ────────────────────────────────────────────────────────────────────────────────
# 1) First, figure out N = total number of raw images in data/Images
# ────────────────────────────────────────────────────────────────────────────────
DATA_ROOT = "/content/ab-unet/data"  # ← adjust to wherever your `data/` folder is
raw_imgs = [
    f for f in os.listdir(os.path.join(DATA_ROOT, "Images"))
    if os.path.isfile(os.path.join(DATA_ROOT, "Images", f)) and f.lower().endswith(".bmp")
]
N = len(raw_imgs)
print(f"Total raw images (N) = {N}")

# ────────────────────────────────────────────────────────────────────────────────
# 2) We will run ACTIVE LEARNING for each acquisition function in turn.
#    At each iteration, we record:
#      (number_of_labeled_images, test_dice).
#    As soon as %_labeled crosses 10%, 20%, 50%, and eventually 100%,
#    we pick the first test_dice that we obtain at (or just above) each threshold.
# ────────────────────────────────────────────────────────────────────────────────

# Which acquisition functions to compare
acq_list = ["random", "entropy", "bald", "js", "kl"]

# The percent thresholds at which we “snapshot” the test Dice
thresholds = [0.10, 0.20, 0.50, 1.00]  # 10%, 20%, 50%, 100%

# This dictionary will hold our final table data:
#    table_data[percent][acq_fn] = test_Dice
table_data = {p: {} for p in thresholds}


# ────────────────────────────────────────────────────────────────────────────────
# 3) A helper function that runs AL for *ONE* acquisition function
# ────────────────────────────────────────────────────────────────────────────────
def run_single_al(acq_type: str,
                  label_split_ratio=0.05,
                  test_split_ratio=0.30,
                  sample_size=10,
                  mc_runs=5,
                  num_epochs=5,
                  batch_size=4,
                  lr=1e-3,
                  device="cuda"):
    """
    Runs active learning until 100% of images are in the labeled pool.
    Returns a dict:  {
        percent_labeled_ever_seen: test_dice_at_that_iteration,
        ...
    }
    We'll only pick out the first time we cross each threshold.
    """

    # 3.1) Step 1: do the initial 3‐way split (Labeled/Unlabeled/Test)
    labeled_dir   = "Labeled_pool"
    unlabeled_dir = "Unlabeled_pool"
    test_dir      = "Test"

    # Delete existing splits if they exist, to start fresh:
    for sub in [labeled_dir, unlabeled_dir, test_dir]:
        full_path = os.path.join(DATA_ROOT, sub)
        if os.path.isdir(full_path):
            # Remove it completely (CAUTION: this deletes any previously moved files)
            import shutil
            shutil.rmtree(full_path)

    # Now re‐create a fresh split
    labeled_unlabeled_test_split(
        base_dir=DATA_ROOT,
        labeled_dir=labeled_dir,
        unlabeled_dir=unlabeled_dir,
        test_dir=test_dir,
        label_split_ratio=label_split_ratio,
        test_split_ratio=test_split_ratio,
        shuffle=True
    )

    # Compute how many images started in the labeled pool:
    labeled_images_root = os.path.join(DATA_ROOT, labeled_dir, "labeled_images")
    num_labeled = len(os.listdir(labeled_images_root))
    # We'll keep track of which threshold‐percentages we have already filled
    filled = {p: False for p in thresholds}
    result_for_this_acq = {}

    iteration = 0

    while True:
        iteration += 1
        print(f"\n--- AL iteration {iteration}  (algorithm = {acq_type}) ---")

        # 3.2) Build DataLoaders
        LAB_IMG_DIR = os.path.join(DATA_ROOT, labeled_dir,   "labeled_images")
        LAB_MSK_DIR = os.path.join(DATA_ROOT, labeled_dir,   "labeled_masks")
        UNL_IMG_DIR = os.path.join(DATA_ROOT, unlabeled_dir, "unlabeled_images")
        UNL_MSK_DIR = os.path.join(DATA_ROOT, unlabeled_dir, "unlabeled_masks")
        TEST_IMG_DIR= os.path.join(DATA_ROOT, test_dir,      "test_images")
        TEST_MSK_DIR= os.path.join(DATA_ROOT, test_dir,      "test_masks")

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

        # 3.3) Train a fresh Bayes‐UNet on *only* the current labeled set
        model     = BayesianUNet(in_channels=3, out_channels=4, features=[64,128,256,512], dropout_prob=0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn   = nn.CrossEntropyLoss()

        print(f"  Training on {len(labeled_loader.dataset)} labeled examples ...")
        for ep in range(num_epochs):
            train_fn(labeled_loader, model, optimizer, loss_fn, None)
            print(f"   → Epoch {ep+1}/{num_epochs} done.")

        # 3.4) Evaluate on the *test* set, compute average test Dice
        model.eval()
        total_dice = 0.0
        count_batches = 0
        with torch.no_grad():
            for (imgs_t, masks_t, _) in test_loader:
                imgs_t  = imgs_t.to(device)
                masks_t = masks_t.to(device)
                preds_t = model(imgs_t)
                batch_acc, batch_dice = check_accuracy_batch(imgs_t, masks_t, model, imgs_t.shape[0], device)
                total_dice += batch_dice.item()
                count_batches += 1
        avg_test_dice = (total_dice / count_batches) if (count_batches>0) else 0.0
        print(f"  → Test Dice = {avg_test_dice:.4f}")

        # 3.5) Check how many are labeled now, as a fraction of N
        num_labeled = len(os.listdir(LAB_IMG_DIR))
        frac_labeled = num_labeled / float(N)
        print(f"  → {num_labeled}/{N} labeled  (= {frac_labeled*100:.1f} %)")

        # 3.6) If we just crossed any threshold (10%,20%,50%,100%), record it
        for p in thresholds:
            if ( (frac_labeled >= p) and (not filled[p]) ):
                filled[p] = True
                result_for_this_acq[p] = avg_test_dice
                print(f"    ** recorded at {p*100:.0f}% → Dice = {avg_test_dice:.4f} **")

        # 3.7) If the labeled pool is now 100% of N, break
        if num_labeled >= N:
            break

        # 3.8) Otherwise, pick a *random subset* of size `sample_size` from the Unlabeled pool,
        #      score them, and move the top‐`sample_size` → labeled.
        all_unlabeled = [
            f for f in os.listdir(UNL_IMG_DIR)
            if os.path.isfile(os.path.join(UNL_IMG_DIR, f))
        ]
        random.shuffle(all_unlabeled)
        subset_to_score = all_unlabeled[:sample_size]

        # Build a tiny DataLoader that yields (tensor, filename) for those images
        class SubsetUnlabeledLoader:
            def __init__(self, img_dir, filenames):
                self.img_dir = img_dir
                self.files   = filenames
            def __iter__(self):
                for fn in self.files:
                    full = os.path.join(self.img_dir, fn)
                    img  = Image.open(full).convert("RGB")
                    t    = TF.to_tensor(img).unsqueeze(0)  # (1,3,H,W)
                    yield t, fn
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

        # Move the top‐`sample_size` scored images into the labeled pool
        move_images_with_dict(
            base_dir      = DATA_ROOT,
            labeled_dir   = labeled_dir,
            unlabeled_dir = unlabeled_dir,
            score_dict    = score_dict,
            num_to_move   = sample_size
        )

    # done entire loop; return the dict of percent→Dice
    return result_for_this_acq


# ────────────────────────────────────────────────────────────────────────────────
# 4) Now run for each acquisition function, collect results
# ────────────────────────────────────────────────────────────────────────────────
for acq in acq_list:
    print(f"\n================ Running AL with acquisition = '{acq}' ================")
    results = run_single_al(
        acq_type=acq,
        label_split_ratio=0.05,
        test_split_ratio=0.30,
        sample_size= int(0.05 * N),  # for example, 5% of N per iteration
        mc_runs=5,
        num_epochs=5,
        batch_size=4,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # `results` is something like {0.10:0.739, 0.20:0.773, 0.50:0.784, 1.00:0.786}
    # Fill our table_data[percent][acq] = dice
    for p in thresholds:
        table_data[p][acq] = results.get(p, np.nan)

# ────────────────────────────────────────────────────────────────────────────────
# 5) Build a pandas DataFrame and print it (this is the “table”)
# ────────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame.from_dict(table_data, orient="index")
df = df[acq_list]  # reorder columns to [“random”,“entropy”,“bald”,“js”,“kl”]
df.index = ["10 %", "20 %", "50 %", "100 %"]
print("\n—— Final Active‐Learning Dice Scores ——")
print(df.to_markdown(tablefmt="github", floatfmt=".3f"))
# You can also save to CSV:
df.to_csv("al_dice_table.csv")

# If you want to inspect in a notebook:
# display(df)

# ────────────────────────────────────────────────────────────────────────────────
# 6) Finally, plot Dice vs. % for each acquisition function
# ────────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
x = [10, 20, 50, 100]

for acq in acq_list:
    y = [ df.loc["10 %", acq],
          df.loc["20 %", acq],
          df.loc["50 %", acq],
          df.loc["100 %", acq] ]
    plt.plot(x, y, marker="o", label=acq)

plt.title("Active‐Learning Test Dice vs. % of Dataset")
plt.xlabel("% of dataset labeled")
plt.ylabel("Test Dice")
plt.xticks(x)
plt.ylim(0.7, 0.82)   # adjust as needed
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Acquisition fn.", loc="lower right")
plt.tight_layout()
plt.savefig("al_dice_plot.png", dpi=200)
plt.show()
