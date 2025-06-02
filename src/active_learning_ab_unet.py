# active_learning_ab_unet.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Make sure PIL and TF are imported before you use them in create_score_dict
from PIL import Image
import torchvision.transforms.functional as TF

from acquisition_functions import (
    score_entropy,
    score_kl_divergence,
    score_js_divergence,
    score_bald
)
from data_utils import (
    labeled_unlabeled_test_split,
    move_images_with_dict,
    get_loaders_active
)
from train import train_fn, check_accuracy_batch
from bayesian_unet import BayesianUNet


################################################################################
# 3.1) UTILITY: CREATE A SCORE DICTIONARY FOR A BATCH OF UNLABELED IMAGES
################################################################################
def create_score_dict(
        model: BayesianUNet,
        unlabeled_loader,
        device: str,
        acquisition_type: str = "bald",
        mc_runs: int = 5,
        num_classes: int = 4
) -> dict:
    """
    Given a trained model, an unlabeled_loader that yields (img, filename),
    compute acquisition score for each image and return an OrderedDict
    filename→score, sorted descending by score.

    acquisition_type in {"entropy", "bald", "kl", "js"}.
    mc_runs = number of MC‐Dropout passes (for entropy, KL, JS, or BALD).
    """
    scores = {}
    model.to(device)

    # Loop over each (img_tensor, filename) in the unlabeled subset
    for img, fname in unlabeled_loader:
        # img: shape (1, C, H, W)
        # fname: a single‐element tuple, e.g. ("IMG_1234.BMP",)
        T_probs = []

        if acquisition_type.lower() in ["entropy", "kl", "js"]:
            # 1) Run a single deterministic forward pass (no dropout)
            model.eval()
            with torch.no_grad():
                logits_std = model(img.to(device))
                p_std = torch.softmax(logits_std, dim=1).cpu().numpy()  # (1, C, H, W)

            # 2) Run T stochastic (dropout‐enabled) forward passes
            model.train()  # keep dropout layers “on”
            soft = nn.Softmax(dim=1)
            for _ in range(mc_runs):
                with torch.no_grad():
                    logits_t = model(img.to(device))
                    p_t = soft(logits_t).cpu().numpy()  # (1, C, H, W)
                    T_probs.append(p_t)
            T_probs = np.concatenate(T_probs, axis=0)  # (T, C, H, W)

            if acquisition_type.lower() == "entropy":
                # Just compute per‐pixel entropy on T_probs
                ent_scores = score_entropy(T_probs)    # shape: (T,) flattened across H×W
                score = ent_scores.mean()

            elif acquisition_type.lower() == "kl":
                # Compute KL divergence between p_std and each T_probs[t]
                P_std = np.repeat(p_std, mc_runs, axis=0)  # (T, C, H, W)
                kl_scores = score_kl_divergence(P_std, T_probs)
                score = kl_scores.mean()

            else:  # acquisition_type == "js"
                P_std = np.repeat(p_std, mc_runs, axis=0)
                js_scores = score_js_divergence(P_std, T_probs)
                score = js_scores.mean()

        else:
            # BALD: uses a specialized helper that runs T passes internally
            score = score_bald(
                model=model,
                image=img,
                T=mc_runs,
                device=device,
                num_classes=num_classes
            )

        # fname is a length‐1 tuple, so we use fname[0] as the key:
        scores[fname[0]] = score

    # Sort by descending score and return as an “ordered” dict:
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ordered_scores = {k: v for k, v in sorted_items}
    return ordered_scores


################################################################################
# 3.2) MAIN ACTIVE‐LEARNING LOOP
################################################################################
def active_learning_loop(
        BASE_DIR: str,
        LABEL_SPLIT_RATIO: float = 0.05,
        TEST_SPLIT_RATIO: float = 0.3,
        sample_size: int = 10,
        acquisition_type: str = "bald",
        mc_runs: int = 5,
        num_epochs: int = 5,
        batch_size: int = 4,
        lr: float = 1e-3,
        num_iterations: int = 10,
        device: str = "cuda"
):
    """
    1) Do the initial three‐way split of BASE_DIR/Images + BASE_DIR/masks into:
         BASE_DIR/Labeled_pool/{labeled_images/, labeled_masks/}
         BASE_DIR/Unlabeled_pool/{unlabeled_images/, unlabeled_masks/}
         BASE_DIR/Test/{test_images/, test_masks/}

    2) Then repeat `num_iterations` times:
       a) Build DataLoaders: labeled_loader, unlabeled_loader, test_loader
       b) Train a fresh AB‐UNet on labeled_loader for `num_epochs`
       c) Evaluate on test_loader and print test accuracy + Dice
       d) From Unlabeled_pool, pick a random subset of size `sample_size`.
          Use `create_score_dict` to compute acquisition scores on that subset.
       e) Move the top‐`sample_size` images from Unlabeled_pool → Labeled_pool
    """
    # 1) Ensure the three‐way split exists
    labeled_dir = "Labeled_pool"
    unlabeled_dir = "Unlabeled_pool"
    test_dir = "Test"

    labeled_unlabeled_test_split(
        base_dir=BASE_DIR,
        labeled_dir=labeled_dir,
        unlabeled_dir=unlabeled_dir,
        test_dir=test_dir,
        label_split_ratio=LABEL_SPLIT_RATIO,
        test_split_ratio=TEST_SPLIT_RATIO,
        shuffle=True
    )

    # 2) Active‐learning iterations
    for it in range(num_iterations):
        print(f"\n===== ACTIVE ITERATION {it + 1}/{num_iterations} =====")

        # 2a) Build DataLoaders
        LABELED_IMG_DIR     = os.path.join(BASE_DIR, labeled_dir,   "labeled_images")
        LABELED_MASK_DIR    = os.path.join(BASE_DIR, labeled_dir,   "labeled_masks")
        UNLABELED_IMG_DIR   = os.path.join(BASE_DIR, unlabeled_dir, "unlabeled_images")
        UNLABELED_MASK_DIR  = os.path.join(BASE_DIR, unlabeled_dir, "unlabeled_masks")
        TEST_IMG_DIR        = os.path.join(BASE_DIR, test_dir,      "test_images")
        TEST_MASK_DIR       = os.path.join(BASE_DIR, test_dir,      "test_masks")

        labeled_loader, unlabeled_loader, test_loader = get_loaders_active(
            labeled_img_dir     = LABELED_IMG_DIR,
            labeled_mask_dir    = LABELED_MASK_DIR,
            unlabeled_img_dir   = UNLABELED_IMG_DIR,
            unlabeled_mask_dir  = UNLABELED_MASK_DIR,
            test_img_dir        = TEST_IMG_DIR,
            test_mask_dir       = TEST_MASK_DIR,
            batch_size          = batch_size,
            transform_labeled   = None,   # you can plug in Albumentations or other transforms here
            transform_unlabeled = None,
            num_workers         = 4
        )

        # 2b) Train a fresh AB‐UNet on labeled data
        model = BayesianUNet(
            in_channels  = 3,
            out_channels = 4,
            features     = [64, 128, 256, 512],
            dropout_prob = 0.1
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn   = nn.CrossEntropyLoss()

        print(f"  Training on {len(labeled_loader.dataset)} labeled examples for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            train_fn(labeled_loader, model, optimizer, loss_fn, None)
            print(f"   Epoch {epoch + 1}/{num_epochs} complete.")

        # 2c) Evaluate on the test set
        print("  Evaluating on test set...")
        total_acc  = 0.0
        total_dice = 0.0
        n_batches  = 0
        for imgs_t, masks_t, _ in test_loader:
            imgs_t  = imgs_t.to(device)
            masks_t = masks_t.to(device)
            with torch.no_grad():
                _ = model(imgs_t)

            # CALL check_accuracy_batch with positional arguments (no keywords)
            batch_acc, batch_dice = check_accuracy_batch(
                imgs_t,
                masks_t,
                model,
                imgs_t.shape[0],
                device
            )
            total_acc  += batch_acc
            total_dice += batch_dice
            n_batches  += 1

        if n_batches > 0:
            print(f"   Test Accuracy (avg) = {total_acc / n_batches:.4f}   "
                  f"Test Dice (avg) = {total_dice / n_batches:.4f}")
        else:
            print("   [Warning] Test set is empty; skipping evaluation.")

        # 2d) From the Unlabeled pool, pick a random subset of size `sample_size` to score
        all_unlabeled_images = [
            f for f in os.listdir(UNLABELED_IMG_DIR)
            if os.path.isfile(os.path.join(UNLABELED_IMG_DIR, f))
        ]
        random.shuffle(all_unlabeled_images)
        subset_to_score = all_unlabeled_images[:sample_size]

        # We need a tiny loader that yields exactly those image tensors + filenames
        class SubsetUnlabeledLoader:
            def __init__(self, img_dir, subset_filenames):
                self.img_dir = img_dir
                self.subset  = subset_filenames

            def __iter__(self):
                for fname in self.subset:
                    img_path = os.path.join(self.img_dir, fname)
                    image = Image.open(img_path).convert("RGB")
                    img_t  = TF.to_tensor(image).unsqueeze(0)  # (1, 3, H, W)
                    yield img_t, fname

            def __len__(self):
                return len(self.subset)

        subset_loader = SubsetUnlabeledLoader(UNLABELED_IMG_DIR, subset_to_score)

        score_dict = create_score_dict(
            model            = model,
            unlabeled_loader = subset_loader,
            device           = device,
            acquisition_type = acquisition_type,
            mc_runs          = mc_runs,
            num_classes      = 4
        )

        # 2e) Move the top‐scoring `sample_size` images (and masks) over
        print(f"  Querying the top {sample_size} images to label according to '{acquisition_type}' scores.")
        move_images_with_dict(
            base_dir      = BASE_DIR,
            labeled_dir   = labeled_dir,
            unlabeled_dir = unlabeled_dir,
            score_dict    = score_dict,
            num_to_move   = sample_size
        )

    print("\n[Active Learning] Completed.")


################################################################################
# 4) DRIVER: call active_learning_loop(...) when this script is run directly
################################################################################
if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────────
    # Adjust BASE_DIR to wherever your “data/” folder lives. For example:
    #   • In Colab:     "/content/ab-unet/data"
    #   • In Kaggle:    "/kaggle/working/ab-unet/data"
    #   • Locally (from ab-unet/src): "../data"
    # ──────────────────────────────────────────────────────────────────────────
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data")
    )

    LABEL_SPLIT_RATIO   = 0.05
    TEST_SPLIT_RATIO    = 0.30
    SAMPLE_SIZE         = 10
    ACQUISITION_TYPE    = "bald"   # could also be "entropy", "kl", or "js"
    MC_RUNS             = 5
    NUM_EPOCHS          = 5
    BATCH_SIZE          = 4
    LEARNING_RATE       = 1e-3
    NUM_ITERATIONS      = 10
    DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

    active_learning_loop(
        BASE_DIR            = BASE_DIR,
        LABEL_SPLIT_RATIO   = LABEL_SPLIT_RATIO,
        TEST_SPLIT_RATIO    = TEST_SPLIT_RATIO,
        sample_size         = SAMPLE_SIZE,
        acquisition_type    = ACQUISITION_TYPE,
        mc_runs             = MC_RUNS,
        num_epochs          = NUM_EPOCHS,
        batch_size          = BATCH_SIZE,
        lr                  = LEARNING_RATE,
        num_iterations      = NUM_ITERATIONS,
        device              = DEVICE
    )
