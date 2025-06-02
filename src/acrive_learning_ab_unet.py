# active_learning_ab_unet.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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

    for img, fname in unlabeled_loader:
        # img: (1, C, H, W), fname: string
        # 1) Run T forward passes and collect softmax outputs (NumPy)
        T_probs = []
        if acquisition_type.lower() in ["entropy", "kl", "js"]:
            # We need both a deterministic softmax AND T stochastic softmaxes.
            model.eval()
            with torch.no_grad():
                logits_std = model(img.to(device))
                p_std = torch.softmax(logits_std, dim=1).cpu().numpy()  # shape (1, C, H, W)
            # MC‐dropout passes:
            model.train()
            soft = nn.Softmax(dim=1)
            for _ in range(mc_runs):
                with torch.no_grad():
                    logits_t = model(img.to(device))
                    p_t = soft(logits_t).cpu().numpy()  # shape (1, C, H, W)
                    T_probs.append(p_t)
            T_probs = np.concatenate(T_probs, axis=0)  # shape (T, C, H, W)

            if acquisition_type.lower() == "entropy":
                # We only need T_probs → shape (T, C, H, W)
                ent_scores = score_entropy(T_probs)
                score = ent_scores.mean()  # average over T
            elif acquisition_type.lower() == "kl":
                # We need p_std repeated T times → shape (T, C, H, W)
                P_std = np.repeat(p_std, mc_runs, axis=0)
                kl_scores = score_kl_divergence(P_std, T_probs)
                score = kl_scores.mean()
            else:  # "js"
                P_std = np.repeat(p_std, mc_runs, axis=0)
                js_scores = score_js_divergence(P_std, T_probs)
                score = js_scores.mean()

        else:  # acquisition_type == "bald"
            score = score_bald(model, img, T=mc_runs, device=device, num_classes=num_classes)

        scores[fname[0]] = score

    # Sort dict by descending score
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
    1) Initial split: BASE_DIR/images/ →
         BASE_DIR/Labeled_pool/{labeled_images/, labeled_masks/}
         BASE_DIR/Unlabeled_pool/{unlabeled_images/, unlabeled_masks/}
         BASE_DIR/Test/{test_images/, test_masks/}
    2) Repeat num_iterations times:
         a) Build DataLoaders: labeled_loader, unlabeled_loader, test_loader
         b) Train model for num_epochs on labeled_loader (supervised)
         c) Score a random subset of size sample_size from unlabeled_loader:
              - Use create_score_dict(...) → top filenames
         d) Move top sample_size images (and masks) from Unlabeled_pool → Labeled_pool
         e) Evaluate on test_loader (optional: log dice)
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

    # 2) Loop
    for it in range(num_iterations):
        print(f"\n===== ACTIVE ITERATION {it + 1}/{num_iterations} =====")
        # 2a) Build DataLoaders
        LABELED_IMG_DIR = os.path.join(BASE_DIR, labeled_dir, "labeled_images")
        LABELED_MASK_DIR = os.path.join(BASE_DIR, labeled_dir, "labeled_masks")
        UNLABELED_IMG_DIR = os.path.join(BASE_DIR, unlabeled_dir, "unlabeled_images")
        UNLABELED_MASK_DIR = os.path.join(BASE_DIR, unlabeled_dir, "unlabeled_masks")
        TEST_IMG_DIR = os.path.join(BASE_DIR, test_dir, "test_images")
        TEST_MASK_DIR = os.path.join(BASE_DIR, test_dir, "test_masks")

        # (a) Trainers usually have transforms defined; here we pass identity transforms
        labeled_loader, unlabeled_loader, test_loader = get_loaders_active(
            labeled_img_dir=LABELED_IMG_DIR,
            labeled_mask_dir=LABELED_MASK_DIR,
            unlabeled_img_dir=UNLABELED_IMG_DIR,
            unlabeled_mask_dir=UNLABELED_MASK_DIR,
            test_img_dir=TEST_IMG_DIR,
            test_mask_dir=TEST_MASK_DIR,
            batch_size=batch_size,
            transform_labeled=None,  # define your own transforms if desired
            transform_unlabeled=None,
            num_workers=4
        )

        # (b) Train AB-UNet on labeled data
        model = BayesianUNet(in_channels=3, out_channels=4, features=[64, 128, 256, 512], dropout_prob=0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        print(f"  Training on {len(labeled_loader.dataset)} labeled examples for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            train_fn(labeled_loader, model, optimizer, loss_fn, None)  # using your train_fn
            print(f"   Epoch {epoch + 1}/{num_epochs} complete.")

        # Optionally compute metrics on test set
        print("  Evaluating on test set...")
        total_acc = 0.0
        total_dice = 0.0
        n_batches = 0
        for imgs_t, masks_t, _ in test_loader:
            imgs_t = imgs_t.to(device)
            masks_t = masks_t.to(device)
            with torch.no_grad():
                preds_t = model(imgs_t)
            batch_acc, batch_dice = check_accuracy_batch(imgs_t, masks_t, model, imgs_t.shape[0], device=device)
            total_acc += batch_acc.item()
            total_dice += batch_dice.item()
            n_batches += 1
        print(f"   Test Accuracy (avg) = {total_acc / n_batches:.4f}   Test Dice (avg) = {total_dice / n_batches:.4f}")

        # (c) From the Unlabeled pool, pick a random SAMPLE of size `sample_size` to score
        all_unlabeled_images = [f for f in os.listdir(UNLABELED_IMG_DIR) if
                                os.path.isfile(os.path.join(UNLABELED_IMG_DIR, f))]
        random.shuffle(all_unlabeled_images)
        subset_to_score = all_unlabeled_images[:sample_size]

        # We need to score each image via our acquisition function. But `create_score_dict`
        # expects a DataLoader. So we build a tiny custom DataLoader that yields only those images:

        class SubsetUnlabeledLoader:
            def __init__(self, img_dir, subset_filenames):
                self.img_dir = img_dir
                self.subset = subset_filenames

            def __iter__(self):
                for fname in self.subset:
                    img_path = os.path.join(self.img_dir, fname)
                    image = Image.open(img_path).convert("RGB")
                    img_t = TF.to_tensor(image).unsqueeze(0)  # shape (1, 3, H, W)
                    yield img_t, fname

            def __len__(self):
                return len(self.subset)

        from PIL import Image
        import torchvision.transforms.functional as TF

        subset_loader = SubsetUnlabeledLoader(UNLABELED_IMG_DIR, subset_to_score)
        score_dict = create_score_dict(
            model=model,
            unlabeled_loader=subset_loader,
            device=device,
            acquisition_type=acquisition_type,
            mc_runs=mc_runs,
            num_classes=4
        )

        # (d) Move top‐scoring `sample_size` images from Unlabeled → Labeled
        print(f"  Querying the top {sample_size} images to label, "
              f"according to {acquisition_type} scores.")
        move_images_with_dict(
            base_dir=BASE_DIR,
            labeled_dir=labeled_dir,
            unlabeled_dir=unlabeled_dir,
            score_dict=score_dict,
            num_to_move=sample_size
        )

    print("\n[Active Learning] Completed.")
