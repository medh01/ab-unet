import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

# Import the modified function (which now returns history)
from active_learning_ab_unet import active_learning_loop

def run_active_learning_experiments(
    base_dir,
    label_split_ratio=0.05,
    test_split_ratio=0.3,
    sample_size=10,
    mc_runs=5,
    num_epochs=5,
    batch_size=4,
    learning_rate=1e-3,
    num_iterations=10,
    device="cuda"
):
    """
    Run active-learning with different acquisition functions ("bald","entropy","kl","js","random").
    Returns a dict mapping each method to its per-iteration metrics list.
    """
    acquisition_methods = ["bald", "entropy", "kl", "js", "random"]
    all_results = {}

    for acq in acquisition_methods:
        print(f"\n=== Running active learning with acquisition = {acq} ===")
        history = active_learning_loop(
            BASE_DIR=base_dir,
            LABEL_SPLIT_RATIO=label_split_ratio,
            TEST_SPLIT_RATIO=test_split_ratio,
            sample_size=sample_size,
            acquisition_type=acq,
            mc_runs=mc_runs,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            num_iterations=num_iterations,
            device=device
        )
        all_results[acq] = history

    return all_results

if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────────
    # Adjust BASE_DIR to your data folder (which must contain “Images/” and “masks/”)
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Active-learning hyperparameters (you can tweak as needed)
    LABEL_SPLIT_RATIO = 0.05
    TEST_SPLIT_RATIO  = 0.30
    SAMPLE_SIZE       = 10
    MC_RUNS           = 5
    NUM_EPOCHS        = 5
    BATCH_SIZE        = 4
    LEARNING_RATE     = 1e-3
    NUM_ITERATIONS    = 10

    # Run the experiments
    results = run_active_learning_experiments(
        base_dir=BASE_DIR,
        label_split_ratio=LABEL_SPLIT_RATIO,
        test_split_ratio=TEST_SPLIT_RATIO,
        sample_size=SAMPLE_SIZE,
        mc_runs=MC_RUNS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_iterations=NUM_ITERATIONS,
        device=DEVICE
    )

    # Save raw results to JSON
    with open("ab_unet_acquisition_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot “Test Dice vs. Dataset Size”
    plt.figure(figsize=(8, 6))
    for acq, history in results.items():
        dice_scores = [h["test_dice"] for h in history]
        labeled_sizes = [h["labeled_size"] for h in history]
        max_labeled = labeled_sizes[-1]
        fractions = [size / max_labeled for size in labeled_sizes]

        plt.plot(
            fractions,
            dice_scores,
            marker='o',
            label=acq.capitalize()
        )

    plt.xlabel("Dataset Size (fraction of final labeled set)")
    plt.ylabel("Dice Metric (Test Set)")
    plt.title("Test Dice vs. Dataset Size for Different Acquisition Functions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_dice_vs_dataset_size.png", dpi=200)
    plt.show()
