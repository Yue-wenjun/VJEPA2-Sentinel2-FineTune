"""
Cloud Ablation Experiment Definitions

Cloud category mapping (set during download in download_s2.py):
    -1 = SCL unavailable  (N0500+ baseline; spectral bands still valid)
     0 = clean            cloud_frac < 0.10
     1 = moderate         0.10 <= cloud_frac < 0.30
     2 = cloudy           cloud_frac >= 0.30

Five experiment groups to test whether cloud-contaminated frames hurt embedding quality.
Core hypothesis: V-JEPA's masked prediction is robust to cloud as a "natural mask",
so Group D embedding quality ≈ Group A despite training on all frames.

Usage:
    from data_pipeline.cloud_ablation import GROUPS, make_ablation_dataset

    train_ds = make_ablation_dataset("D", sequences_csv, split="train")
    eval_ds  = make_ablation_dataset("D", sequences_csv, split="eval")
"""

from dataclasses import dataclass

from data_pipeline.sentinel2_dataset import Sentinel2Dataset


@dataclass
class AblationGroup:
    name: str
    desc: str
    train_cats: list[int] | None  # None = all categories
    eval_cats:  list[int] | None


GROUPS: dict[str, AblationGroup] = {
    "A": AblationGroup(
        name="A", desc="clean baseline",
        train_cats=[0], eval_cats=[0],
    ),
    "B": AblationGroup(
        name="B", desc="clean + moderate (pipeline default)",
        train_cats=[0, 1], eval_cats=[0, 1],
    ),
    "C": AblationGroup(
        name="C", desc="all frames including cloudy and SCL-unknown",
        train_cats=None, eval_cats=None,           # None = no filter, includes cat=-1
    ),
    "D": AblationGroup(
        name="D", desc="train on all frames, evaluate on clean only",
        train_cats=None, eval_cats=[0],            # train includes cat=-1
    ),
    "E": AblationGroup(
        name="E", desc="train on clean+moderate, evaluate on all frames",
        train_cats=[0, 1], eval_cats=None,         # eval includes cat=-1
    ),
}


def make_ablation_dataset(
    group: str,
    sequences_csv: str,
    split: str = "train",
    base_dir: str = "",
    frames_per_clip: int = 8,
    crop_size: int = 384,
) -> Sentinel2Dataset:
    """
    Create a Sentinel2Dataset for the given ablation group and split.

    Args:
        group:  one of "A", "B", "C", "D", "E"
        split:  "train" or "eval"
        sequences_csv: path to sequences.csv produced by download_s2.py
        base_dir: optional base directory for resolving relative paths in CSV

    Returns:
        Sentinel2Dataset filtered to the appropriate cloud categories.

    Example — reproduce Group D experiment:
        train_ds = make_ablation_dataset("D", "s2_data/sequences.csv", split="train")
        eval_ds  = make_ablation_dataset("D", "s2_data/sequences.csv", split="eval")
    """
    if group not in GROUPS:
        raise ValueError(f"Unknown group '{group}'. Choose from: {list(GROUPS)}")
    cfg = GROUPS[group]
    cats = cfg.train_cats if split == "train" else cfg.eval_cats
    return Sentinel2Dataset(
        sequences_csv=sequences_csv,
        base_dir=base_dir,
        cloud_cats=cats,
        frames_per_clip=frames_per_clip,
        crop_size=crop_size,
    )


def print_group_summary(sequences_csv: str, base_dir: str = "") -> None:
    """Print sequence counts for all five groups (train split)."""
    print(f"{'Group':<6} {'Train cats':<14} {'Eval cats':<14} {'#sequences':<12} Description")
    print("-" * 72)
    for key, g in GROUPS.items():
        ds = make_ablation_dataset(key, sequences_csv, split="train", base_dir=base_dir)
        train_str = str(g.train_cats) if g.train_cats is not None else "all"
        eval_str  = str(g.eval_cats)  if g.eval_cats  is not None else "all"
        print(f"{key:<6} {train_str:<14} {eval_str:<14} {len(ds):<12} {g.desc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarise cloud ablation group sizes")
    parser.add_argument("sequences_csv")
    parser.add_argument("--base_dir", default="")
    args = parser.parse_args()
    print_group_summary(args.sequences_csv, args.base_dir)
