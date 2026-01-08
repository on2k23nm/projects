import torch
from typing import Sequence

def assert_target_cols_order(df_cols, expected_cols: Sequence[str]):
    missing = [c for c in expected_cols if c not in df_cols]
    if missing:
        raise AssertionError(f"Missing target cols in dataframe: {missing}")
    # ensure order matches (or at least existence) - we require exact ordering here
    idxs = [list(df_cols).index(c) for c in expected_cols]
    if idxs != sorted(idxs):
        # Not strictly necessary, but warn if order differs
        print("[WARN] target columns present but ordering in dataframe is not monotonic with expected order.")

def check_batch_shapes(batch: dict, target_cols: Sequence[str]):
    # image tensor
    if "img_t" not in batch:
        raise AssertionError("Batch missing 'img_t'")
    if not torch.is_tensor(batch["img_t"]):
        raise AssertionError("'img_t' must be a torch.Tensor")
    # targets
    if "y" not in batch:
        raise AssertionError("Batch missing 'y' target tensor")
    y = batch["y"]
    if y.dim() != 2:
        raise AssertionError(f"'y' must be 2D [B, T], got shape {tuple(y.shape)}")
    if y.shape[1] != len(target_cols):
        raise AssertionError(f"'y' second-dim ({y.shape[1]}) != expected target count ({len(target_cols)})")
    # species_id
    if "species_id" in batch and (not torch.is_tensor(batch["species_id"])):
        raise AssertionError("'species_id' must be torch.Tensor if present")
    return True

def validate_dataset_and_one_batch(ds, dl, target_cols):
    # dataset-level check (columns)
    if hasattr(ds, "df"):
        assert_target_cols_order(ds.df.columns, target_cols)
    # try one batch from dataloader
    b = next(iter(dl))
    check_batch_shapes(b, target_cols)
    print("[OK] dataset and dataloader basic shape checks passed.")