from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import torch
from torch import nn

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

@dataclass
class PruningConfig:
    threshold: float
    min_remaining: int = 1
    verbose: bool = True

@torch.no_grad()
def greedy_pruning(
    model: nn.Module,
    X_orig: torch.Tensor,          # (1, 4, L)
    X_edited: torch.Tensor,        # (1, 4, L)
    score_fn: ScoreFn,             # compares full vs candidate outputs
    cfg: PruningConfig,
) -> torch.Tensor:
    """
    Generic greedy pruning:
      - model(X_orig / X_edited) -> y
      - score_fn(y_full, y_candidate) -> scalar effect
    """
    model.eval()
    X_hat = X_edited.clone()

    # 1. Find all edited positions
    diff_mask = (X_orig != X_hat).sum(dim=1) > 0       # (1, L) -> (L,)
    diff_idxs = {int(i) for i in torch.nonzero(diff_mask, as_tuple=False).flatten()}
    n_total = len(diff_idxs)
    if n_total == 0:
        return X_hat

    # 2. Compute baseline output once
    y_full = model(X_hat)

    remaining = n_total
    step = 0
    while remaining > cfg.min_remaining and diff_idxs:
        step += 1
        best_score = float("inf")
        best_idx = None

        # 3. Try reverting each edit
        for idx in diff_idxs:
            X_mod = X_hat.clone()
            X_mod[0, :, idx] = X_orig[0, :, idx]
            y_mod = model(X_mod)

            score = float(score_fn(y_full, y_mod))
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None or best_score >= cfg.threshold:
            # No edit is cheap enough to remove
            break

        diff_idxs.remove(best_idx)
        X_hat[0, :, best_idx] = X_orig[0, :, best_idx]
        remaining = len(diff_idxs)

        if cfg.verbose:
            print(
                f"[PRUNE] step={step} pruned_idx={best_idx} "
                f"best_score={best_score:.4g} remaining={remaining}/{n_total}"
            )

    return X_hat
