# File for the various loss functions that different Ledidi Objectives use
import torch
import torch.nn.functional as F

def ratio_inverted_ballpark_loss(y_hat, y_bar=None, low=0.5, high=2.0):
    """Quadratic penalty inside [low, high]; zero loss outside."""
    inside_low  = (y_hat - low).clamp(min=0.0)
    inside_high = (high - y_hat).clamp(min=0.0)
    inside_dist = torch.minimum(inside_low, inside_high)
    inside_dist *= 10
    inside_loss = inside_dist ** 2
    return inside_loss.mean()


def make_extruding_to_stable_loss(bx, by, ratio_min):

    biggest_stripe = max(bx, by)
    def loss_fn(y_hat: torch.Tensor, y_bar=None) -> torch.Tensor:
        y = y_hat.squeeze(0) if y_hat.dim() == 2 else y_hat  # (3,)
        corner_val, mean_x, mean_y = y[0], y[1], y[2]

        ratio_curr = corner_val / biggest_stripe

        gap = (ratio_min - ratio_curr) / 0.08   

        return 4 * F.softplus(gap)
    return loss_fn

def make_stable_to_extruding_loss(used_ratio, bx, by):
    biggest_stripe = max(bx, by)
    def loss_fn(y_hat: torch.Tensor, y_bar=None) -> torch.Tensor:
        y = y_hat.squeeze(0) if y_hat.dim() == 2 else y_hat  # (3,)
        corner_val, mean_x, mean_y = y[0], y[1], y[2]

        ratio_curr = corner_val / biggest_stripe

        """over_desired = torch.clamp(ratio_curr - used_ratio, min=0.0)

        return 20 * over_desired"""
        
        gap = (ratio_curr - used_ratio) / 0.08

        return 4 * F.softplus(gap)
    return loss_fn

def stripe_diff_loss(y_hat, thresh: float = 10):
    """
    Zero loss while |y_hat| ≤ thresh.
    Quadratic penalty once we leave that band.
    """
    overshoot = torch.clamp(y_hat.abs() - thresh, min=0.0) * 10
    return (overshoot ** 2).mean()
