# File for the various loss functions that different Ledidi Objectives use
import torch


def ratio_ballpark_loss(y_hat, y_bar=None, low=0.5, high=2.0):
    """Zero loss inside [low, high]; quadratic penalty outside."""
    below = torch.clamp(low - y_hat, min=0.0)
    above = torch.clamp(y_hat - high, min=0.0)
    return (below ** 2 + above ** 2).mean()


def ratio_inverted_ballpark_loss(y_hat, y_bar=None, low=0.5, high=2.0):
    """Quadratic penalty inside [low, high]; zero loss outside."""
    inside_low  = (y_hat - low).clamp(min=0.0)
    inside_high = (high - y_hat).clamp(min=0.0)
    inside_dist = torch.minimum(inside_low, inside_high)
    inside_dist *= 10
    inside_loss = inside_dist ** 2
    return inside_loss.mean()


def relative_asymmetry_loss(y_hat: torch.Tensor, y_initial: float):
    """
    Creates a loss that encourages y_hat to become more asymmetrical 
    relative to its starting value, y_initial.
    """
    if y_initial > 1.0:
        # If initially right-asymmetric (ratio > 1), push the ratio higher.
        return 20 * -y_hat.mean()
    else:
        # If initially left-asymmetric (ratio < 1), push the ratio lower.
        return 20 * y_hat.mean()


def stripe_ratio_loss(y_hat, accept_ratio):
    """
    Zero loss while |y_hat| ≤ thresh.
    Quadratic penalty once we leave that band.
    """
    loss = torch.clamp(y_hat - accept_ratio, min=0.0)
    return (5*loss)**2


def make_extruding_to_stable_loss(stripe_pattern, bx, by):

    biggest_stripe = max(bx, by)
    def loss_fn(y_hat: torch.Tensor, y_bar=None) -> torch.Tensor:
        y = y_hat.squeeze(0) if y_hat.dim() == 2 else y_hat  # (3,)
        corner_val, mean_x, mean_y = y[0], y[1], y[2]

        ratio_min = 1.6

        ratio_curr = corner_val / biggest_stripe

        gap = (ratio_min - ratio_curr) / 0.08    

        #inc_x = F.softplus((mean_x - bx) / 0.03)
        #inc_y = F.softplus((mean_y - by) / 0.03)

        return 4 * F.softplus(gap) #+ inc_y + inc_x #+ loss_keep_x + loss_keep_y
    return loss_fn

def stripe_diff_loss(y_hat, thresh: float = 10):
    """
    Zero loss while |y_hat| ≤ thresh.
    Quadratic penalty once we leave that band.
    """
    overshoot = torch.clamp(y_hat.abs() - thresh, min=0.0)
    return (overshoot ** 2).mean()
