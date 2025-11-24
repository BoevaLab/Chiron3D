from torch import nn
import torch

class RatioWrapper(nn.Module):
    """Light wrapper that outputs the X-/Y-stripe ratio for one loop."""
    def __init__(
        self,
        core_model: nn.Module,
        i: int,
        j: int,
        stripe: str = "X",
        ignore_k: int = 15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.core = core_model.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.i, self.j, self.k = i, j, ignore_k
        self.stripe = stripe
        self.eps = eps

    def forward(self, x):
        i, j, k = self.i, self.j, self.k
        #m = torch.exp(self.core(x)) - 1  # undo log1p
        m = self.core(x)
        if m.ndim == 2:
            m = m.unsqueeze(0)
        sum_x = m[:, i, i + k : j].sum(dim=-1)
        sum_y = m[:, i : j - k, j - 1].sum(dim=-1)
        if self.stripe == "X":
            ratio = sum_x / (sum_y + self.eps)
        else:  # "Y"
            ratio = sum_y / (sum_x + self.eps)
        return ratio.unsqueeze(-1)
    
class CornerWrapper(nn.Module):
    """Light wrapper that outputs the X-/Y-stripe ratio for one loop."""
    def __init__(
        self,
        core_model: nn.Module,
        i: int,
        j: int,
        stripe: str = "X",
        ignore_k: int = 15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.core = core_model.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.i, self.j, self.k = i, j, ignore_k
        self.stripe = stripe
        self.eps = eps

    def forward(self, x):
        i, j, k = self.i, self.j, self.k
        #m = torch.exp(self.core(x)) - 1  # undo log1p
        m = self.core(x)
        m = m.squeeze()
        stripe_x = m[i, i + k : j-3]
        stripe_y = m[i+3 : j - k, j - 1]

        mean_x = stripe_x.mean()
        mean_y = stripe_y.mean()

        corner_val = m[j-1, i]

        corner_vals = torch.stack([
            m[i, j-1], m[i, j-2], m[i, j-3],    # horizontal (corner + 2 left)
            m[i+1, j-1], m[i+2, j-1]            # vertical (2 below)
        ])
        corner_region = corner_vals.mean()
        ratio = corner_region / max(mean_x, mean_y)
        return ratio.unsqueeze(-1)


class StripeWrapper(nn.Module):
    """Light wrapper that outputs the X-/Y-stripe ratio for one loop."""
    def __init__(
        self,
        core_model: nn.Module,
        i: int,
        j: int,
        stripe: str = "X",
        ignore_k: int = 15,
        eps: float = 1e-6,
        base_sum_x=None, 
        base_sum_y=None
    ):
        super().__init__()
        self.core = core_model.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.i, self.j, self.k = i, j, ignore_k
        self.stripe = stripe
        self.eps = eps
        self.base_sum_x = base_sum_x
        self.base_sum_y = base_sum_y


    def forward(self, x):
        i, j, k = self.i, self.j, self.k
        m = self.core(x)
        sum_x = m[:, i, i + k : j].sum(dim=-1)
        sum_y = m[:, i : j - k, j - 1].sum(dim=-1)
        if self.stripe == "X":
            diff = self.base_sum_x - sum_y
        else:  # stripe == "Y"
            diff = self.base_sum_y - sum_x
        return diff.unsqueeze(-1)

class HiChIPStripeAndCorner(nn.Module):
    """Light wrapper that outputs the X-/Y-stripe ratio for one loop."""

    def __init__(
        self,
        core_model: nn.Module,
        i: int,
        j: int,
        ignore_k: int = 15,
    ):
        super().__init__()
        self.core = core_model.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.i, self.j, self.k = i, j, ignore_k

    def forward(self, x):
        i, j, k = self.i, self.j, self.k
        m =self.core(x)  
        m = m.squeeze()
        stripe_x = m[i, i + k : j]
        stripe_y = m[i : j - k, j - 1]

        mean_x = stripe_x.mean()
        mean_y = stripe_y.mean()

        corner_val = m[j-1, i]

        y = torch.stack([corner_val, mean_x, mean_y], dim=0)

        return y.unsqueeze(0)
    

def scalar_from_wrapper(wrapper: nn.Module, seq_1hot: torch.Tensor) -> float:
    """Return scalar output for a single sequence (4,L)"""
    with torch.no_grad():
        return wrapper(seq_1hot.unsqueeze(0)).item()
