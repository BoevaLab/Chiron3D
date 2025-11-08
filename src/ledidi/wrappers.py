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
        m = torch.exp(self.core(x)) - 1  # undo log1p
        sum_x = m[:, i, i + k : j].sum(dim=-1)
        sum_y = m[:, i : j - k, j - 1].sum(dim=-1)
        if self.stripe == "X":
            ratio = sum_x / (sum_y + self.eps)
        else:  # "Y"
            ratio = sum_y / (sum_x + self.eps)
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



def scalar_from_wrapper(wrapper: nn.Module, seq_1hot: torch.Tensor) -> float:
    """Return scalar output for a single sequence (4,L)"""
    with torch.no_grad():
        return wrapper(seq_1hot.unsqueeze(0)).item()
