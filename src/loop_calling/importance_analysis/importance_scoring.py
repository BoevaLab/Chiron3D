import torch
from captum.attr import DeepLift, IntegratedGradients, InputXGradient, Saliency
from torch import nn


class BaseImportanceScorer:
    def __init__(self, device):
        self.device = device
    def compute_scores(self, model, element, stripe, ignore_k):
        """Compute importance scores given a model and a dataset element."""
        raise NotImplementedError("Must implement compute_scores in subclass")

class GradientScorer(BaseImportanceScorer):
    def __init__(self, device):
        super().__init__(device)

    def compute_scores(self, model, element, stripe, ignore_k):
        # Replace this with your actual gradient computation logic
        input_ = element["sequence"].unsqueeze(0).to(self.device)
        if "features" in element:
            input_ = torch.cat((input_, element["features"]), dim=0)

        input_.requires_grad_()

        def forward_fn(x):
            # x: [1, C, L]
            output = model(x).squeeze(0)  # [L, L]

            i = element["relative_loop_start"]
            j = element["relative_loop_end"]

            sum_x = output[i, i + ignore_k: j].sum()
            sum_y = output[i: j - ignore_k, j - 1].sum()

            if stripe == "X":
                if sum_y <= 0:
                    raise ValueError("sum_y <= 0, skipping")
                return (sum_x / sum_y).view(1)
            elif stripe == "STABLE":
                corner_peak = output[i, j-1]
                mean_x = output[i, i + ignore_k: j].mean()
                mean_y = output[i: j - ignore_k, j - 1].mean()
                total = corner_peak - mean_x - mean_y
                return total.view(1) 
            else:
                if sum_x <= 0:
                    raise ValueError("sum_x <= 0, skipping")
                return (sum_y / sum_x).view(1)

        ig = InputXGradient(forward_fn)
        try:
            attributions = ig.attribute(input_)
        except ValueError:
            return None, None
        return attributions.squeeze().t(), None
