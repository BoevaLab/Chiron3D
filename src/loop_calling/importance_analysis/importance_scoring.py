import torch
from captum.attr import DeepLift, IntegratedGradients, InputXGradient, Saliency
from torch import nn


class BaseImportanceScorer:
    def __init__(self, device):
        self.device = device
    def compute_scores(self, model, element, stripe, ignore_k):
        """Compute importance scores given a model and a dataset element."""
        raise NotImplementedError("Must implement compute_scores in subclass")


def calculate_input_x_gradient(model, element, stripe, compare_stripes, device, ignore_k=0):
    """
    Calculate input x gradient for a model's output with respect to the input.

    Args:
        model (torch.nn.Module): The model to compute gradients for.
        element (dict): Input element containing 'sequence' and optionally 'features'.
        stripe (str): 'X' or 'Y'.
        compare_stripes (bool): If True, sum of stripes are compared
        device (torch.device): Device to run computations on.
        ignore_k (int): Ignore the first k values (closest to the diagonal) of a stripe

    Returns:
        torch.Tensor: Input x gradient tensor.
    """
    # Prepare input
    test_input = element["sequence"].clone()
    if "features" in element:
        test_input = torch.cat((test_input, element["features"].permute(1, 0).clone()), dim=1)

    test_input = test_input.unsqueeze(0)  # Add batch dimension
    test_input = test_input.permute(0, 2, 1).to(device)
    test_input.requires_grad_()  # Enable gradient computation for input

    # Forward pass
    output = (torch.exp(model(test_input).squeeze(0)) - 1)

    sum_x = output[element["relative_loop_start"],
            element["relative_loop_start"] + ignore_k:element["relative_loop_end"]].sum()
    sum_y = output[element["relative_loop_start"]:element["relative_loop_end"] - ignore_k,
            element["relative_loop_end"] - 1].sum()
    if stripe == "X":
        if compare_stripes:
            if sum_y <= 0:
                return None
            (sum_x / sum_y).backward()
        else:
            sum_x.backward()
    else:
        if compare_stripes:
            if sum_x <= 0:
                return None
            (sum_y / sum_x).backward()
        else:
            sum_y.backward()

            # Compute input x gradient
    input_x_gradient = test_input.grad.detach().cpu() * test_input.detach().cpu()

    return input_x_gradient.squeeze().permute(1, 0)

class GradientScorer(BaseImportanceScorer):
    def __init__(self, device):
        super().__init__(device)

    def compute_scores(self, model, element, stripe, ignore_k):
        # Replace this with your actual gradient computation logic
        input_ = element["sequence"].clone()
        if "features" in element:
            input_ = torch.cat((input_, element["features"]), dim=0)

        input_ = input_.unsqueeze(0)  # Add batch dimension
        input_.requires_grad_()

        def forward_fn(x):
            # x: [1, C, L]
            output = model(x).squeeze(0)  # [L, L]

            i = element["relative_loop_start"]
            j = element["relative_loop_end"]

            sum_x = output[i, i + ignore_k: j].sum()
            sum_y = output[i: j - ignore_k, j - 1].sum()

            print(stripe)
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


# --- DeepLIFT Scoring Implementation --- Doesnt work for now as Corigami uses MultiHeadAttention which is not supported
# and results in high convergence delta (difference between f(x) - f(baseline) and sum of attributions (should be 0)

class CustomForward(nn.Module):
    def __init__(self, model, element, stripe, ignore_k):
        super().__init__()
        self.model = model
        self.element = element
        self.stripe = stripe
        self.k = ignore_k

    def forward(self, input_tensor):
        output = (torch.exp(self.model(input_tensor)) - 1)

        i = self.element["relative_loop_start"]
        j = self.element["relative_loop_end"]

        sum_x = output[:, i, i + self.k:j].sum(dim=1)

        sum_y = output[:, i:j - self.k, j - 1].sum(dim=1)

        if self.stripe == "X":
            ratio = torch.where(sum_y > 0, sum_x / sum_y, torch.tensor(0.0, device=input_tensor.device))
            return ratio
        else:
            ratio = torch.where(sum_x > 0, sum_y / sum_x, torch.tensor(0.0, device=input_tensor.device))
            return ratio


class DeepLiftScorer(BaseImportanceScorer):
    def __init__(self, device):
        super().__init__(device)

    def compute_scores(self, model, element, stripe, ignore_k):
        input_ = element["sequence"].clone()
        if "features" in element:
            input_ = torch.cat((input_, element["features"].clone()), dim=0)

        input_ = input_.unsqueeze(0)
        input_ = input_.permute(0, 2, 1).to(self.device)
        baseline = torch.zeros_like(input_)
        forward_pass = CustomForward(model, element, stripe, ignore_k)
        deeplift = DeepLift(forward_pass)
        attributions, delta = deeplift.attribute(input_, baselines=baseline, return_convergence_delta=True)        
        return attributions.squeeze().t(), delta
        # return in format (feature, position) with feature either acgtn or ctcf - also return convergence deltas


class IntegratedGradientsScorer(BaseImportanceScorer):
    def __init__(self, device):
        super().__init__(device)

    def compute_scores(self, model, element, stripe, ignore_k):
        # prepare your input & baseline as before…
        input_ = element["sequence"].clone()
        if "features" in element:
            input_ = torch.cat((input_, element["features"].clone()), dim=0)

        input_ = input_.unsqueeze(0).permute(0, 2, 1).to(self.device)  # [1,C,L]
        baseline = torch.zeros_like(input_)

        def forward_fn(x):
            output = torch.exp(model(x).squeeze(0)) - 1  # [L, L]
            i, j = element["relative_loop_start"], element["relative_loop_end"]
            sum_x = output[i, i + ignore_k : j].sum()
            sum_y = output[i : j - ignore_k, j - 1].sum()
            if sum_x == 0:
                sum_x += 1e-6
            if sum_y == 0:
                sum_y += 1e-6
            return (sum_x / sum_y).view(1) if stripe == "X" else (sum_y / sum_x).view(1)

        ig = IntegratedGradients(forward_fn)

        attributions, delta = ig.attribute(
            input_,
            baselines=baseline,
            n_steps=20,
            method="gausslegendre",
            internal_batch_size=1,
            return_convergence_delta=True
        )

        return attributions.squeeze().t(), delta
