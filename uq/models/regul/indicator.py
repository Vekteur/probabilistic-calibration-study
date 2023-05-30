import torch


def relaxed_indicator(a, b, s):
    """Indicator function that is relaxed so that it can be backpropagated through.
    indicator(a <= b) ~= relaxed_indicator(a, b)
    """
    return torch.sigmoid(s * (b - a))


def exact_forward_relaxed_indicator(a, b, s):
    """Indicator function that is exact but can be backpropagated through using a relaxation
    of this function.
    """
    return relaxed_indicator(a, b, s=s) - relaxed_indicator(a, b, s=s).detach() + (a < b).float()


def indicator(a, b, s=None, exact_forward=False):
    if s is None:
        return (a <= b).float()
    if exact_forward:
        return exact_forward_relaxed_indicator(a, b, s)
    return relaxed_indicator(a, b, s)
