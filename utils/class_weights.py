def compute_class_weights(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Inverse-frequency class weights to penalise errors on rare classes more.

    weight_c = N / (num_classes × count_c)

    Returns a (num_classes,) float tensor suitable for nn.CrossEntropyLoss(weight=...).
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1)          # avoid division by zero
    N      = labels.numel()
    weights = N / (num_classes * counts)
    return weights
