from dataset.dataset_utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import models as models

from opacus import PrivacyEngine


### LOSS BASED ATTACKS ###
def compute_losses(dataloader, model, criterion, device):
    """
    Compute per-sample losses, logits, and labels for a given dataloader and model.

    """
    model.eval()
    losses, logits, labels = [], [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Store results
            losses.append(loss.cpu().numpy())
            logits.append(outputs.cpu())
            labels.append(targets.cpu())

    return {
        "loss": np.concatenate(losses),
        "logit": torch.cat(logits),
        "labels": torch.cat(labels),
    }


def compute_losses_for_indexes(dataset, indexes, model, criterion, device):
    """
    Evaluate per-sample losses, logits, and labels for specific dataset indexes.
    """

    dataloader = DataLoader(
        DatasetSplit(dataset, indexes),
        batch_size=200,
        shuffle=False,
        num_workers=0,
    )

    return compute_losses(dataloader, model, criterion, device)


### GRADIENT BASE ATTACKS ###
def compute_all_gradient_metrics(
    cos_model,
    initial_loader,
    test_dataloader,
    test_set,
    train_set,
    train_idxs,
    val_idxs,
    model_grads,
    lr,
    device,
    use_dp,
):

    optimizer = optim.SGD(cos_model.parameters(), lr, momentum=0.9, weight_decay=0.0005)

    # Differential privacy mitigation
    privacy_engine = PrivacyEngine()
    cos_model, optimizer, samples_loader = privacy_engine.make_private(
        module=cos_model,
        optimizer=optimizer,
        data_loader=initial_loader,
        noise_multiplier=3 if use_dp else 0,
        max_grad_norm=50 if use_dp else 1e10,
    )

    tarin_dataloader = DataLoader(
        DatasetSplit(train_set, train_idxs), batch_size=10, shuffle=False, num_workers=0
    )
    val_dataloader = DataLoader(
        DatasetSplit(train_set, val_idxs), batch_size=10, shuffle=False, num_workers=0
    )
    test_dataloader = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=0)

    train_cos, train_diffs, train_norm = compute_gradient_metrics(
        tarin_dataloader, optimizer, cos_model, device, model_grads
    )
    val_cos, val_diffs, val_norm = compute_gradient_metrics(
        val_dataloader, optimizer, cos_model, device, model_grads
    )
    test_cos, test_diffs, test_norm = compute_gradient_metrics(
        test_dataloader, optimizer, cos_model, device, model_grads
    )

    return (
        train_cos,
        train_diffs,
        train_norm,
        val_cos,
        val_diffs,
        val_norm,
        test_cos,
        test_diffs,
        test_norm,
    )


def compute_gradient_metrics(samples_ldr, optimizer, cos_model, device, model_grads):
    """
    Compute cosine similarity, gradient differences, and gradient norms for all samples in a DataLoader.

    Args:
        samples_ldr (DataLoader): DataLoader for the dataset to compute metrics on.
        cos_model (torch.nn.Module): Model used to calculate gradients.
        model_grads (torch.Tensor): Reference gradients of the target model.
    """
    model_grads = model_grads.to(device)
    cos_model.train()

    # Precompute the squared norm of the model gradients
    model_diff_norm = torch.norm(model_grads, p=2) ** 2

    # Metrics storage
    cos_scores, grad_diffs, grad_norms = [], [], []

    for inputs, targets in samples_ldr:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward and backward passes
        outputs = cos_model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # Extract gradients for all samples in the batch
        batch_grads = extract_sample_gradients(cos_model)

        # Compute metrics for each sample
        for sample_grad in batch_grads:
            cos_scores.append(F.cosine_similarity(sample_grad, model_grads, dim=0))
            grad_diffs.append(
                model_diff_norm - torch.norm(model_grads - sample_grad, p=2) ** 2
            )
            grad_norms.append(torch.norm(sample_grad, p=2) ** 2)

    return (
        torch.tensor(cos_scores).cpu(),
        torch.tensor(grad_diffs).cpu(),
        torch.tensor(grad_norms).cpu(),
    )


def extract_sample_gradients(model):
    """
    Extract per-sample gradients for all parameters in a model.

    """
    sample_grads = []

    for param in model.parameters():
        if param.requires_grad:
            # Flatten gradients for each sample and concatenate
            # The i-th dimension is the grad of the parameter of the i-th sample
            sample_grads.append(param.grad_sample.flatten(start_dim=1))

    return torch.cat(sample_grads, dim=1)
