import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def normalize(x, eps=1e-12):
    return x / torch.norm(x, dim=1, p=2, keepdim=True).clamp_min(eps)


def perturb_targets_norm_count(targets, target_labels, ncount, nviews, epsilon=1.0, offset=0.0):
    views = []
    ix = torch.randperm(targets.shape[0], device=targets.device)
    if ix.shape[0] < ncount:
        rep_count = math.ceil(ncount / ix.shape[0])
        ix = ix.repeat(rep_count)[:ncount]
        ix = ix[torch.randperm(ix.shape[0], device=targets.device)]
    else:
        ix = ix[:ncount]

    for _ in range(nviews):
        rand = ((torch.rand(ncount, targets.shape[1], device=targets.device) - offset) * epsilon)
        views.append(normalize(targets[ix] + rand))

    return views, target_labels[ix]


def simplex_loss(feat, labels, assigned_targets, assigned_targets_label, unassigned_targets):
    if feat.numel() == 0:
        return feat.new_tensor(0.0)

    unique_labels = torch.unique(labels)
    averaged = feat.new_zeros(len(unique_labels), feat.shape[1])
    for i, label in enumerate(unique_labels):
        averaged[i] = feat[labels == label].mean(dim=0)
    averaged = normalize(averaged)

    if assigned_targets.numel() > 0:
        assigned_targets_label = assigned_targets_label.to(labels.device)
        mask = ~torch.isin(assigned_targets_label, unique_labels)
        assigned_targets_not_in_batch = assigned_targets[mask]
    else:
        assigned_targets_not_in_batch = feat.new_zeros(0, feat.shape[1])

    pieces = [averaged, assigned_targets_not_in_batch]
    if unassigned_targets.numel() > 0:
        pieces.append(unassigned_targets)
    all_targets = normalize(torch.cat(pieces, dim=0))

    sim = F.cosine_similarity(all_targets[None, :, :], all_targets[:, None, :], dim=-1)
    return torch.log(torch.exp(sim).sum(dim=1)).sum() / all_targets.shape[0]


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, margin=0.0):
        if len(features.shape) < 3:
            raise ValueError("features must have shape [batch_size, n_views, feature_dim].")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        device = features.device
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask.")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features.")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        logits = logits.clone()
        logits[mask > 0] -= margin

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))

        mask_sum = mask.sum(1)
        valid = mask_sum > 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / mask_sum[valid]
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class PseudoTargetClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim, reserve_mode="all", temperature=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.reserve_mode = reserve_mode
        self.temperature = temperature
        reserve_count = num_classes if reserve_mode == "all" else feat_dim
        self.register_buffer("classifiers", torch.zeros(num_classes, feat_dim))
        self.register_buffer("seen_mask", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("rv", normalize(torch.randn(reserve_count, feat_dim)))
        self.register_buffer("rv_available", torch.ones(reserve_count, dtype=torch.bool))

    def find_reserve_vectors_all(self, epochs=1000, lr=1.0):
        points = nn.Parameter(normalize(self.rv.detach().clone()))
        opt = torch.optim.SGD([points], lr=lr)
        for _ in range(epochs):
            sim = F.cosine_similarity(points[None, :, :], points[:, None, :], dim=-1)
            loss = torch.log(torch.exp(sim / self.temperature).sum(dim=1)).sum() / points.shape[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            points.data = normalize(points.data)
        self.rv.copy_(points.detach())

    @torch.no_grad()
    def assign_targets(self, class_ids, prototypes):
        class_ids = torch.as_tensor(class_ids, device=self.classifiers.device, dtype=torch.long)
        prototypes = normalize(prototypes.to(self.classifiers.device))
        new_mask = ~self.seen_mask[class_ids]
        class_ids = class_ids[new_mask]
        prototypes = prototypes[new_mask]
        if class_ids.numel() == 0:
            return []

        available_idx = torch.nonzero(self.rv_available, as_tuple=False).flatten()
        if available_idx.numel() < class_ids.numel():
            raise RuntimeError(
                f"Not enough OrCo reserve vectors: need {class_ids.numel()}, have {available_idx.numel()}."
            )
        available_rv = self.rv[available_idx]
        cost = torch.matmul(prototypes, normalize(available_rv).T).detach().cpu().numpy()
        _, col_ind = linear_sum_assignment(cost, maximize=True)
        col_ind = torch.as_tensor(col_ind, device=self.classifiers.device, dtype=torch.long)
        selected_idx = available_idx[col_ind]
        assigned = self.rv[selected_idx]
        self.classifiers[class_ids] = assigned
        self.seen_mask[class_ids] = True
        self.rv_available[selected_idx] = False
        return class_ids.detach().cpu().tolist()

    def get_classifier_weights(self):
        return self.classifiers[self.seen_mask]

    def get_classifier_labels(self):
        return torch.nonzero(self.seen_mask, as_tuple=False).flatten()

    def get_unassigned_targets(self):
        return self.rv[self.rv_available]

    def get_logits(self, features):
        features = normalize(features)
        classifiers = normalize(self.classifiers)
        return F.linear(features, classifiers) / self.temperature

    def forward(self, features):
        return self.get_logits(features)
