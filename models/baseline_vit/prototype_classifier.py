import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim, temperature=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("seen_mask", torch.zeros(num_classes, dtype=torch.bool))

    def reset(self):
        self.prototypes.zero_()
        self.seen_mask.zero_()

    def set_prototypes(self, class_ids, prototypes):
        class_ids = torch.as_tensor(class_ids, device=self.prototypes.device, dtype=torch.long)
        prototypes = prototypes.to(self.prototypes.device)
        self.prototypes[class_ids] = F.normalize(prototypes, p=2, dim=-1)
        self.seen_mask[class_ids] = True

    def get_logits(self, features):
        features = F.normalize(features, p=2, dim=-1)
        prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        return F.linear(features, prototypes) / self.temperature

    def forward(self, features):
        return self.get_logits(features)


class PrototypeAwareClassifier(PrototypeClassifier):
    def __init__(self, num_classes, feat_dim, temperature=1.0, context_weight=0.5):
        super().__init__(num_classes, feat_dim, temperature)
        self.context_weight = context_weight

    def get_context(self, features, prototypes):
        similarity = F.linear(features, prototypes)
        if self.seen_mask.any():
            similarity = similarity.masked_fill(~self.seen_mask.unsqueeze(0), -1e9)
        weights = torch.softmax(similarity, dim=-1)
        return F.linear(weights, prototypes.t())

    def get_logits(self, features):
        features = F.normalize(features, p=2, dim=-1)
        prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        context = self.get_context(features, prototypes)
        features = F.normalize(features + self.context_weight * context, p=2, dim=-1)
        return F.linear(features, prototypes) / self.temperature
