import torch
import torch.nn as nn

from .layers import BottleneckAdapter


class CombineBlock(nn.Module):
    def __init__(
        self, dim, num_forward_layers, hidden_dim=48, drop=0.1,
        combine_ablation="none", fusion_method="conditional_weights",
    ):
        super().__init__()
        valid_combine_ablations = {"none", "no_f_att", "no_f_mlp", "no_h"}
        if combine_ablation not in valid_combine_ablations:
            raise ValueError(f"Unsupported combine_ablation: {combine_ablation}")
        valid_fusion_methods = {"simple_average", "fixed_weights", "conditional_weights"}
        if fusion_method not in valid_fusion_methods:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")
        self.num_forward_layers = num_forward_layers
        self.combine_ablation = combine_ablation
        self.fusion_method = fusion_method
        self.num_candidates = self.get_num_candidates(num_forward_layers, combine_ablation)
        self.combine_mlp = BottleneckAdapter(dim, hidden_dim=hidden_dim, drop=drop)
        self.weight_mlp = nn.Sequential(
            nn.Linear(dim, self.num_candidates),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.fixed_logits = nn.Parameter(torch.zeros(self.num_candidates))

    @staticmethod
    def get_num_candidates(num_forward_layers, combine_ablation):
        if combine_ablation == "none":
            return 3 * num_forward_layers
        if combine_ablation == "no_f_att":
            return 2 * num_forward_layers
        if combine_ablation == "no_f_mlp":
            return 2 * num_forward_layers
        if combine_ablation == "no_h":
            return 2 * num_forward_layers
        raise ValueError(f"Unsupported combine_ablation: {combine_ablation}")

    @staticmethod
    def pool_tokens(features):
        return [feature.mean(dim=1) for feature in features]

    def forward(self, forward_outputs):
        h_features = self.pool_tokens(forward_outputs["h"])
        f_att_features = self.pool_tokens(forward_outputs["f_att"])
        f_mlp_features = self.pool_tokens(forward_outputs["f_mlp"])

        candidates = []
        condition = h_features[-1]
        if self.combine_ablation != "no_h":
            candidates.extend(h_features)
        if self.combine_ablation != "no_f_att":
            candidates.extend(f_att_features)
        if self.combine_ablation != "no_f_mlp":
            candidates.extend(f_mlp_features)
        if not candidates:
            raise RuntimeError("CombineBlock requires at least one candidate feature.")

        condition = self.combine_mlp(condition)
        features = torch.stack([self.combine_mlp(feature) for feature in candidates], dim=1)

        batch_size = features.shape[0]
        if self.fusion_method == "simple_average":
            return features.mean(dim=1)
        if self.fusion_method == "fixed_weights":
            weights = torch.softmax(self.fixed_logits, dim=0).view(1, self.num_candidates, 1)
            return torch.sum(weights * features, dim=1)

        weights = torch.softmax(self.weight_mlp(condition), dim=1).view(batch_size, self.num_candidates, 1)
        return torch.sum(weights * features, dim=1)
