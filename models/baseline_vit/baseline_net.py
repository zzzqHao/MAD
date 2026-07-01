import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .combine_block import CombineBlock
from .forward_block import ForwardBlockStack
from .orco import PseudoTargetClassifier
from .projector import Projector
from .prototype_classifier import PrototypeAwareClassifier
from .vit_loader import build_frozen_vit


class MABHead(nn.Module):
    def __init__(self, dim, num_heads=8, hidden_dim=2048, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        squeeze_output = x.ndim == 2
        if squeeze_output:
            x = x.unsqueeze(1)
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        x = self.out_norm(x)
        if squeeze_output:
            x = x.squeeze(1)
        return x


class BaselineViTProtoNet(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.args = args
        self.mode = mode

        self.encoder = build_frozen_vit(args)
        self.feature_extractor = getattr(args, "feature_extractor", "forward_combine")
        if self.feature_extractor == "vit":
            self.forward_blocks = None
            self.combine_block = None
        elif self.feature_extractor in ("forward_block", "forward_combine"):
            forward_layers = getattr(self.encoder, "forward_layers", [])
            num_forward_layers = len(forward_layers)
            frozen_blocks = [self.encoder.blocks[layer_idx] for layer_idx in forward_layers]
            self.forward_blocks = ForwardBlockStack(
                dim=args.encoder_outdim,
                num_heads=args.num_heads,
                num_layers=num_forward_layers,
                token_nums=getattr(args, "forward_token_nums", 1),
                hidden_dim=getattr(args, "forward_hidden_dim", 48),
                active_attn_hidden_dim=getattr(args, "forward_active_attn_hidden_dim", 8),
                attn_scale=getattr(args, "forward_attn_scale", 0.1),
                ffn_scale=getattr(args, "forward_ffn_scale", 0.1),
                drop=getattr(args, "forward_drop", 0.1),
                attn_drop=getattr(args, "forward_attn_drop", 0.1),
                frozen_blocks=frozen_blocks,
                prompt_context=True,
                prompt_context_hidden_dim=getattr(args, "incremental_prompt_context_hidden_dim", 48),
                prompt_context_scale=getattr(args, "incremental_prompt_context_scale", 0.1),
                prompt_mode=getattr(args, "prompt_mode", "prompt_incremental"),
                active_ablation=getattr(args, "active_ablation", "none"),
            )
            if self.feature_extractor == "forward_combine":
                self.combine_block = CombineBlock(
                    dim=args.encoder_outdim,
                    num_forward_layers=num_forward_layers,
                    hidden_dim=getattr(args, "combine_hidden_dim", 48),
                    drop=getattr(args, "combine_drop", 0.1),
                    combine_ablation=getattr(args, "combine_ablation", "none"),
                    fusion_method=getattr(args, "fusion_method", "conditional_weights"),
                )
            else:
                self.combine_block = None
        else:
            raise ValueError(f"Unsupported feature_extractor: {self.feature_extractor}")
        classifier = getattr(args, "classifier", "fagg_mab_project")
        if classifier not in ("fagg_mab_project", "orco_fagg_mab_project"):
            raise ValueError("Only classifier=fagg_mab_project or classifier=orco_fagg_mab_project is supported.")
        self.classifier = classifier
        self.mab_head = MABHead(
            dim=args.encoder_outdim,
            num_heads=getattr(args, "mab_num_heads", 8),
            hidden_dim=getattr(args, "mab_hidden_dim", 2048),
            drop=getattr(args, "mab_drop", 0.1),
            attn_drop=getattr(args, "mab_attn_drop", 0.1),
        )
        self.mab_res_scale = getattr(args, "mab_res_scale", 0.1)
        self.projector = Projector(args.encoder_outdim, args.proj_hidden_dim, args.proj_output_dim)
        if classifier == "orco_fagg_mab_project":
            self.fc = PseudoTargetClassifier(
                args.num_classes,
                args.proj_output_dim,
                reserve_mode=getattr(args, "orco_reserve_mode", "all"),
                temperature=getattr(args, "orco_temperature", 1.0),
            )
        else:
            self.fc = PrototypeAwareClassifier(
                args.num_classes,
                args.proj_output_dim,
                args.proto_temperature,
                getattr(args, "proto_context_weight", 0.5),
            )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def extract_forward_outputs(self, x, use_prompt_context=True):
        with torch.no_grad():
            encodings = self.encoder(x)

        layer_tokens = encodings.get("forward_layer_tokens") if isinstance(encodings, dict) else None
        if not layer_tokens:
            raise RuntimeError("Forward Block is enabled, but the encoder did not return forward_layer_tokens.")
        return self.forward_blocks(layer_tokens, use_prompt_context=use_prompt_context)

    def extract_features(self, x, use_prompt_context=True):
        with torch.no_grad():
            encodings = self.encoder(x)

        if self.feature_extractor == "vit":
            tokens = encodings["x"] if isinstance(encodings, dict) else encodings
            if tokens.ndim == 3:
                f_agg = tokens[:, 0]
            else:
                f_agg = tokens
        elif self.feature_extractor in ("forward_block", "forward_combine"):
            layer_tokens = encodings.get("forward_layer_tokens") if isinstance(encodings, dict) else None
            if not layer_tokens:
                raise RuntimeError("Forward Block is enabled, but the encoder did not return forward_layer_tokens.")
            forward_outputs = self.forward_blocks(layer_tokens, use_prompt_context=use_prompt_context)
            if self.feature_extractor == "forward_combine":
                f_agg = self.combine_block(forward_outputs)
            else:
                f_agg = forward_outputs["feature"]
        return {
            "f_agg": f_agg,
        }

    def encode(self, x):
        if self.feature_extractor == "vit":
            with torch.no_grad():
                encodings = self.encoder(x)
            tokens = encodings["x"] if isinstance(encodings, dict) else encodings
            if tokens.ndim != 3:
                raise RuntimeError(f"Expected frozen ViT token output with shape [B, N, C], got {tuple(tokens.shape)}.")
            mab_tokens = self.mab_head(tokens)
            z_n = self.projector(mab_tokens[:, 0])
            return z_n

        features = self.extract_features(x)
        f_agg = features["f_agg"]
        refined = self.mab_head(f_agg) + self.mab_res_scale * f_agg
        return self.projector(refined)

    @torch.no_grad()
    def update_incremental_prompt_context(self, loader, class_list):
        if self.feature_extractor == "vit" or self.forward_blocks is None:
            return None
        if getattr(self.args, "prompt_mode", "prompt_incremental") != "prompt_incremental":
            self.forward_blocks.set_prompt_context(None)
            return None

        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        class_set = set(int(class_id) for class_id in class_list)
        layer_sums = None
        sample_count = 0
        for images, labels in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            labels = labels.to(device)
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for class_id in class_set:
                mask |= labels == class_id
            if not mask.any():
                continue

            forward_outputs = self.extract_forward_outputs(images, use_prompt_context=False)
            h_list = forward_outputs["h"]
            if layer_sums is None:
                layer_sums = torch.zeros(len(h_list), self.args.encoder_outdim, device=device)
            for layer_idx, h in enumerate(h_list):
                layer_sums[layer_idx] += h[mask].sum(dim=(0, 1))
            sample_count += int(mask.sum().item()) * self.forward_blocks.token_nums

        if layer_sums is None or sample_count == 0:
            self.forward_blocks.set_prompt_context(None)
            if was_training:
                self.train()
            return None

        layer_context = layer_sums / sample_count
        self.forward_blocks.set_prompt_context(layer_context)

        if was_training:
            self.train()
        return layer_context

    def set_base_prompts_trainable(self, trainable=True):
        if self.forward_blocks is not None:
            self.forward_blocks.set_base_prompts_trainable(trainable)

    def forward_metric(self, x):
        features = self.encode(x)
        logits = self.fc(features)
        return logits, features

    def forward(self, input, **kwargs):
        if self.mode == "backbone":
            with torch.no_grad():
                return self.encoder(input)
        if self.mode == "encoder":
            return self.encode(input)
        return self.forward_metric(input)

    @torch.no_grad()
    def update_prototypes(self, loader, class_list):
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        features_by_class = {int(c): [] for c in class_list}
        sample_counts = {int(c): 0 for c in class_list}

        tqdm_gen = tqdm(loader, desc="Prototype update", dynamic_ncols=True, leave=False)
        for images, labels in tqdm_gen:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            labels = labels.to(device)

            features = self.encode(images)
            for class_id in features_by_class:
                mask = labels == class_id
                if mask.any():
                    features_by_class[class_id].append(features[mask].detach())
                    sample_counts[class_id] += int(mask.sum().item())
            tqdm_gen.set_postfix(updated=sum(count > 0 for count in sample_counts.values()), refresh=False)

        class_ids = []
        prototypes = []
        for class_id, class_features in features_by_class.items():
            if not class_features:
                continue
            stacked = torch.cat(class_features, dim=0)
            class_ids.append(class_id)
            prototypes.append(stacked.mean(dim=0))

        if prototypes:
            prototypes = torch.stack(prototypes, dim=0)
            if hasattr(self.fc, "assign_targets"):
                self.fc.assign_targets(np.array(class_ids), prototypes)
            else:
                self.fc.set_prototypes(np.array(class_ids), prototypes)

        if was_training:
            self.train()

        return {
            "updated_classes": class_ids,
            "sample_counts": {class_id: sample_counts[class_id] for class_id in class_ids},
        }

    def get_trainable_params(self):
        trainable = [(name, param.shape) for name, param in self.named_parameters() if param.requires_grad]
        for name, shape in trainable:
            print(f"Parameter: {name}, Shape: {shape}")
