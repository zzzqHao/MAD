import os

import torch
from timm.models import create_model

import models.vision_transformer


def candidate_vit_paths(args):
    paths = []
    if getattr(args, "vit_pretrained_path", None):
        paths.append(args.vit_pretrained_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    workspace_root = os.path.abspath(os.path.join(project_root, ".."))
    paths.extend(
        [
            os.path.join(workspace_root, "pretrained", "ViT-B-16.pt"),
            os.path.join(project_root, "pretrained", "ViT-B-16.pt"),
        ]
    )
    return paths


def parse_layer_indices(value, depth):
    if value == "all":
        return list(range(depth))
    if isinstance(value, str):
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        layers = [int(item) for item in value]
    if not layers:
        raise ValueError("forward_layers must contain at least one layer index.")
    if len(layers) != len(set(layers)):
        raise ValueError(f"forward_layers must be unique, got: {layers}")
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= depth]
    if invalid_layers:
        raise ValueError(f"forward_layers {invalid_layers} are outside the valid range [0, {depth - 1}].")
    return layers


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        if "TorchScript archive" not in str(exc) and "zip file" not in str(exc):
            raise
        return torch.jit.load(path, map_location="cpu")


def _extract_state_dict(checkpoint):
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "visual"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def _convert_clip_visual_state_dict(state_dict):
    converted = {}

    direct_map = {
        "visual.conv1.weight": "patch_embed.proj.weight",
        "visual.ln_post.weight": "norm.weight",
        "visual.ln_post.bias": "norm.bias",
    }

    for src, dst in direct_map.items():
        if src in state_dict:
            converted[dst] = state_dict[src].float()

    if "visual.class_embedding" in state_dict:
        converted["cls_token"] = state_dict["visual.class_embedding"].float().reshape(1, 1, -1)
    if "visual.positional_embedding" in state_dict:
        converted["pos_embed"] = state_dict["visual.positional_embedding"].float().unsqueeze(0)

    for i in range(12):
        prefix = f"visual.transformer.resblocks.{i}"
        block = f"blocks.{i}"
        block_map = {
            f"{prefix}.attn.in_proj_weight": f"{block}.attn.qkv.weight",
            f"{prefix}.attn.in_proj_bias": f"{block}.attn.qkv.bias",
            f"{prefix}.attn.out_proj.weight": f"{block}.attn.proj.weight",
            f"{prefix}.attn.out_proj.bias": f"{block}.attn.proj.bias",
            f"{prefix}.ln_1.weight": f"{block}.norm1.weight",
            f"{prefix}.ln_1.bias": f"{block}.norm1.bias",
            f"{prefix}.ln_2.weight": f"{block}.norm2.weight",
            f"{prefix}.ln_2.bias": f"{block}.norm2.bias",
            f"{prefix}.mlp.c_fc.weight": f"{block}.mlp.fc1.weight",
            f"{prefix}.mlp.c_fc.bias": f"{block}.mlp.fc1.bias",
            f"{prefix}.mlp.c_proj.weight": f"{block}.mlp.fc2.weight",
            f"{prefix}.mlp.c_proj.bias": f"{block}.mlp.fc2.bias",
        }
        for src, dst in block_map.items():
            if src in state_dict:
                converted[dst] = state_dict[src].float()

    return converted


def load_local_vit_weights(model, path):
    checkpoint = _load_checkpoint(path)
    state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()

    if any(key.startswith("visual.") for key in state_dict):
        state_dict = _convert_clip_visual_state_dict(state_dict)
    else:
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded local ViT weights from: {path}")
    print(f"Loaded keys: {len(filtered)}, skipped keys: {len(skipped)}, missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def build_frozen_vit(args):
    if getattr(args, "vit_pretrained_path", None) and not os.path.exists(args.vit_pretrained_path):
        raise FileNotFoundError(f"vit_pretrained_path does not exist: {args.vit_pretrained_path}")
    local_vit_path = next((path for path in candidate_vit_paths(args) if os.path.exists(path)), None)
    encoder = create_model(args.model, False if local_vit_path else args.pretrained)
    if local_vit_path:
        load_local_vit_weights(encoder, local_vit_path)
    for param in encoder.parameters():
        param.requires_grad = False
    feature_extractor = getattr(args, "feature_extractor", "forward_combine")
    encoder.return_forward_layer_tokens = feature_extractor in ("forward_block", "forward_combine")
    encoder.forward_layers = parse_layer_indices(getattr(args, "forward_layers", "all"), encoder.depth)
    encoder.eval()
    return encoder
