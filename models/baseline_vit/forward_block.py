import torch
import torch.nn as nn

from .layers import BottleneckAdapter


class ActiveAttention(nn.Module):
    def __init__(self, dim, hidden_dim=8, drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = dim ** -0.5
        self.q_proj = BottleneckAdapter(dim, hidden_dim=hidden_dim, drop=0.)
        self.k_proj = BottleneckAdapter(dim, hidden_dim=hidden_dim, drop=0.)
        self.v_proj = BottleneckAdapter(dim, hidden_dim=hidden_dim, drop=0.)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
        self.q_norm = norm_layer(dim)
        self.kv_norm = norm_layer(dim)

    def forward(self, prompt):
        prompt_len = prompt.shape[1]
        norm_prompt = self.q_norm(prompt)
        norm_ref_feat = self.kv_norm(prompt)
        q = self.q_proj(norm_prompt)
        k = self.k_proj(norm_ref_feat)
        v = self.v_proj(norm_ref_feat)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = attn @ v
        x = self.proj_drop(x)
        return x[:, :prompt_len]


class ActiveBlock(nn.Module):
    def __init__(
        self, dim, num_heads, hidden_dim=48, attn_scale=0.1, ffn_scale=0.1,
        drop=0., attn_drop=0., active_attn_hidden_dim=8, norm_layer=nn.LayerNorm,
        active_ablation="none",
    ):
        super().__init__()
        valid_active_ablations = {"none", "no_proj", "no_self_att", "no_mlp"}
        if active_ablation not in valid_active_ablations:
            raise ValueError(f"Unsupported active_ablation: {active_ablation}")
        self.active_ablation = active_ablation
        self.norm1 = norm_layer(dim)
        self.attn = ActiveAttention(dim, hidden_dim=active_attn_hidden_dim, drop=attn_drop, norm_layer=norm_layer)
        self.norm2 = norm_layer(dim)
        self.mlp = BottleneckAdapter(dim, hidden_dim=hidden_dim, drop=drop)
        self.attn_scale = nn.Parameter(torch.ones(1) * attn_scale)
        self.ffn_scale = nn.Parameter(torch.ones(1) * ffn_scale)

    def forward(self, x):
        if self.active_ablation != "no_self_att":
            x = x + self.attn(self.norm1(x)) * self.attn_scale
        if self.active_ablation != "no_mlp":
            x = x + self.mlp(self.norm2(x)) * self.ffn_scale
        return x


class ForwardBlock(nn.Module):
    def __init__(
        self, dim, num_heads, hidden_dim=48, attn_scale=0.1, ffn_scale=0.1,
        drop=0., attn_drop=0., active_attn_hidden_dim=8, norm_layer=nn.LayerNorm,
        active_ablation="none",
    ):
        super().__init__()
        valid_active_ablations = {"none", "no_proj", "no_self_att", "no_mlp"}
        if active_ablation not in valid_active_ablations:
            raise ValueError(f"Unsupported active_ablation: {active_ablation}")
        self.active_ablation = active_ablation
        self.input_proj = BottleneckAdapter(dim, hidden_dim=hidden_dim, drop=drop)
        self.active = ActiveBlock(
            dim, num_heads, hidden_dim=hidden_dim, attn_scale=attn_scale,
            ffn_scale=ffn_scale, drop=drop, attn_drop=attn_drop,
            active_attn_hidden_dim=active_attn_hidden_dim, norm_layer=norm_layer,
            active_ablation=active_ablation,
        )

    @staticmethod
    def frozen_attention(frozen_attn, query, key_value):
        bsz, query_tokens, channels = query.shape
        kv_tokens = key_value.shape[1]
        qkv_query = frozen_attn.qkv(query).reshape(
            bsz, query_tokens, 3, frozen_attn.num_heads, channels // frozen_attn.num_heads
        ).permute(2, 0, 3, 1, 4)
        qkv_kv = frozen_attn.qkv(key_value).reshape(
            bsz, kv_tokens, 3, frozen_attn.num_heads, channels // frozen_attn.num_heads
        ).permute(2, 0, 3, 1, 4)
        q = qkv_query[0]
        k, v = qkv_kv[1], qkv_kv[2]

        attn = (q @ k.transpose(-2, -1)) * frozen_attn.scale
        attn = frozen_attn.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(bsz, query_tokens, channels)
        x = frozen_attn.proj(x)
        return frozen_attn.proj_drop(x)

    def forward(self, prev_query, prompt, layer_tokens, frozen_block):
        active_query = prev_query + prompt
        if self.active_ablation != "no_proj":
            active_query = self.input_proj(active_query)
        active_query = self.active(active_query)
        f_att_delta = self.frozen_attention(
            frozen_block.attn,
            frozen_block.norm1(active_query),
            frozen_block.norm1(layer_tokens),
        )
        f_att = active_query + frozen_block.drop_path1(frozen_block.ls1(f_att_delta))
        f_mlp = frozen_block.drop_path2(frozen_block.ls2(frozen_block.mlp(frozen_block.norm2(f_att))))
        h = f_att + f_mlp
        return h, f_att, f_mlp


class ForwardBlockStack(nn.Module):
    def __init__(
        self, dim, num_heads, num_layers, token_nums=1, hidden_dim=48,
        active_attn_hidden_dim=8, attn_scale=0.1, ffn_scale=0.1,
        drop=0., attn_drop=0., frozen_blocks=None, prompt_context=True,
        prompt_context_hidden_dim=48, prompt_context_scale=0.1,
        prompt_mode="prompt_incremental",
        active_ablation="none",
    ):
        super().__init__()
        valid_active_ablations = {"none", "no_proj", "no_self_att", "no_mlp"}
        if active_ablation not in valid_active_ablations:
            raise ValueError(f"Unsupported active_ablation: {active_ablation}")
        valid_prompt_modes = {"no_prompt", "prompt_only", "prompt_incremental"}
        if prompt_mode not in valid_prompt_modes:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        self.dim = dim
        self.token_nums = token_nums
        self.frozen_blocks = list(frozen_blocks) if frozen_blocks is not None else None
        self.num_layers = num_layers
        self.prompt_mode = prompt_mode
        self.active_ablation = active_ablation
        self.prompts = nn.ParameterList([nn.Parameter(torch.empty(token_nums, dim)) for _ in range(num_layers)])
        for prompt in self.prompts:
            nn.init.uniform_(prompt, -1, 1)
        self.prompt_context = prompt_context
        if self.prompt_context:
            self.context_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
            self.context_scale = nn.Parameter(torch.ones(1) * prompt_context_scale)
            self.context_generators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, prompt_context_hidden_dim),
                    nn.GELU(),
                    nn.Linear(prompt_context_hidden_dim, token_nums * dim),
                )
                for _ in range(num_layers)
            ])
            for generator in self.context_generators:
                nn.init.zeros_(generator[-1].weight)
                nn.init.zeros_(generator[-1].bias)
            self.register_buffer("current_layer_prompt_context", torch.zeros(num_layers, dim))
            self.has_prompt_context = False
        else:
            self.context_norms = None
            self.context_scale = None
            self.context_generators = None
            self.register_buffer("current_layer_prompt_context", torch.zeros(num_layers, dim))
            self.has_prompt_context = False
        self.blocks = nn.ModuleList([
            ForwardBlock(
                dim, num_heads, hidden_dim=hidden_dim, attn_scale=attn_scale,
                ffn_scale=ffn_scale, drop=drop, attn_drop=attn_drop,
                active_attn_hidden_dim=active_attn_hidden_dim,
                active_ablation=active_ablation,
            )
            for _ in range(num_layers)
        ])

    def set_prompt_context(self, layer_context):
        if not self.prompt_context:
            return
        if layer_context is None:
            self.current_layer_prompt_context.zero_()
            self.has_prompt_context = False
            return
        if layer_context.shape != (self.num_layers, self.dim):
            raise RuntimeError(
                f"Expected layer prompt context shape {(self.num_layers, self.dim)}, "
                f"got {tuple(layer_context.shape)}."
            )
        self.current_layer_prompt_context.copy_(
            layer_context.detach().to(self.current_layer_prompt_context.device)
        )
        self.has_prompt_context = True

    def set_base_prompts_trainable(self, trainable=True):
        for prompt in self.prompts:
            prompt.requires_grad_(trainable)

    def has_active_prompt_context(self):
        return self.has_prompt_context or bool(torch.count_nonzero(self.current_layer_prompt_context).item())

    def get_prompt(self, layer_idx, prompt, use_prompt_context=True):
        if self.prompt_mode == "no_prompt":
            return torch.zeros_like(prompt)
        if self.prompt_mode == "prompt_only":
            return prompt
        if not (self.prompt_context and self.has_active_prompt_context() and use_prompt_context):
            return prompt
        context = self.context_norms[layer_idx](self.current_layer_prompt_context[layer_idx])
        delta = self.context_generators[layer_idx](context).view(self.token_nums, self.dim)
        return prompt + self.context_scale * delta

    def forward(self, layer_tokens, use_prompt_context=True):
        if self.frozen_blocks is None:
            raise RuntimeError("ForwardBlockStack requires frozen ViT blocks for EfficientFSL-style Frozen Blocks.")
        if len(layer_tokens) != len(self.blocks):
            raise RuntimeError(f"Expected {len(self.blocks)} layer token sets, got {len(layer_tokens)}.")
        if len(self.frozen_blocks) != len(self.blocks):
            raise RuntimeError(f"Expected {len(self.blocks)} frozen blocks, got {len(self.frozen_blocks)}.")
        batch_size = layer_tokens[0].shape[0]
        h = layer_tokens[0].new_zeros(batch_size, self.token_nums, layer_tokens[0].shape[-1])
        h_list = []
        f_att_list = []
        f_mlp_list = []
        for layer_idx, (prompt, block, tokens, frozen_block) in enumerate(
            zip(self.prompts, self.blocks, layer_tokens, self.frozen_blocks)
        ):
            prompt = self.get_prompt(layer_idx, prompt, use_prompt_context=use_prompt_context)
            prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
            h, f_att, f_mlp = block(h, prompt, tokens, frozen_block)
            h_list.append(h)
            f_att_list.append(f_att)
            f_mlp_list.append(f_mlp)
        return {
            "feature": h.mean(dim=1),
            "h": h_list,
            "f_att": f_att_list,
            "f_mlp": f_mlp_list,
        }
