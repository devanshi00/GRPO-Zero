import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# LlamaConfig
# ---------------------------
@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    hidden_act: str = "silu"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    attention_bias: bool = False

# ---------------------------
# LlamaRMSNorm
# ---------------------------
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# ---------------------------
# Rotary Embedding
# ---------------------------
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [batch, seq, ...], position_ids: [batch, seq]
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seqlen, head_dim)

# ---------------------------
# Attention
# ---------------------------
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        bsz, seqlen, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.o_proj(attn_output)

# ---------------------------
# FeedForward (MLP)
# ---------------------------
class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = F.silu if config.hidden_act == "silu" else getattr(F, config.hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# ---------------------------
# Transformer Block
# ---------------------------
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

# ---------------------------
# LlamaModel
# ---------------------------
class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config, device=device)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        bsz, seqlen = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states

# ---------------------------
# LlamaForCausalLM
# ---------------------------
class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.model = LlamaModel(config, device=device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        config_file = Path(ckpt_path) / "config.json"
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config = LlamaConfig(**config_dict)

        with torch.device("meta"):
            model = cls(config, device=device)

        import safetensors.torch

        model_weight_files = sorted(Path(ckpt_path).glob("model*.safetensors"))
        weights = {}
        for file in model_weight_files:
            weights.update(safetensors.torch.load_file(file, device="cpu"))
        # remove "model." prefix from keys if present
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True)
        return model.to(device)

# # ---------------------------
# # Example usage
# # ---------------------------
# if __name__ == "__main__":
#     config = LlamaConfig()
#     model = LlamaForCausalLM(config)
#     input_ids = torch.randint(0, config.vocab_size, (2, 16))
#     output = model(input_ids)
#     print(output["logits"].shape)  # (batch, seq_len, vocab_size)
