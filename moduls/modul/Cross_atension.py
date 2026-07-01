import math
from typing import Optional, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F

# from Share_Moe import ShareMOE
from .Share_Moe import ShareMOE


# Define type alias for KV Cache
KVCache = Tuple[torch.Tensor, torch.Tensor]

# --- Helper Dependencies ---
ACT2FN = {
    "silu": nn.SiLU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
}

# --- Helper Functions and RoPE/Norm Modules (Remaining unchanged) ---

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped-Query Attention helper function: repeats K/V heads to match the number of Q heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """RoPE helper function: rotates half of the last dimension of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """RoPE core application function."""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    # ... (code omitted) ...
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

class RMSNorm(nn.Module):
    # ... (code omitted) ...
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

# [Modification 1]: Removed GatedInputEmbedding class

class Attention(nn.Module):
    # ... (code omitted) ...
    def __init__(
        self, 
        hidden_size: int = 128, 
        num_attention_heads: int = 8, 
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)
        self.dropout = attention_dropout

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KVCache] = None, 
        use_cache: bool = False,                  
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = q_len + (past_key_value[0].shape[2] if past_key_value is not None else 0)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


# ---------------------------------------------
# --- Standard FFN Module (Alternative for non-MoE mode) ---
# ---------------------------------------------
class MLP(nn.Module):
    
    def __init__(self, 
        hidden_size: int = 128, 
        # intermediate_size: int = 512, # Original setting for intermediate layer dimension
        intermediate_size: int = 2,   # Changed to a multiplier of the input size
        hidden_act: str = "silu"
    ):
        super().__init__()

        # intermediate = hidden_size * 2
        intermediate = hidden_size * intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        # [Modified Comment]: MLP module returns its output and None (as a placeholder for aux_loss)
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)), None


# ---------------------------------------------
# --- Core Module (DecoderLayer) ---
# ---------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_size: int = 128, 
        num_attention_heads: int = 8, 
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 512,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        # FFN/MoE toggle parameters
        use_moe: bool = False, 
        intermediate_size: int = 2, 
        hidden_act: str = "silu",
        # ShareMOE specific parameters
        expert_hidden_mult: int = 2,
        top_k: int = 2,
        expert_number: int = 4,
        shared_experts_number: int = 2,
    ):
        super().__init__()
        
        self.use_moe = use_moe # Store toggle state
        
        self.self_attn = Attention(
            hidden_size, num_attention_heads, num_key_value_heads,
            max_position_embeddings, rope_theta, attention_dropout
        )
        
        if self.use_moe:
            # Use ShareMOE
            self.mlp_or_moe = ShareMOE(
                hidden_dim=hidden_size, 
                expert_hidden_mult=expert_hidden_mult,
                top_k=top_k,
                expert_number=expert_number,
                shared_experts_number=shared_experts_number
            )
        else:
            # Use standard MLP
            self.mlp_or_moe = MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act
            )
            
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[torch.Tensor]]:
        
        # 1. Attention Sub-layer
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            normed_hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + attn_output 
        
        # 2. FFN/MoE Sub-layer
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Unified call, receives output and aux_loss (MLP returns None)
        mlp_output, aux_loss = self.mlp_or_moe(normed_hidden_states) 
        
        hidden_states = residual + mlp_output
        
        # Return hidden_states, present_key_value, and aux_loss
        return hidden_states, present_key_value, aux_loss

class MultiLayerDecoder(nn.Module):
    # [Modification 2]: Removed input_dim, removed GatedInputEmbedding
    def __init__(
        self, 
        # Removed input_dim: int = 1,
        hidden_size: int = 128, # Input feature dimension must now match hidden_size
        num_hidden_layers: int = 4, # Number of hidden layers
        num_attention_heads: int = 8,  # Number of attention heads
        num_key_value_heads: int = 4,  # GQA, number of KV heads
        max_position_embeddings: int = 512,  # Maximum positional embeddings
        rms_norm_eps: float = 1e-6,  # RMSNorm epsilon
        rope_theta: float = 10000.0,  # RoPE theta
        attention_dropout: float = 0.0,  # Attention dropout
        # Toggle and FFN parameters
        use_moe: bool = False,           # Whether to use MoE
        intermediate_size: int = 2,      # FFN intermediate size as multiplier of input
        hidden_act: str = "silu",        # FFN intermediate activation function
        # ShareMOE specific parameters
        expert_hidden_mult: int = 2,     # Expert hidden layer multiplier
        top_k: int = 2,                  # Number of experts selected
        expert_number: int = 4,          # Number of experts
        shared_experts_number: int = 2,  # Number of shared experts   
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        # [Modification 2]: Removed input_embedding, assuming input feature dim is hidden_size
        # self.input_embedding = GatedInputEmbedding(input_dim, hidden_size, hidden_act)
        
        layer_params = {
            "hidden_size": hidden_size, 
            "num_attention_heads": num_attention_heads, 
            "num_key_value_heads": num_key_value_heads,
            "max_position_embeddings": max_position_embeddings,
            "rms_norm_eps": rms_norm_eps,
            "rope_theta": rope_theta,
            "attention_dropout": attention_dropout,
            # Pass toggle parameters
            "use_moe": use_moe, 
            "intermediate_size": intermediate_size, 
            "hidden_act": hidden_act,
            # ShareMOE specific parameters
            "expert_hidden_mult": expert_hidden_mult,
            "top_k": top_k,
            "expert_number": expert_number,
            "shared_experts_number": shared_experts_number,
        }
        
        self.layers = nn.ModuleList([DecoderLayer(**layer_params) for _ in range(num_hidden_layers)])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self, 
        input_values: torch.Tensor, # Shape should be (batch_size, seq_length, hidden_size)
        attention_mask: Optional[torch.Tensor] = None, 
        past_key_values: Optional[List[KVCache]] = None, 
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCache]], List[torch.Tensor]]:

        batch_size, seq_length, input_dim = input_values.shape
        device = input_values.device

        # print('Device used by model:', device)
        
        # [Modified Comment]: Check if input dimension matches
        if input_dim != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_dim}) must match MultiLayerDecoder hidden_size ({self.hidden_size}), "
                "because GatedInputEmbedding was removed and no dimension mapping is performed."
            )

        if attention_mask is not None: 
            attention_mask = attention_mask.to(device)
        else: # Can be empty
            # Causal encoding
            # print('Causal encoding')
            causal_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
            # attention_mask = (
            #     torch.zeros_like(causal_mask, dtype=torch.float)
            #     .masked_fill_(causal_mask, -torch.finfo(torch.float).max)
            # )
            attention_mask = (
                torch.zeros_like(causal_mask, dtype=torch.float)
                .masked_fill_(causal_mask, float('-inf'))
            )
            # print('attention_mask:', attention_mask)
           

            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_attention_heads, -1, -1)
            attention_mask = attention_mask.to(device)

          
        
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2] 
        
        # [Modification 2]: Directly use input as hidden states, no embedding step
        hidden_states = input_values
        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device).unsqueeze(0)
        
        next_key_values = [] if use_cache else None
        all_aux_losses = [] 

        # [Modified Comment]: Loop through all decoder layers
        for layer_idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            
            # [Modified Comment]: Call DecoderLayer, returning hidden_states, present_key_value, aux_loss
            hidden_states, present_key_value, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                next_key_values.append(present_key_value)
            
            # Collect only in MoE mode and when aux_loss exists (MLP mode returns None)
            if layer.use_moe and aux_loss is not None:
                all_aux_losses.append(aux_loss)
        
        hidden_states = self.norm(hidden_states)
        
        # Return final hidden_states, KV cache, and list of aux_losses
        return hidden_states, next_key_values, all_aux_losses


# --- Verification Execution ---
if __name__ == '__main__':
    
    # Example 1: ShareMOE mode (use_moe=True)
    # Note: input_dim removed, hidden_size must match the input data feature dimension
    HIDDEN_SIZE = 64
    decoder_moe = MultiLayerDecoder(
        num_hidden_layers=2, hidden_size=HIDDEN_SIZE, # hidden dim = 64
        use_moe=True, expert_number=4, shared_experts_number=2
    )
    
    # Example 2: FFN mode (use_moe=False)
    decoder_ffn = MultiLayerDecoder(
        num_hidden_layers=2, hidden_size=HIDDEN_SIZE, # hidden dim = 64
        use_moe=False, intermediate_size=4 
    )
    
    print("--- Mode Switching Verification ---")
    print(f"MOE Decoder FFN Type: {type(decoder_moe.layers[0].mlp_or_moe).__name__}")
    print(f"FFN Decoder FFN Type: {type(decoder_ffn.layers[0].mlp_or_moe).__name__}")
    
    print("\n--- MOE Mode Run Demo (with Loss) ---")
    seq_len = 5
    # Input feature dimension must now be HIDDEN_SIZE (64)
    dummy_input = torch.randn(2, seq_len, HIDDEN_SIZE)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_mask = torch.zeros_like(causal_mask, dtype=torch.float).masked_fill_(causal_mask, -torch.finfo(torch.float).max)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(2, decoder_moe.num_attention_heads, -1, -1)

    output_moe, _, aux_losses_moe = decoder_moe(dummy_input, attention_mask=attention_mask, use_cache=False)
    # output_moe, _, aux_losses_moe = decoder_moe(dummy_input, use_cache=False) # Handled internally

    
    print(f"MOE Output Size: {output_moe.shape}")
    print(f"Number of auxiliary losses collected in MOE mode: {len(aux_losses_moe)}")
    print(f"MOE mode Layer 0 loss value: {aux_losses_moe[0].item():.4f}")

    print("\n--- FFN Mode Run Demo (no Loss) ---")
    output_ffn, _, aux_losses_ffn = decoder_ffn(dummy_input, attention_mask=attention_mask, use_cache=False)
    
    print(f"FFN Output Size: {output_ffn.shape}")
    print(f"Number of auxiliary losses collected in FFN mode: {len(aux_losses_ffn)}")
    
    print("\nFFN/ShareMOE switching logic successfully implemented, and GatedInputEmbedding removed. ✅")

    print('decoder_ffn parameters:', decoder_ffn)
    output_moe, _, aux_losses_moe = decoder_moe(dummy_input, use_cache=False) # Handled internally