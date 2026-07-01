import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import math
import warnings  # Import the warnings module for printing warning messages

# warnings.warn() only outputs once by default
warnings.simplefilter("always", UserWarning)  # Set to always show to ensure visibility

# Ensure mamba-ssm library is correctly installed
# from mamba_ssm import Mamba2 
from mamba_ssm.modules.mamba2_macos import Mamba2MacOS as Mamba2  

# from Share_Moe import ShareMOE
from .Share_Moe import ShareMOE

# --- Helper Dependencies (Remain Unchanged) ---
ACT2FN = {
    "silu": nn.SiLU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
}

# --- RoPE Helper Functions (Remain Unchanged) ---

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped-Query Attention helper function: Repeats K/V heads to match the number of Q heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """RoPE helper function: Rotates half of the last dimension of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Core application function for Rotary Positional Embeddings (RoPE)."""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    """Module implementing Rotary Positional Embedding (RoPE)."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

# --- RMSNorm (Remain Unchanged) ---
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
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

# --- Downsampler Module (Remain Unchanged) ---
class Downsampler(nn.Module):
    """Uses 1D convolution to downsample the sequence, halving the sequence length."""
    def __init__(self, hidden_size: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # B L D -> B D L
        x = hidden_states.permute(0, 2, 1)
        x = self.conv(x)
        # B D L -> B L D
        x = x.permute(0, 2, 1)
        return x
        

class MLP(nn.Module):
    """Standard FFN module (GeLU gated)."""
    def __init__(self, 
                 hidden_size: int = 128, 
                 # intermediate_size: int = 512, # Original setting for intermediate layer dim
                 intermediate_size: int = 2, # Modified to be a multiplier
                 hidden_act: str = "silu"):
        super().__init__()
     
        # intermediate = hidden_size * multiplier
        intermediate = hidden_size * intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        output = self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
        return output, None # FFN does not return aux_loss

# --- MambaMoEDecoderLayer (Remain Unchanged) ---
class MambaMoEDecoderLayer(nn.Module):
    """Serial structure: Mamba Block (SSM) + Optional Downsampling + FFN/MoE Block."""
    def __init__(
        self, 
        hidden_size: int = 128, rms_norm_eps: float = 1e-6,
        d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 32, 
        use_moe: bool = False, intermediate_size: int = 2, hidden_act: str = "silu",
        expert_hidden_mult: int = 2, top_k: int = 2, expert_number: int = 4, shared_experts_number: int = 2,
        use_downsampling: bool = False,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_downsampling = use_downsampling
        
        self.mamba_block = Mamba2(d_model=hidden_size, d_state=d_state, d_conv=d_conv, 
                                  expand=expand, headdim=headdim, device=None)
        
        if self.use_downsampling:
            # Downsampling layer
            self.downsampler = Downsampler(hidden_size)
        else:
            self.downsampler = None
        
        # Choice between FFN or MoE
        self.mlp_or_moe = ShareMOE(
            hidden_dim=hidden_size, expert_hidden_mult=expert_hidden_mult, top_k=top_k,
            expert_number=expert_number, shared_experts_number=shared_experts_number
        ) if self.use_moe else MLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act
        )
            
        self.post_mamba_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # 1. Mamba Block
        mamba_output = self.mamba_block(hidden_states)
        hidden_states = mamba_output 

        # 2. Optional sequence downsampling
        if self.use_downsampling and self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            
        # 3. RMSNorm + FFN/MoE
        residual = hidden_states
        normed_hidden_states = self.post_mamba_layernorm(hidden_states)
        
        mlp_output, aux_loss = self.mlp_or_moe(normed_hidden_states)
        
        # 4. Residual Connection
        hidden_states = residual + mlp_output 
        
        return hidden_states, aux_loss

# --- Cross-Attention Gate Module ---


class SingleCrossAttentionNonRoPE(nn.Module):
    """Standard Single-head/Multi-head Cross Attention (without RoPE)."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]
        
        normed_query = self.norm(query) 
        Q = self.q_proj(normed_query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # $Q K^T$
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Restore shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        output = query + self.out_proj(attn_output) # Residual connection
        
        return output

class SingleCrossAttentionRoPE(nn.Module):
    """Cross Attention with Rotary Positional Embedding (RoPE)."""
    def __init__(
        self, 
        hidden_size: int = 128, 
        num_heads: int = 8, 
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE is only applied to K/V (for Cross-Attention)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)
        self.norm = RMSNorm(hidden_size)
        self.dropout = attention_dropout

    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor, 
    ) -> torch.Tensor:
        
        B, L_q, _ = query.shape
        L_kv = key_value.shape[1]
        
        normed_query = self.norm(query) 
        
        query_states = self.q_proj(normed_query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(key_value).view(B, L_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(key_value).view(B, L_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Compute RoPE
        position_ids = torch.arange(0, L_kv, dtype=torch.int64, device=key_value.device).unsqueeze(0).expand(B, L_kv)
        cos, sin = self.rotary_emb(key_value, seq_len=L_kv)
        # Apply RoPE only to Key
        k_embed = apply_rotary_pos_emb(key_states, key_states, cos, sin, position_ids)[1]
        
        # GQA/MQA repetition
        key_states = repeat_kv(k_embed, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, L_q, self.hidden_size)
        output = query + self.o_proj(attn_output) # Residual connection
        
        return output

class MultiLayerCrossAttentionGate(nn.Module):
    """
    Multi-layer Cross-Attention Gate module.
    Used to integrate outputs from stacked MambaMoE layers into a fixed-length representation 
    (Gate Tokens) or a variable-length representation (Last Layer).
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 4, 
        gate_seq_len: int = 1, 
        num_gate_layers: int = 1, 
        query_source: str = 'token', # 'token', 'last_layer', or 'first_layer'
        gate_type: str = 'standard', # 'standard' or 'rope'
        use_gate: bool = True,       # [New] Control whether to enable the Gate
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gate_seq_len = gate_seq_len
        self.query_source = query_source
        self.gate_type = gate_type
        self.use_gate = use_gate # [New] Save flag
        
        if not self.use_gate:
            # If Gate is disabled, no need to initialize tokens or layers
            self.gate_tokens = None
            self.gate_layers = None
            warnings.warn("MultiLayerCrossAttentionGate is configured with use_gate=False. Forward Pass will return the last layer output directly.", UserWarning)
            return

        if self.query_source == 'token':
            # Gate Tokens: Acting as Query for Cross-Attention, dimensions [1, gate_seq_len, hidden_size]
            self.gate_tokens = nn.Parameter(torch.randn(1, gate_seq_len, hidden_size))
        else:
            self.gate_tokens = None
        
        if gate_type == 'rope':
            GateLayer = SingleCrossAttentionRoPE
        elif gate_type == 'standard':
            GateLayer = SingleCrossAttentionNonRoPE
        else:
            raise ValueError(f"Unsupported Gate type: {gate_type}")
            
        layer_args = {"hidden_size": hidden_size, "num_heads": num_heads}
        
        if gate_type == 'rope':
            layer_args.update({
                "num_key_value_heads": kwargs.get("num_key_value_heads", num_heads // 2),
                "max_position_embeddings": kwargs.get("max_position_embeddings", 2048),
                "rope_theta": kwargs.get("rope_theta", 10000.0),
                "attention_dropout": kwargs.get("attention_dropout", 0.0)
            })

        self.gate_layers = nn.ModuleList([
            GateLayer(**layer_args) for _ in range(num_gate_layers)
        ])
        
    def forward(self, stacked_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            stacked_outputs (List[torch.Tensor]): List containing outputs of all MambaMoE layers.
                                                 Outputs may have different shapes due to downsampling.
        Returns:
            torch.Tensor: Final Gate output (fused representation).
        """
        if not stacked_outputs:
            raise ValueError("Stacked outputs list cannot be empty.")
        
        if not self.use_gate:
            # If Gate is disabled, return last layer output
            return stacked_outputs[-1]
            
        B, _, D = stacked_outputs[0].shape # Get Batch and Dim from first tensor
        
        # Concatenate outputs of all layers along the sequence dimension as K and V for Attention
        key_value = torch.cat(stacked_outputs, dim=1) 
        
        # Determine Query
        if self.query_source == 'token':
            # Use learnable Gate Tokens as Query; output length is gate_seq_len
            query = self.gate_tokens.expand(B, -1, -1) 
        elif self.query_source == 'last_layer':
            # Use last layer output as Query; output length matches last layer
            query = stacked_outputs[-1] 
        elif self.query_source == 'first_layer':
            # Use first layer output as Query; output length matches first layer
            query = stacked_outputs[0]
        else:
            raise ValueError(f"Invalid query_source setting: {self.query_source}. Must be 'token' or 'last_layer'.")
        
        # Pass through multi-layer Cross-Attention sequentially
        for layer in self.gate_layers:
            query = layer(query, key_value)
        
        return query

# ---------------------------------------------
# --- MambaMoEStackGateModel ---
# ---------------------------------------------

class MambaMoEStackGateModel(nn.Module):
    """
    Mamba MoE Stack Gate Model:
    1. MambaMoE Module: Contains Mamba Block, optional Downsampling, and FFN/MoE Block.
    2. Cross-Attention Gate: Integrates outputs of all MambaMoE layers.
    3. Features:
        - First layer non-downsampling: Forces the first layer to skip downsampling when use_downsampling=True.
        - Dynamic layer limit: Automatically truncates executable layers based on input sequence length.
        - Controllable Gate: Toggle Gate module with use_gate.
    """
    def __init__(
        self, 
        hidden_size: int = 128, num_hidden_layers: int = 4, rms_norm_eps: float = 1e-6,
        d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 32, 
        use_moe: bool = False, intermediate_size: int = 2, hidden_act: str = "silu",
        expert_hidden_mult: int = 2, top_k: int = 2, expert_number: int = 4, shared_experts_number: int = 2,
        use_downsampling: bool = False,
        # Gate Parameters
        use_gate: bool = True,           # Whether to enable Gate module
        num_heads: int = 4, 
        gate_seq_len: int = 1,           # Valid only if query_source='token'
        num_gate_layers: int = 1,        # Layer count in Gate module
        query_source: str = 'token',     # Gate Query source: 'token' or 'last_layer'
        gate_type: str = 'standard',     # Gate Attention type: 'standard' or 'rope'
        num_key_value_heads: int = 4,    # GQA parameter for RoPE Gate
        max_position_embeddings: int = 2048, # RoPE parameter for RoPE Gate
        rope_theta: float = 10000.0,     # RoPE parameter for RoPE Gate
    ):
        super().__init__()
        
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.query_source = query_source
        self.gate_type = gate_type
        self.use_downsampling = use_downsampling
        self.use_gate = use_gate         
        
        # 1. Mamba MoE Layers (Logic handles first layer non-downsampling)
        layer_params = {
            "hidden_size": hidden_size, "rms_norm_eps": rms_norm_eps,
            "d_state": d_state, "d_conv": d_conv, "expand": expand, "headdim": headdim, 
            "use_moe": use_moe, "intermediate_size": intermediate_size, "hidden_act": hidden_act,
            "expert_hidden_mult": expert_hidden_mult, "top_k": top_k, 
            "expert_number": expert_number, "shared_experts_number": shared_experts_number,
        }
        
        self.layers = nn.ModuleList()
        if self.use_downsampling and self.num_hidden_layers > 0:
            # a. Add the first layer, forced to skip downsampling
            first_layer_params = layer_params.copy()
            first_layer_params["use_downsampling"] = False
            self.layers.append(MambaMoEDecoderLayer(**first_layer_params))
            
            # b. Add the remaining layers with downsampling enabled
            downsampling_layer_params = layer_params.copy()
            downsampling_layer_params["use_downsampling"] = True
            for _ in range(self.num_hidden_layers - 1):
                self.layers.append(MambaMoEDecoderLayer(**downsampling_layer_params))
        else:
            # All layers without downsampling
            no_downsampling_params = layer_params.copy()
            no_downsampling_params["use_downsampling"] = False
            for _ in range(self.num_hidden_layers):
                self.layers.append(MambaMoEDecoderLayer(**no_downsampling_params))

        # 2. Cross-Attention Gate
        gate_kwargs = {
            "num_heads": num_heads, "num_key_value_heads": num_key_value_heads,
            "max_position_embeddings": max_position_embeddings, "rope_theta": rope_theta,
        }
        self.cross_attention_gate = MultiLayerCrossAttentionGate(
            hidden_size, 
            gate_seq_len=gate_seq_len, num_gate_layers=num_gate_layers, 
            query_source=query_source, gate_type=gate_type, 
            use_gate=use_gate,
            **gate_kwargs
        )
        
        # 3. Final Norm (Normalizes Gate output if enabled, otherwise normalizes last layer output)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self, 
        input_values: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        _, seq_length, input_dim = input_values.shape
        
        if input_dim != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_dim}) must match MambaMoEStackGateModel hidden_size ({self.hidden_size})."
            )
        
        # Dynamic layer count limitation logic
        layers_to_run = self.layers
        print(f"Sampling: {self.use_downsampling}, Seq Length: {seq_length}, Requested Layers: {self.num_hidden_layers}")
        if self.use_downsampling:
            # Calculate how many downsampling steps are possible starting from the second layer
            max_downsampling_layers = math.floor(math.log2(seq_length)) if seq_length > 1 else 0
            # Total executable layers = 1 non-downsampling layer + N downsampling layers
            max_executable_layers = 1 + max_downsampling_layers

            print(f"Seq Length: {seq_length}, Max Executable Layers: {max_executable_layers}")
            
            if self.num_hidden_layers > max_executable_layers:
                warnings.warn(
                    f"Model requested {self.num_hidden_layers} layers, but for sequence length {seq_length}, "
                    f"only {max_executable_layers} layers can be executed (1 non-downsampling + {max_downsampling_layers} downsampling). "
                    f"Truncating to {max_executable_layers} layers.",
                    UserWarning
                )
               
                # Only take the first max_executable_layers from created layers
                layers_to_run = self.layers[:max_executable_layers]

        hidden_states = input_values 
        stacked_outputs = [] 
        all_aux_losses = [] 

        # 1. Sequential pass through Mamba MoE Layers and collect outputs
        for layer in layers_to_run:
            hidden_states, aux_loss = layer(hidden_states) 
            stacked_outputs.append(hidden_states)
            
            if layer.use_moe and aux_loss is not None:
                all_aux_losses.append(aux_loss)
        
        # 2. Gate integrates all layer outputs or uses last layer output directly
        gated_output = self.cross_attention_gate(stacked_outputs)
        
        # 3. Final normalization
        final_states = self.norm(gated_output)
        
        # Return final representation and auxiliary losses
        return final_states, all_aux_losses

# --- Verification Run ---
if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    batch, length, HIDDEN_DIM = 2, 64, 32
    INPUT_DIM = HIDDEN_DIM
    gate_len = 5 
    
    x = torch.randn(batch, length, INPUT_DIM).to(device)

    # Verification 1: Standard Config (No downsampling, Gate enabled)
    print("\n--- Verification 1: Standard Config (No Downsampling, Gate Tokens Enabled) ---")
    num_layers_no_ds = 4
    model_standard = MambaMoEStackGateModel(
        hidden_size=HIDDEN_DIM, num_hidden_layers=num_layers_no_ds,
        use_moe=False,
        gate_seq_len=gate_len,
        query_source='token', 
        use_downsampling=False,
        use_gate=True,
    ).to(device)
    y_standard, _ = model_standard(x)
    # Expected output length is gate_seq_len
    print(f"Actual final output shape: {y_standard.shape}")
    assert y_standard.shape == (batch, gate_len, HIDDEN_DIM)
    print("✅ Standard Gate configuration forward pass successful!")

    # Verification 2: Gate Disabled (Returns last layer output)
    print("\n--- Verification 2: Gate Disabled (Returns Last Layer Output) ---")
    model_no_gate = MambaMoEStackGateModel(
        hidden_size=HIDDEN_DIM, num_hidden_layers=num_layers_no_ds,
        use_moe=False,
        query_source='token',
        use_downsampling=False,
        use_gate=False, 
    ).to(device)
    y_no_gate, _ = model_no_gate(x)
    # Output length should match input as there is no downsampling
    print(f"Actual final output shape: {y_no_gate.shape}")
    assert y_no_gate.shape == (batch, length, HIDDEN_DIM)
    print("✅ Gate disabled forward pass successful! (Returns last layer output)")
    
    # Verification 3: Downsampling Enabled (First layer non-downsampling)
    print("\n--- Verification 3: Downsampling Enabled (First Layer Non-downsampling, Gate Enabled) ---")
    num_layers_ds = 4 
    
    # Helper: calculate final seq length
    def get_final_len(l_in, total_layers):
        if total_layers < 1: return l_in
        l_out = l_in # First layer length remains unchanged
        num_ds_layers = total_layers - 1
        for _ in range(num_ds_layers):
            # 1D Conv output length for kernel=3, stride=2, padding=1
            l_out = math.floor((l_out - 1) / 2 + 1) 
        return l_out
    
    expected_len = get_final_len(length, num_layers_ds)

    model_downsample = MambaMoEStackGateModel(
        hidden_size=HIDDEN_DIM, num_hidden_layers=num_layers_ds,
        use_moe=False,
        gate_seq_len=gate_len,
        query_source='token',
        use_downsampling=True,
        use_gate=True,
    ).to(device)

    assert not model_downsample.layers[0].use_downsampling, "First layer should not downsample!"
    assert model_downsample.layers[1].use_downsampling, "Second layer should downsample!"
    print("✅ Layer configuration correct (1st non-downsampling, subsequent downsampling).")

    y_down, _ = model_downsample(x)
    print(f"Input seq length: {length}, via {num_layers_ds} layers (1 non-downsampling + {num_layers_ds-1} downsampling)")
    print(f"Expected final length: {gate_len} (Gate Token length)")
    print(f"Actual final output shape: {y_down.shape}")
    assert y_down.shape == (batch, gate_len, HIDDEN_DIM)
    print("✅ Pyramid downsampling + Gate forward pass successful!")

    # Verification 4: Dynamic Layer Limit
    print("\n--- Verification 4: Dynamic Layer Limit (Gate Enabled) ---")
    
    input_len_short = 32 
    requested_layers_too_many = 8 
    x_short = torch.randn(batch, input_len_short, INPUT_DIM).to(device)
    
    # Theoretically length 32 supports 1 + floor(log2(32)) = 6 layers
    max_layers_expected = 1 + math.floor(math.log2(input_len_short))
    
    model_layer_limit = MambaMoEStackGateModel(
        hidden_size=HIDDEN_DIM, num_hidden_layers=requested_layers_too_many,
        use_moe=False,
        gate_seq_len=gate_len,
        query_source='token',
        use_downsampling=True,
        use_gate=True,
    ).to(device)

    # Should trigger a warning
    y_limit, _ = model_layer_limit(x_short)
    
    print(f"Actual final output shape: {y_limit.shape}")
    assert y_limit.shape == (batch, gate_len, HIDDEN_DIM)
    print("✅ Dynamic layer limit successful, output shape matches expectation (Gate Token)!")

    # Verification 5: Gate Disabled + Dynamic Layer Limit
    print("\n--- Verification 5: Gate Disabled + Dynamic Layer Limit ---")
    
    final_len_expected = get_final_len(input_len_short, max_layers_expected)
    
    model_layer_limit_no_gate = MambaMoEStackGateModel(
        hidden_size=HIDDEN_DIM, num_hidden_layers=requested_layers_too_many,
        use_moe=False,
        query_source='last_layer',
        use_downsampling=True,
        use_gate=False, 
    ).to(device)

    y_limit_no_gate, _ = model_layer_limit_no_gate(x_short)
    
    print(f"Actual final output shape: {y_limit_no_gate.shape}")
    # Matches the actual running last layer sequence length
    assert y_limit_no_gate.shape == (batch, final_len_expected, HIDDEN_DIM)
    print("✅ Disable Gate + Dynamic Layer Limit successful! (Matches last layer seq length)")

    # Verification 6: Main Parameters
    print("\n--- Verification 6: Main Parameters ---")
    model_layer_nolimit = MambaMoEStackGateModel(
        hidden_size=HIDDEN_DIM, num_hidden_layers=requested_layers_too_many,
        num_heads=8, num_key_value_heads=4, # GQA config
        use_moe=True, gate_seq_len=gate_len, 
        num_gate_layers=2, 
        query_source='last_layer',
        gate_type='rope', 
        use_downsampling=True,
        use_gate=True, 
    ).to(device)
    
    y_nolimit, _ = model_layer_nolimit(x_short)
    print(f"Actual final output shape: {y_nolimit.shape}")

    print(model_layer_limit_no_gate)