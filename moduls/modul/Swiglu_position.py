import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Components: RMSNorm and Activation Function Dictionary ---

# Activation function dictionary
ACT2FN = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}

class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable weight for normalization, applied to the feature dimension (dim)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate reciprocal of the root mean square: rsqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Ensure calculation precision in float32, then restore the result to the original data type
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# --- 1. Embedding Layer: TimeMoeInputEmbedding ---

class TimeMoeInputEmbedding(nn.Module):
    """
    Embeds time series using a GLU-like MLP layer.
    Topology: act_fn(Gate(x)) * Embedding(x)
    """
    def __init__(self, input_size: int, hidden_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear layer 1: Used to generate the gating signal
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        # Linear layer 2: Used to generate the main embedding
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        
        # Activation function
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unknown activation function: {hidden_act}")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # GLU-like structure
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb

# --- 2. Main Module: InputProcessor ---

class InputProcessor(nn.Module):
    """
    Implements the following topology: 
    Input (B, T, F) 
    -> [Optional Pre-RMSNorm(F)] 
    -> Embedding (GLU) [B, T, H] 
    -> [Optional Add Positional Encoding + Norm(H)] 
    -> [Optional Post-RMSNorm(H)] 
    -> Output
    """
    def __init__(
        self,
        input_size: int,  # Input dimension
        hidden_size: int, # Output dimension
        max_position_embeddings: int,
        hidden_act: str = "silu",
        norm_eps: float = 1e-6,
        
        # Topology control parameters
        use_pre_norm: bool = False,         # Controls whether to use RMSNorm(x) on input
        use_pos_embedding: bool = True,     # Controls whether to use Positional Encoding
        use_post_norm: bool = False,        # Controls whether to use final RMSNorm
        normalize_pos_emb: bool = True,     # Controls whether to normalize the positional encoding

        # **[Modification A]** New parameter: Controls the type of Embedding
        embedding_type: str = "timemoe", # Options: "timemoe" (TimeMoeInputEmbedding), "linear" (nn.Linear)
    ):
        super().__init__()
        
        self.use_pre_norm = use_pre_norm
        self.use_pos_embedding = use_pos_embedding
        self.use_post_norm = use_post_norm
        self.normalize_pos_emb = normalize_pos_emb
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        
        # --- 1. Optional RMSNorm (Pre-Norm) ---
        # Applied to the input feature dimension F (input_size)
        if self.use_pre_norm:
            self.pre_norm = RMSNorm(input_size, eps=norm_eps)

        # ------------------- 2. Embedding Layer (Core Modification) -------------------
        # **[Modification B]** Initialize different Embedding modules based on embedding_type
        self.embedding_type = embedding_type
        if embedding_type == "timemoe":
            # Option 1: Use TimeMoeInputEmbedding (original complex structure)
            self.input_embedding = TimeMoeInputEmbedding(
                input_size=input_size, 
                hidden_size=hidden_size, 
                hidden_act=hidden_act
            )
        elif embedding_type == "linear":
            # Option 2: Use simple linear layer mapping (nn.Linear)
            # Implements a simple F -> H mapping
            self.input_embedding = nn.Linear(input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")
        # -------------------------------------------------------------------

        # --- 3. Learnable Positional Encoding ---
        if self.use_pos_embedding:
            # Learnable parameters of shape T x hidden_size
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, hidden_size
            )
            
            # Controllable positional encoding normalization
            if self.normalize_pos_emb:
                # Applied to the hidden_size dimension
                self.pos_norm = RMSNorm(hidden_size, eps=norm_eps)

        # --- 4. Optional RMSNorm (Post-Norm) ---
        # Applied to the hidden_size dimension
        if self.use_post_norm:
            self.post_norm = RMSNorm(hidden_size, eps=norm_eps)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, F)
        Returns:
            torch.Tensor: Output tensor (B, T, hidden_size)
        """
        
        B, T, F = x.size()

        # ----------------- Stage 1: Pre-Norm -----------------
        if self.use_pre_norm:
            x = self.pre_norm(x)

        # ----------------- Stage 2: Embedding -----------------
        hidden_states = self.input_embedding(x) # [B, T, H]

        # ----------------- Stage 3: Add Positional Encoding -----------------
        if self.use_pos_embedding:
            # Sequence length truncation handling
            T_actual = min(T, self.max_position_embeddings)
            
            if T > self.max_position_embeddings:
                # Truncate sequence
                hidden_states = hidden_states[:, :T_actual, :]
            
            # Generate position indices
            position_ids = torch.arange(T_actual, dtype=torch.long, device=x.device).unsqueeze(0)
            
            # Retrieve positional embeddings [1, T_actual, H]
            position_embeddings = self.position_embeddings(position_ids)

            # Controllable positional encoding normalization
            if self.normalize_pos_emb:
                position_embeddings = self.pos_norm(position_embeddings)

            # Addition
            hidden_states = hidden_states + position_embeddings

        # ----------------- Stage 4: Post-Norm -----------------
        if self.use_post_norm:
            hidden_states = self.post_norm(hidden_states)

        return hidden_states

# --- Example Usage ---
if __name__ == '__main__':
    # Define parameters
    INPUT_F = 10        # Input feature dimension F
    HIDDEN_H = 64       # Hidden dimension hidden_size
    MAX_POS_T = 128     # Maximum sequence length T
    BATCH_B = 4
    SEQ_L = 32          # Current input sequence length T

    # Simulate input data (B, T, F)
    input_data = torch.randn(BATCH_B, SEQ_L, INPUT_F)

    # ----------------------------------------------------
    # **Scenario: Implementing the recommended complete topology**
    # RMSNorm(x) -> Embedding -> Positional Encoding + Normalization -> RMSNorm
    # ----------------------------------------------------
    print("--- Scenario: Recommended Complete Topology ---")
    processor = InputProcessor(
        input_size=INPUT_F,
        hidden_size=HIDDEN_H,
        max_position_embeddings=MAX_POS_T,
        
        use_pre_norm=True,          # Recommended: Use Pre-Norm
        use_pos_embedding=True,     # Use positional encoding
        normalize_pos_emb=True,     # Normalize positional encoding
        use_post_norm=True          # Use Post-Norm
    )

    output = processor(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total model parameters: {sum(p.numel() for p in processor.parameters() if p.requires_grad)}")

    # ----------------------------------------------------
    # **Scenario: Embedding Only + Positional Encoding without Normalization**
    # ----------------------------------------------------
    print("\n--- Scenario: Basic Embedding + Unnormalized Positional Encoding ---")
    processor_basic = InputProcessor(
        input_size=INPUT_F,
        hidden_size=HIDDEN_H,
        max_position_embeddings=MAX_POS_T,
        
        use_pre_norm=False,
        use_pos_embedding=True,
        normalize_pos_emb=False,
        use_post_norm=False
    )
    output_basic = processor_basic(input_data)
    print(f"Output shape: {output_basic.shape}")

    # ----------------------------------------------------
    # **Scenario 2: Using nn.Linear mapping**
    # ----------------------------------------------------
    print("\n--- Scenario 2: Using nn.Linear Mapping ---")
    processor_linear = InputProcessor(
        input_size=INPUT_F,
        hidden_size=HIDDEN_H,
        max_position_embeddings=MAX_POS_T,
        embedding_type="linear", # **Key modification point: Specify linear layer**
        use_pre_norm=False,
        use_pos_embedding=False,
        normalize_pos_emb=False,
        use_post_norm=True
    )
    output_linear = processor_linear(input_data)
    print(f"Output shape: {output_linear.shape}")
    print(processor_linear)

    from torchinfo import summary
    # Input tensor shape for the model (B, T, F)
    INPUT_SIZE = (4, 32, 10) 

    print("--- Detailed Model Structure and Parameter Statistics (Using torchinfo) ---")
    summary(
        processor_linear, 
        input_size=INPUT_SIZE,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        row_settings=["var_names"]
    )


    